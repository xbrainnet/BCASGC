from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def normalize_adjacency(a: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    if add_self_loops:
        eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype).expand_as(a)
        a = a + eye
    degree = a.sum(dim=-1).clamp_min(1e-8)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)


class DenseGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a_norm, self.linear(x))


class CommunityDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, candidate_ks: tuple[int, ...], dropout: float):
        super().__init__()
        self.gcn1 = DenseGraphConv(input_dim, hidden_dim)
        self.gcn2 = DenseGraphConv(hidden_dim, hidden_dim)
        self.heads = nn.ModuleDict({str(k): nn.Linear(hidden_dim, k) for k in candidate_ks})
        self.dropout = dropout

    def forward(self, x: torch.Tensor, structural_adj: torch.Tensor) -> dict[int, torch.Tensor]:
        a_norm = normalize_adjacency(structural_adj)
        h = F.relu(self.gcn1(x, a_norm))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.gcn2(h, a_norm))
        # Paper Eq. (4) uses ReLU, so affiliation strengths are non-negative
        # without being constrained to sum to one or to lie below one.
        return {int(k): F.relu(head(h)) for k, head in self.heads.items()}


def functional_adjacency(x: torch.Tensor, threshold: float) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    corr = torch.bmm(x, x.transpose(1, 2))
    adj = (corr >= threshold).to(x.dtype)
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype).unsqueeze(0)
    return adj * (1.0 - eye)


def community_score(
    affiliation: torch.Tensor,
    structural_adj: torch.Tensor,
    functional_adj: torch.Tensor,
    rho: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    z = affiliation > rho
    shared = torch.bmm(z.float(), z.float().transpose(1, 2)) > 0
    edge = structural_adj > 0
    eye = torch.eye(edge.size(-1), device=edge.device, dtype=torch.bool).unsqueeze(0)
    edge = edge & ~eye
    coverage = (shared & edge).sum((1, 2)).float() / edge.sum((1, 2)).clamp_min(1)

    scores_cc, scores_con, sizes = [], [], []
    for j in range(z.size(-1)):
        member = z[:, :, j]
        size = member.sum(1).float()
        pair = member.unsqueeze(2) & member.unsqueeze(1)
        subgraph = edge.to(affiliation.dtype) * pair.to(affiliation.dtype)
        triangles = (torch.bmm(subgraph, subgraph) * subgraph).sum((1, 2)) / 6.0
        possible_triangles = (size * (size - 1) * (size - 2) / 6.0).clamp_min(1)
        clustering = torch.where(size >= 3, triangles / possible_triangles, torch.zeros_like(size))

        inside_f = (functional_adj.bool() & pair).sum((1, 2)).float()
        outside_pair = member.unsqueeze(2) & (~member).unsqueeze(1)
        outside_f = (functional_adj.bool() & outside_pair).sum((1, 2)).float()
        conductance = outside_f / (inside_f + outside_f).clamp_min(1)
        scores_cc.append(clustering)
        scores_con.append(conductance)
        sizes.append(size)
    sizes = torch.stack(sizes, 1)
    weights = sizes / sizes.sum(1, keepdim=True).clamp_min(1)
    avg_cc = (torch.stack(scores_cc, 1) * weights).sum(1)
    avg_con = (torch.stack(scores_con, 1) * weights).sum(1)
    return coverage + alpha * avg_cc - beta * avg_con


def build_message_matrix(affiliation: torch.Tensor, rho: float) -> torch.Tensor:
    thresholded = affiliation * (affiliation > rho)
    binary = (thresholded > 0).to(affiliation.dtype)
    shared = torch.bmm(binary, binary.transpose(1, 2)) > 0
    # E[u,v] = sum of source-u affiliations over communities shared by u and v.
    directed_strength = torch.bmm(thresholded, binary.transpose(1, 2))
    weighted = shared.to(affiliation.dtype) * directed_strength
    return normalize_adjacency(weighted, add_self_loops=True)


class AdaptiveVariableGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, message_matrix: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.bmm(message_matrix, self.linear(x)))


@dataclass
class ASVGCNOutput:
    logits: torch.Tensor
    affiliations: dict[int, torch.Tensor]
    selected_k: torch.Tensor
    selected_affiliation: list[torch.Tensor]
    scores: torch.Tensor


class ASVGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 240,
        hidden_dim: int = 64,
        graph_dim: int = 64,
        min_k: int = 3,
        max_k: int = 16,
        rho: float = 0.5,
        alpha: float = 0.5,
        beta: float = 1.8,
        fc_threshold: float = 0.5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.candidate_ks = tuple(range(min_k, max_k + 1))
        self.rho, self.alpha, self.beta = rho, alpha, beta
        self.fc_threshold, self.dropout = fc_threshold, dropout
        self.detector = CommunityDetector(input_dim, hidden_dim, self.candidate_ks, dropout)
        self.vgcn1 = AdaptiveVariableGraphConv(input_dim, graph_dim)
        self.vgcn2 = AdaptiveVariableGraphConv(graph_dim, graph_dim)
        self.classifier = nn.Linear(graph_dim, num_classes)

    def forward(self, fmri: torch.Tensor, dti: torch.Tensor) -> ASVGCNOutput:
        affiliations = self.detector(fmri, dti)
        f_adj = functional_adjacency(fmri, self.fc_threshold)
        score_list = [community_score(affiliations[k], dti, f_adj, self.rho, self.alpha, self.beta) for k in self.candidate_ks]
        scores = torch.stack(score_list, dim=1)
        selected_idx = scores.argmax(dim=1)

        pooled, selected_affiliation = [], []
        for b in range(fmri.size(0)):
            k = self.candidate_ks[int(selected_idx[b])]
            affiliation = affiliations[k][b : b + 1]
            message = build_message_matrix(affiliation, self.rho)
            h = self.vgcn1(fmri[b : b + 1], message)
            h = F.dropout(h, self.dropout, self.training)
            h = self.vgcn2(h, message)
            pooled.append(h.mean(dim=1))
            selected_affiliation.append(affiliation.squeeze(0))
        logits = self.classifier(torch.cat(pooled, dim=0))
        selected_k = torch.tensor(self.candidate_ks, device=fmri.device)[selected_idx]
        return ASVGCNOutput(logits, affiliations, selected_k, selected_affiliation, scores)

fmri = torch.randn(4, 90, 240)

dti = torch.rand(4, 90, 90)
dti = (dti + dti.transpose(1, 2)) / 2

model = ASVGCN(
    num_classes=2,
    input_dim=240,
    hidden_dim=64,
    graph_dim=64,
    min_k=3,
    max_k=16,
)

model.eval()

with torch.no_grad():
    output = model(fmri, dti)

print(output.logits.shape)
