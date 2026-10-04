import torch
import torch.nn.functional as F


def bernoulli_poisson_loss(affiliation: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    logits = torch.bmm(affiliation, affiliation.transpose(1, 2)) + 1e-6
    n = logits.size(-1)
    eye = torch.eye(n, device=logits.device, dtype=torch.bool).unsqueeze(0)
    edges = (adjacency > 0) & ~eye
    nonedges = (~edges) & ~eye
    edge_probability = (-torch.expm1(-logits)).clamp_min(1e-8)
    edge_loss = -torch.log(edge_probability)
    nonedge_loss = logits
    per_graph_edges = (edge_loss * edges).sum((1, 2)) / edges.sum((1, 2)).clamp_min(1)
    per_graph_nonedges = (nonedge_loss * nonedges).sum((1, 2)) / nonedges.sum((1, 2)).clamp_min(1)
    return 0.5 * (per_graph_edges + per_graph_nonedges).mean()


def total_loss(output, labels, adjacency, community_weight: float):
    classification = F.cross_entropy(output.logits, labels)
    community = torch.stack([
        bernoulli_poisson_loss(affiliation, adjacency)
        for affiliation in output.affiliations.values()
    ]).mean()
    return classification + community_weight * community, classification.detach(), community.detach()
