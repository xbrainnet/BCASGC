from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from tqdm import trange

try:
    from asvgcn.data import BrainDataset, load_brain_arrays
    from asvgcn.losses import total_loss
    from asvgcn.metrics import classification_metrics
    from asvgcn.model import ASVGCN
except ImportError:
    from data import BrainDataset, load_brain_arrays
    from losses import total_loss
    from metrics import classification_metrics
    from model import ASVGCN


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, selected_ks = [], [], []
    for fmri, dti, y in loader:
        output = model(fmri.to(device), dti.to(device))
        probabilities.append(output.logits.softmax(1).cpu().numpy())
        labels.append(y.numpy())
        selected_ks.append(output.selected_k.cpu().numpy())
    y_true = np.concatenate(labels)
    prob = np.concatenate(probabilities)
    metrics = classification_metrics(y_true, prob)
    return metrics, y_true, prob, np.concatenate(selected_ks)


def make_model(config, num_classes, input_dim, alpha, beta):
    return ASVGCN(
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        graph_dim=config["graph_dim"],
        min_k=config["min_k"],
        max_k=config["max_k"],
        rho=config["rho"],
        alpha=alpha,
        beta=beta,
        fc_threshold=config["fc_threshold"],
        dropout=config["dropout"],
    )


def train_candidate(config, arrays, fold, train_dataset, val_loader, device, alpha, beta):
    # Reset the random state so all grid candidates use the same initialization
    # and mini-batch order within the current outer fold.
    candidate_seed = config["seed"] + fold
    seed_everything(candidate_seed)
    generator = torch.Generator().manual_seed(candidate_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )
    model = make_model(
        config,
        len(np.unique(arrays.labels)),
        arrays.fmri.shape[-1],
        alpha,
        beta,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best_state = None
    best_score = -np.inf
    best_epoch = 0
    best_metrics = None
    bad_epochs = 0
    selection_metric = config.get("selection_metric", "ACC")

    for epoch in trange(
        config["epochs"],
        desc=f"fold {fold}: alpha={alpha}, beta={beta}",
        leave=False,
    ):
        model.train()
        for fmri, dti, y in train_loader:
            fmri, dti, y = fmri.to(device), dti.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(fmri, dti)
            loss, _, _ = total_loss(output, y, dti, config["community_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        score = val_metrics[selection_metric]
        if score > best_score + 1e-6:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_metrics = val_metrics
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid validation checkpoint")
    return best_state, best_score, best_epoch, best_metrics


def run_fold(config, arrays, fold, train_val_idx, test_idx, device, output_dir):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=config["validation_fraction"], random_state=config["seed"] + fold
    )
    relative_train, relative_val = next(splitter.split(train_val_idx, arrays.labels[train_val_idx]))
    train_idx, val_idx = train_val_idx[relative_train], train_val_idx[relative_val]
    train_dataset = BrainDataset(arrays, train_idx)
    val_loader = DataLoader(
        BrainDataset(arrays, val_idx),
        batch_size=config["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        BrainDataset(arrays, test_idx),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    best_state = None
    best_val = -np.inf
    best_epoch = 0
    best_alpha = None
    best_beta = None
    grid_rows = []

    for alpha, beta in product(config["alpha_grid"], config["beta_grid"]):
        state, score, epoch, val_metrics = train_candidate(
            config,
            arrays,
            fold,
            train_dataset,
            val_loader,
            device,
            alpha,
            beta,
        )
        grid_rows.append({
            "fold": fold,
            "alpha": alpha,
            "beta": beta,
            "best_epoch": epoch,
            "selection_metric": config.get("selection_metric", "ACC"),
            "selection_score": score,
            "val_ACC": val_metrics["ACC"],
            "val_AUC": val_metrics["AUC"],
            "val_SPE": val_metrics["SPE"],
            "val_SEN": val_metrics["SEN"],
        })
        if score > best_val + 1e-6:
            best_val = score
            best_state = state
            best_epoch = epoch
            best_alpha = alpha
            best_beta = beta

    with (output_dir / f"fold_{fold:02d}_grid_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=grid_rows[0].keys())
        writer.writeheader()
        writer.writerows(grid_rows)

    model = make_model(
        config,
        len(np.unique(arrays.labels)),
        arrays.fmri.shape[-1],
        best_alpha,
        best_beta,
    ).to(device)
    model.load_state_dict(best_state)
    # The test set is evaluated only once, after grid-search selection is complete.
    test_metrics, y_true, prob, selected_k = evaluate(model, test_loader, device)
    checkpoint = {
        "model_state": best_state,
        "config": config,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "best_validation_score": best_val,
        "label_values": np.unique(arrays.labels).tolist(),
    }
    torch.save(checkpoint, output_dir / f"fold_{fold:02d}.pt")
    np.savez_compressed(
        output_dir / f"fold_{fold:02d}_predictions.npz",
        indices=test_idx, labels=y_true, probabilities=prob, selected_k=selected_k,
    )
    return {
        "fold": fold,
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "best_epoch": best_epoch,
        "validation_score": best_val,
        **test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train cognition-driven ASVGCN with nested validation")
    parser.add_argument("--config", required=True, help="JSON configuration path")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    requested_device = config.get("device", "cuda")
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    arrays = load_brain_arrays(
        config["data"], config.get("fmri_key"), config.get("dti_key"), config.get("label_key"), config["nodes"]
    )
    if arrays.fmri.shape[1:] != (config["nodes"], config["time_points"]):
        raise ValueError(f"Expected fMRI [B,{config['nodes']},{config['time_points']}], got {arrays.fmri.shape}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    cv = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    rows = []
    all_idx = np.arange(len(arrays.labels))
    for fold, (train_val_idx, test_idx) in enumerate(cv.split(all_idx, arrays.labels), start=1):
        rows.append(run_fold(config, arrays, fold, train_val_idx, test_idx, device, output_dir))
        with (output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    summary = {metric: {"mean": float(np.mean([r[metric] for r in rows])), "std": float(np.std([r[metric] for r in rows], ddof=1))} for metric in ("ACC", "AUC", "SPE", "SEN")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
