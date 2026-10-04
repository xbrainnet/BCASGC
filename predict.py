from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from asvgcn.data import BrainDataset, load_brain_arrays
except ImportError:
    from data import BrainDataset, load_brain_arrays
from train import make_model


def main():
    parser = argparse.ArgumentParser(description="Run ASVGCN inference from one fold checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="predictions.npz")
    parser.add_argument("--fmri-key")
    parser.add_argument("--dti-key")
    parser.add_argument("--label-key")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    arrays = load_brain_arrays(args.data, args.fmri_key, args.dti_key, args.label_key, config["nodes"])
    model = make_model(
        config,
        len(checkpoint["label_values"]),
        arrays.fmri.shape[-1],
        checkpoint["best_alpha"],
        checkpoint["best_beta"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = BrainDataset(arrays, np.arange(len(arrays.labels)))
    probabilities, selected_k = [], []
    with torch.no_grad():
        for fmri, dti, _ in torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"]):
            output = model(fmri, dti)
            probabilities.append(output.logits.softmax(1).numpy())
            selected_k.append(output.selected_k.numpy())
    np.savez_compressed(args.output, probabilities=np.concatenate(probabilities), selected_k=np.concatenate(selected_k))
    print(json.dumps({"output": str(Path(args.output).resolve()), "subjects": len(dataset)}))


if __name__ == "__main__":
    main()
