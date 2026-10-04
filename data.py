from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


@dataclass
class BrainArrays:
    fmri: np.ndarray
    dti: np.ndarray
    labels: np.ndarray


def _pick(mapping, requested: Optional[str], candidates: tuple[str, ...]):
    if requested:
        if requested not in mapping:
            raise KeyError(f"Key '{requested}' not found. Available keys: {sorted(mapping)}")
        return mapping[requested]
    for key in candidates:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Could not infer a key from {candidates}. Available keys: {sorted(mapping)}")


def _canonical_fmri(x: np.ndarray, nodes: int = 90) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 3:
        raise ValueError(f"fMRI must be 3-D, received {x.shape}")
    node_axes = [i for i, n in enumerate(x.shape) if n == nodes]
    if not node_axes:
        raise ValueError(f"No ROI axis of length {nodes} in fMRI shape {x.shape}")
    node_axis = node_axes[0]
    remaining = [i for i in range(3) if i != node_axis]
    time_axis = max(remaining, key=lambda i: x.shape[i])
    batch_axis = next(i for i in remaining if i != time_axis)
    return np.transpose(x, (batch_axis, node_axis, time_axis)).astype(np.float32, copy=False)


def _canonical_dti(x: np.ndarray, batch: int, nodes: int = 90) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim == 2:
        if x.shape != (nodes, nodes):
            raise ValueError(f"A shared DTI matrix must be ({nodes}, {nodes}), received {x.shape}")
        x = np.repeat(x[None], batch, axis=0)
    elif x.ndim == 3:
        axes = list(range(3))
        roi_axes = [i for i in axes if x.shape[i] == nodes]
        if len(roi_axes) < 2:
            raise ValueError(f"DTI must contain two ROI axes of length {nodes}, received {x.shape}")
        batch_axis = next(i for i in axes if i not in roi_axes[:2])
        x = np.transpose(x, (batch_axis, roi_axes[0], roi_axes[1]))
    else:
        raise ValueError(f"DTI must be 2-D or 3-D, received {x.shape}")
    if x.shape != (batch, nodes, nodes):
        raise ValueError(f"Expected DTI shape ({batch}, {nodes}, {nodes}), received {x.shape}")
    return x.astype(np.float32, copy=False)


def load_brain_arrays(
    path: str | Path,
    fmri_key: Optional[str] = None,
    dti_key: Optional[str] = None,
    label_key: Optional[str] = None,
    nodes: int = 90,
) -> BrainArrays:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as f:
            raw = {k: f[k] for k in f.files}
    elif path.suffix.lower() == ".mat":
        raw = {k: v for k, v in sio.loadmat(path).items() if not k.startswith("__")}
    else:
        raise ValueError("Only .npz and MATLAB v7.2-or-earlier .mat files are supported")

    fmri = _canonical_fmri(_pick(raw, fmri_key, ("fmri", "fMRI", "data", "X", "X_data")), nodes)
    dti = _canonical_dti(_pick(raw, dti_key, ("dti", "DTI", "G", "adj", "A")), len(fmri), nodes)
    labels = np.asarray(_pick(raw, label_key, ("labels", "label", "y", "gnd"))).reshape(-1)
    if len(labels) != len(fmri):
        raise ValueError(f"Label count {len(labels)} does not match batch size {len(fmri)}")
    _, labels = np.unique(labels, return_inverse=True)
    return BrainArrays(fmri=fmri, dti=dti, labels=labels.astype(np.int64))


class BrainDataset(Dataset):
    def __init__(self, arrays: BrainArrays, indices: np.ndarray):
        self.fmri = arrays.fmri[indices]
        self.dti = arrays.dti[indices]
        self.labels = arrays.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.fmri[index].copy()
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True).clip(min=1e-6)
        x = (x - mean) / std

        a = np.nan_to_num(self.dti[index], copy=True)
        a = np.maximum(a, 0.0)
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 0.0)
        positive = a[a > 0]
        if positive.size:
            a /= positive.max().clip(min=1e-6)
        return torch.from_numpy(x), torch.from_numpy(a), torch.tensor(self.labels[index])

