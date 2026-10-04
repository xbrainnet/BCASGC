from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(42)
    subjects, nodes, time_points = 40, 90, 240

    labels = np.repeat(np.arange(2), subjects // 2).astype(np.int64)
    fmri = rng.normal(0, 1, size=(subjects, nodes, time_points)).astype(np.float32)

    # Add a weak class-dependent signal so the complete classification path is exercised.
    fmri[labels == 1, :8, 40:100] += 0.35

    dti = rng.uniform(0, 1, size=(subjects, nodes, nodes)).astype(np.float32)
    dti = (dti + dti.transpose(0, 2, 1)) / 2
    dti = (dti > 0.75).astype(np.float32)
    diagonal = np.arange(nodes)
    dti[:, diagonal, diagonal] = 0

    output = Path("data/random_brain_data.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, fmri=fmri, dti=dti, labels=labels)
    print(f"Saved {output.resolve()}")
    print(f"fMRI: {fmri.shape}, DTI: {dti.shape}, labels: {labels.shape}")


if __name__ == "__main__":
    main()

