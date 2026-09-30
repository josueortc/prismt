"""Convert the lab's atlas masks (.npy, not in git) into the small asset the MATLAB app uses.

    python tools/convert_atlas_npy.py [grid_values.npy] [mask_atlas_new.npy]

grid_values.npy: 82 x 256 x 256 binary tiles, a schematic grid of 41 regions x 2 hemispheres;
tiles 2k and 2k+1 (0-based) are the left and right tile of region k. They are shown the way
the pre-rebuild plots showed them: transposed, then cropped to [30:-42, 24:-24] (184 x 208).
mask_atlas_new.npy: 184 x 208 x 52 binary Allen-atlas parcels (left/right interleaved); some
pixels belong to several parcels, so they are kept as a stack of masks.

Writes matlab/+prismt/+atlas/private/atlases.mat (checked into git) with provenance.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(grid_path: Path, allen_path: Path) -> Path:
    g = np.load(grid_path) > 0.5
    a = np.load(allen_path) > 0.5
    if g.shape != (82, 256, 256) or (g.sum(0) > 1).any():
        raise SystemExit(f"{grid_path}: expected 82 non-overlapping 256x256 tiles")
    labels = np.zeros((256, 256), dtype=np.uint16)
    for k in range(82):
        labels[g[k]] = k + 1
    labels = labels.T[30:-42, 24:-24]
    missing = sorted(set(range(1, 83)) - set(np.unique(labels).tolist()))
    if missing:
        raise SystemExit(f"crop removed tiles {missing}")
    side = np.array(["L" if k % 2 == 0 else "R" for k in range(82)], dtype=object)
    x = np.array([np.nonzero(labels == k + 1)[1].mean() + 1 for k in range(82)])
    y = np.array([np.nonzero(labels == k + 1)[0].mean() + 1 for k in range(82)])
    left_x = x[0::2].mean()
    if left_x > x[1::2].mean():
        side = np.where(side == "L", "R", "L").astype(object)
    out = ROOT / "matlab" / "+prismt" / "+atlas" / "private" / "atlases.mat"
    savemat(out, {
        "grid82_labels": labels,
        "grid82_hemisphere": side,
        "grid82_x": x, "grid82_y": y,
        "allen52_masks": a.astype(np.uint8),
        "provenance": {"grid82_source_sha256": sha(grid_path), "allen52_source_sha256": sha(allen_path),
                       "grid82_transform": "labels(tile) then transpose then crop [30:-42, 24:-24]",
                       "allen52_empty_parcels_1based": [21, 22, 49, 50]},
    }, do_compression=True)
    return out


if __name__ == "__main__":
    grid = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "grid_values.npy"
    allen = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "mask_atlas_new.npy"
    print(main(grid, allen))
