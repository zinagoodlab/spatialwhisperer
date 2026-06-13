"""Build a mapping from (batch_name, patch_idx) -> (sample_name, tissue_type) for PanNuke.

Reconstructs the deterministic batching and grid layout used by convert_lmdb_to_hdf.py
to trace each processed patch back to its original PanNuke sample and tissue type.

Each PanNuke test sample has 4 tiles (2x2 of 224x224 = 448x448 per sample).
Samples are sorted alphabetically and grouped into batches of 50.
Within each batch, samples are laid out in a grid (row-major order).

Inputs (Snakemake):
  - snakemake.input.splits_csv: train_test_val_split.csv
  - snakemake.input.types_npy: PanNuke fold2 types.npy
  - snakemake.input.adatas: list of processed h5ad files (for patch coordinates)

Outputs:
  - snakemake.output.mapping: CSV with columns: batch_name, patch_idx, sample_name, tissue_type
"""

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

BATCH_SIZE = 50
TILE_SIZE = 224
TILES_PER_SAMPLE = 4  # 2x2 grid


def grid_dims(n):
    """Reproduce the grid layout from convert_lmdb_to_hdf.py."""
    grid_side = math.isqrt(n)
    if grid_side * grid_side < n:
        grid_w = grid_side + 1
        grid_h = math.ceil(n / grid_w)
    else:
        grid_w = grid_h = grid_side
    return grid_h, grid_w


# Load splits -> get sorted test sample names
splits = {}
with open(snakemake.input.splits_csv) as f:
    for row in csv.DictReader(f):
        splits[row["sample_name"]] = row["train_test_val_split"]
test_samples = sorted(k for k, v in splits.items() if v == "test")

# Load tissue types for fold2
types = np.load(snakemake.input.types_npy)

# Build sample_name -> tissue_type mapping
# sample_name is "fold2_{local_idx}" where local_idx indexes into types array
sample_to_tissue = {}
for sn in test_samples:
    fold, local_idx = sn.split("_", 1)
    local_idx = int(local_idx)
    sample_to_tissue[sn] = str(types[local_idx])

# Process each batch
adata_fps = sorted(Path(p) for p in snakemake.input.adatas)

rows = []
for adata_fp in adata_fps:
    batch_name = adata_fp.stem.replace("_patch", "")
    # Determine batch_start from name (e.g., "batch_00050" -> 50)
    batch_start = int(batch_name.split("_")[1])
    batch_end = min(batch_start + BATCH_SIZE, len(test_samples))
    batch_samples = test_samples[batch_start:batch_end]
    n_samples = len(batch_samples)

    # Grid layout for samples within this batch
    _, grid_w_samples = grid_dims(n_samples)

    # Each sample is 2x2 tiles. Tile grid for 4 tiles:
    _, tile_grid_w = grid_dims(TILES_PER_SAMPLE)  # = 2

    # Sample occupies tile_grid_h * TILE_SIZE rows and tile_grid_w * TILE_SIZE cols
    sample_h_tiles = 2  # tiles in y
    sample_w_tiles = 2  # tiles in x

    # Load h5ad to get actual patch coordinates
    adata = ad.read_h5ad(adata_fp)

    for i in range(adata.n_obs):
        obs = adata.obs.iloc[i]
        x_arr = int(obs["x_array"])
        y_arr = int(obs["y_array"])

        # Map array coords to sample index in this batch
        s_col = x_arr // sample_w_tiles
        s_row = y_arr // sample_h_tiles
        s_idx = s_row * grid_w_samples + s_col

        if s_idx < n_samples:
            sample_name = batch_samples[s_idx]
            tissue = sample_to_tissue[sample_name]
        else:
            # Grid padding position (no sample here)
            sample_name = None
            tissue = None

        rows.append(
            {
                "batch_name": batch_name,
                "patch_idx": adata.obs_names[i],
                "x_array": x_arr,
                "y_array": y_arr,
                "sample_name": sample_name,
                "tissue_type": tissue,
            }
        )

df = pd.DataFrame(rows)
# Drop patches in padding positions (shouldn't exist since the h5ad only has non-empty patches)
df = df.dropna(subset=["sample_name"])

out = Path(snakemake.output.mapping)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(
    f"Mapping: {len(df)} patches from {df['sample_name'].nunique()} samples across {df['tissue_type'].nunique()} tissue types"
)
print(f"Tissue type distribution:\n{df['tissue_type'].value_counts().to_string()}")
