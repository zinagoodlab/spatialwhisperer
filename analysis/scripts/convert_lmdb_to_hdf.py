"""Convert LMDB tile-based dataset to individual HDF files per sample (or batch).

Reads an LMDB database (e.g., Lizard or PanNuke from PathoCellBench),
filters to test-split samples, and stitches tiles back into
images stored as HDF files matching the CRC HDF format:
  - img: (3, H, W) uint8
  - gt_inst: (1, H, W) uint16 — instance segmentation
  - gt_ct: (1, H, W) uint8 — semantic class per pixel

When batch_size > 1, multiple small samples are grouped into one HDF file
by laying them out in a grid, avoiding thousands of tiny files for datasets
like PanNuke.

Usage via Snakemake (snakemake.input/output/params).
"""

import lmdb
import pickle
import io
import h5py
import numpy as np
import csv
import math
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyUnpickler(pickle.Unpickler):
    """Handle numpy version incompatibility (numpy._core → numpy.core)."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


lmdb_path = snakemake.input.lmdb_dir
splits_csv = snakemake.input.splits_csv
output_dir = Path(snakemake.output.output_dir)
sample_list_file = Path(snakemake.output.sample_list)

tile_size = int(snakemake.params.get("tile_size", 224))
# Number of samples to group into a single HDF file (1 = one HDF per sample)
batch_size = int(snakemake.params.get("batch_size", 1))

# Read split info — only keep test samples
splits = {}
with open(splits_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        splits[row["sample_name"]] = row["train_test_val_split"]

test_samples = sorted([k for k, v in splits.items() if v == "test"])
logger.info(f"Found {len(test_samples)} test samples")

# Read all test tiles from LMDB
env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=256)
sample_tiles = {}  # sample_name -> list of (tile_idx, data_dict)

with env.begin() as txn:
    cursor = txn.cursor()
    for key, val in cursor:
        data = NumpyUnpickler(io.BytesIO(val)).load()
        if not isinstance(data, dict):
            continue
        sn = data.get("sample_name", "")
        if sn not in test_samples:
            continue
        # Extract tile index from tile_name (e.g., "crag_11_TILE_42" → 42)
        tile_name = data.get("tile_name", "")
        parts = tile_name.rsplit("_", 1)
        tile_idx = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
        if sn not in sample_tiles:
            sample_tiles[sn] = []
        sample_tiles[sn].append((tile_idx, data))

env.close()

logger.info(f"Loaded tiles for {len(sample_tiles)} test samples")


def stitch_tiles(tiles, tile_size):
    """Stitch a list of (tile_idx, data) into a single image/mask set."""
    n_tiles = len(tiles)
    grid_side = math.isqrt(n_tiles)
    if grid_side * grid_side < n_tiles:
        grid_w = grid_side + 1
        grid_h = math.ceil(n_tiles / grid_w)
    else:
        grid_w = grid_h = grid_side

    H = grid_h * tile_size
    W = grid_w * tile_size

    img = np.zeros((3, H, W), dtype=np.uint8)
    gt_ct = np.zeros((1, H, W), dtype=np.uint8)
    gt_inst = np.zeros((1, H, W), dtype=np.uint16)

    max_inst_id = 0
    for tile_idx, data in tiles:
        row = tile_idx // grid_w
        col = tile_idx % grid_w
        y0 = row * tile_size
        x0 = col * tile_size

        tile_img = data["image"]  # (3, tile_size, tile_size) uint8
        tile_sem = data["semantic_mask"]  # (tile_size, tile_size) uint8
        tile_inst = data["instance_mask"]  # (tile_size, tile_size)

        img[:, y0 : y0 + tile_size, x0 : x0 + tile_size] = tile_img
        gt_ct[0, y0 : y0 + tile_size, x0 : x0 + tile_size] = tile_sem

        # Renumber instance IDs to be globally unique
        tile_inst_copy = tile_inst.copy().astype(np.uint16)
        mask = tile_inst_copy > 0
        tile_inst_copy[mask] += max_inst_id
        if mask.any():
            max_inst_id = tile_inst_copy[mask].max()
        gt_inst[0, y0 : y0 + tile_size, x0 : x0 + tile_size] = tile_inst_copy

    return img, gt_ct, gt_inst


def stitch_samples_into_grid(sample_images, tile_size):
    """Stitch multiple per-sample images (already stitched from tiles) into a larger grid.

    Each sample_image is (img, gt_ct, gt_inst) with shapes (3,H,W), (1,H,W), (1,H,W).
    Returns a single stitched (img, gt_ct, gt_inst) with globally unique instance IDs.
    """
    n = len(sample_images)
    grid_side = math.isqrt(n)
    if grid_side * grid_side < n:
        grid_w = grid_side + 1
        grid_h = math.ceil(n / grid_w)
    else:
        grid_w = grid_h = grid_side

    # All sample images should have the same shape (they come from the same dataset)
    sample_h = max(si[0].shape[1] for si in sample_images)
    sample_w = max(si[0].shape[2] for si in sample_images)

    H = grid_h * sample_h
    W = grid_w * sample_w

    img = np.zeros((3, H, W), dtype=np.uint8)
    gt_ct = np.zeros((1, H, W), dtype=np.uint8)
    gt_inst = np.zeros((1, H, W), dtype=np.uint16)

    max_inst_id = 0
    for idx, (si_img, si_ct, si_inst) in enumerate(sample_images):
        row = idx // grid_w
        col = idx % grid_w
        y0 = row * sample_h
        x0 = col * sample_w
        sh, sw = si_img.shape[1], si_img.shape[2]

        img[:, y0 : y0 + sh, x0 : x0 + sw] = si_img
        gt_ct[0, y0 : y0 + sh, x0 : x0 + sw] = si_ct[0]

        si_inst_copy = si_inst.copy()
        mask = si_inst_copy > 0
        si_inst_copy[mask] += max_inst_id
        if mask.any():
            max_inst_id = si_inst_copy[mask].max()
        gt_inst[0, y0 : y0 + sh, x0 : x0 + sw] = si_inst_copy[0]

    return img, gt_ct, gt_inst


# Convert samples into HDF files
output_dir.mkdir(parents=True, exist_ok=True)
written_names = []

sorted_samples = sorted(sample_tiles.keys())

if batch_size <= 1:
    # One HDF per sample
    for sn in sorted_samples:
        tiles = sample_tiles[sn]
        img, gt_ct, gt_inst = stitch_tiles(tiles, tile_size)

        hdf_path = output_dir / f"{sn}.hdf"
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("img", data=img, compression="gzip")
            f.create_dataset("gt_ct", data=gt_ct, compression="gzip")
            f.create_dataset("gt_inst", data=gt_inst, compression="gzip")

        written_names.append(sn)
        logger.info(
            f"Wrote {hdf_path} ({len(tiles)} tiles, {img.shape[1]}x{img.shape[2]} px)"
        )
else:
    # Batch multiple samples into one HDF
    for batch_start in range(0, len(sorted_samples), batch_size):
        batch_samples = sorted_samples[batch_start : batch_start + batch_size]
        batch_name = f"batch_{batch_start:05d}"

        # Stitch each sample's tiles first
        sample_images = []
        for sn in batch_samples:
            tiles = sample_tiles[sn]
            img, gt_ct, gt_inst = stitch_tiles(tiles, tile_size)
            sample_images.append((img, gt_ct, gt_inst))

        # Stitch samples into a grid
        big_img, big_ct, big_inst = stitch_samples_into_grid(sample_images, tile_size)

        hdf_path = output_dir / f"{batch_name}.hdf"
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("img", data=big_img, compression="gzip")
            f.create_dataset("gt_ct", data=big_ct, compression="gzip")
            f.create_dataset("gt_inst", data=big_inst, compression="gzip")

        written_names.append(batch_name)
        total_tiles = sum(len(sample_tiles[sn]) for sn in batch_samples)
        logger.info(
            f"Wrote {hdf_path} ({len(batch_samples)} samples, {total_tiles} tiles, {big_img.shape[1]}x{big_img.shape[2]} px)"
        )

# Write sample/batch list for downstream rules
with open(sample_list_file, "w") as f:
    for name in written_names:
        f.write(name + "\n")

logger.info(
    f"Conversion complete: {len(written_names)} HDF files written to {output_dir}"
)
