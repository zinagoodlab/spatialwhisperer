"""Score PanNuke patches for tissue type prediction via zero-shot image-text similarity.

For each batch, loads the model and computes similarity scores between patch image
embeddings and text embeddings of the 19 PanNuke tissue type names.

Text queries are the bare tissue names (lowercase, underscores replaced with spaces).

Inputs (Snakemake):
  - snakemake.input.model: model checkpoint path
  - snakemake.input.adata: h5ad file for one batch
  - snakemake.input.image: TIFF image for the batch

Outputs:
  - snakemake.output.scores: CSV of raw logit scores (n_patches x 19 tissue types)
"""

import pyarrow  # must be first (glibc workaround)

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import torch

from cellwhisperer.utils.inference import score_left_vs_right
from cellwhisperer.utils.model_io import load_cellwhisperer_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PanNuke tissue types (19). Names derived from types.npy values, lowercased,
# with underscores/hyphens replaced by spaces. Must match normalization in
# compute_tissue_type_metrics.py.
TISSUE_TYPES = [
    "adrenal gland",
    "bile duct",
    "bladder",
    "breast",
    "cervix",
    "colon",
    "esophagus",
    "head and neck",
    "kidney",
    "liver",
    "lung",
    "ovarian",
    "pancreatic",
    "prostate",
    "skin",
    "stomach",
    "testis",
    "thyroid",
    "uterus",
]

pl_model, _, _, _ = load_cellwhisperer_model(
    model_path=snakemake.input.model, eval=True
)
model = pl_model.model

adata = ad.read_h5ad(snakemake.input.adata)
# Attach the image path so the model can load image data
adata.uns["image_path"] = str(snakemake.input.image)

logger.info(
    "Scoring %d patches against %d tissue types", adata.n_obs, len(TISSUE_TYPES)
)

scores, _ = score_left_vs_right(
    left_input=adata,
    right_input=TISSUE_TYPES,
    model=model,
    logit_scale=model.discriminator.temperature.exp(),
    average_mode=None,
    grouping_keys=None,
    batch_size=128,
    score_norm_method=None,
    use_image_data=True,
)

# scores shape: (n_tissue_types, n_patches) -> transpose to (n_patches, n_tissue_types)
scores_np = scores.T.cpu().numpy()
scores_df = pd.DataFrame(scores_np, columns=TISSUE_TYPES)
scores_df.to_csv(snakemake.output.scores, index=False)
logger.info("Saved tissue type scores to %s", snakemake.output.scores)
