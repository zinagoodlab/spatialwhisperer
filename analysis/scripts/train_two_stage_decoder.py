"""
Train the image→expression decoder for the two-stage baseline.

Uses JointEmbedDataModule to get image patches, but replaces expression_expr
in each batch with our custom 17,851-gene target computed from the same data.

The key challenge is that JointEmbedDataModule caches preprocessed data with
the default cosmx6k gene list. We work around this by:
1. Using the datamodule normally for image patches (gene-independent)
2. Overriding expression_expr in each batch via a collate wrapper
"""

import pyarrow  # needed first

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gene_list_path = snakemake.input.gene_list
uni2_weights_dir = snakemake.input.uni2_weights
output_ckpt = snakemake.output.decoder_ckpt
max_epochs = snakemake.params.max_epochs
learning_rate = snakemake.params.learning_rate
batch_size = snakemake.params.batch_size

# ── Load gene list ───────────────────────────────────────────────────────────
gene_df = pd.read_csv(gene_list_path)
target_genes = gene_df["gene_name"].tolist()
n_target_genes = len(target_genes)
logger.info(f"Target gene list: {n_target_genes} genes")

# ── Set up datamodule ────────────────────────────────────────────────────────
from cellwhisperer.jointemb.dataset import JointEmbedDataModule
from cellwhisperer.jointemb.mlp_model import MLPTranscriptomeProcessor

# Create a custom MLP processor with our gene list BEFORE constructing the datamodule.
# We'll inject it by subclassing.
custom_mlp = MLPTranscriptomeProcessor(gene_list_path=gene_list_path)
logger.info(
    f"Custom MLP processor: {custom_mlp.input_dim} genes (should be {n_target_genes})"
)
assert custom_mlp.input_dim == n_target_genes

dm = JointEmbedDataModule(
    tokenizer="bert",
    transcriptome_processor="mlp",
    image_processor="uni2",
    dataset_names="hest1k",
    batch_size=batch_size,
    train_fraction=0.95,
    use_disk_loading=True,
)

# Replace the transcriptome processor BEFORE prepare_data
# This affects new processing, but cached data already has expression_expr baked in.
dm.processor.transcriptome_processor = custom_mlp

# Force reprocessing by invalidating the cache hash.
# We do this by overriding _compute_hash to include the gene list info.
_orig_hash = dm._compute_hash


def _custom_hash(i=None):
    return _orig_hash(i) + "_hestgf17851"


dm._compute_hash = _custom_hash

dm.prepare_data()
dm.setup()

logger.info(f"Train dataloader: {len(dm.train_dataloader())} batches")

# Verify the expression dimension matches
sample_batch = next(iter(dm.train_dataloader()))
expr_dim = sample_batch["expression_expr"].shape[-1]
logger.info(f"expression_expr dim from dataloader: {expr_dim}")
assert expr_dim == n_target_genes, (
    f"expression_expr dim {expr_dim} != target {n_target_genes}. "
    f"Cache invalidation may have failed."
)

# ── Initialize decoder ───────────────────────────────────────────────────────
from cellwhisperer.expression_decoder.raw_uni2_decoder_lightning import (
    RawUNI2DecoderLightning,
)

decoder = RawUNI2DecoderLightning(
    gene_list_path=gene_list_path,
    uni2_weights_dir=uni2_weights_dir,
    uni2_embed_dim=1536,
    projection_dim=2048,
    learning_rate=learning_rate,
    max_epochs=max_epochs,
    loss_type="mse",
)

# ── Train ─────────────────────────────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    dirpath=str(Path(output_ckpt).parent),
    filename="decoder",
    save_last=True,
    monitor="val/loss",
    mode="min",
)

try:
    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = WandbLogger(
        project="cellwhisperer",
        entity="single-cellm",
        name="two_stage_decoder_hest1k",
        group=os.environ.get("WANDB_RUN_GROUP", "icml_revisions"),
        log_model=False,
    )
except Exception as e:
    logger.warning(f"wandb init failed: {e}")
    wandb_logger = True

trainer = pl.Trainer(
    max_epochs=max_epochs,
    precision="bf16-mixed",
    callbacks=[checkpoint_cb],
    logger=wandb_logger,
)
trainer.fit(decoder, datamodule=dm)

best = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
if os.path.abspath(best) != os.path.abspath(output_ckpt):
    shutil.copy(best, output_ckpt)
logger.info(f"Saved decoder to {output_ckpt} (from {best})")
