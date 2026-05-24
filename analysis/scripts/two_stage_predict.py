"""
Two-stage baseline prediction for PathoCellBench:
  Stage 1: H&E image → UNI2 → decoder → predicted gene expression
  Stage 2: predicted expression → Geneformer → fine-tuned classifier → cell type probs

Outputs a score CSV in the same format as CONCH/PLIP/SpotWhisperer baselines
(one column per cell type, one row per patch/cell).
"""

import pyarrow  # needed

import anndata
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load inputs from snakemake ──────────────────────────────────────────────
adata = anndata.read_h5ad(snakemake.input.adata)
image_path = snakemake.input.image

decoder_ckpt = snakemake.input.decoder_ckpt
classifier_weights = snakemake.input.classifier_weights
gene_list_path = snakemake.input.gene_list
label_mapping_csv = snakemake.input.label_mapping

seed = int(snakemake.wildcards.seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

# ── Stage 0: Extract UNI2 embeddings from image patches ─────────────────────
logger.info("Extracting UNI2 embeddings from image patches...")

from cellwhisperer.jointemb.uni_model import UNIModel, UNIConfig, UNIProcessor

uni_config = UNIConfig(cell_level_model=False, context_model=True)
uni_model = UNIModel.from_pretrained(
    str(Path(snakemake.input.uni2_weights) / "pytorch_model.bin"),
    config=uni_config,
)
uni_model = uni_model.to(device).eval()

# Use UNIProcessor to extract patches (context 224x224 + cell 56x56)
uni_processor = UNIProcessor()
patches = uni_processor(adata, return_tensors="pt")
patches_ctx = patches["patches_ctx"]  # [N, 3, 224, 224]
patches_cell = patches["patches_cell"]  # [N, 3, 56, 56]

logger.info(f"Extracted {patches_ctx.shape[0]} patches")

# Forward through UNI2
all_image_embeds = []
with torch.no_grad():
    for i in range(0, patches_ctx.shape[0], batch_size):
        ctx_batch = patches_ctx[i : i + batch_size].to(device)
        cell_batch = patches_cell[i : i + batch_size].to(device)
        _, embeds = uni_model(patches_ctx=ctx_batch, patches_cell=cell_batch)
        all_image_embeds.append(embeds.cpu())

image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, 1536]
logger.info(f"UNI2 embeddings shape: {image_embeds.shape}")

# ── Stage 1: Decode expression from UNI2 embeddings ─────────────────────────
logger.info("Decoding gene expression from UNI2 embeddings...")

from cellwhisperer.expression_decoder.raw_uni2_decoder_lightning import (
    RawUNI2DecoderLightning,
)

decoder_module = RawUNI2DecoderLightning.load_from_checkpoint(
    decoder_ckpt,
    gene_list_path=gene_list_path,
    # strict=False needed because decoder.ckpt was saved before the object.__setattr__
    # fix, so it contains _uni2_model weights as extra keys. The projection/decoder
    # weights load correctly; the UNI2 weights are simply discarded.
    strict=False,
)
decoder_module = decoder_module.to(device).eval()

predicted_expression = []
with torch.no_grad():
    for i in range(0, image_embeds.shape[0], batch_size):
        batch_embeds = image_embeds[i : i + batch_size].to(device)
        pred = decoder_module(batch_embeds)
        predicted_expression.append(pred.cpu())

predicted_expression = torch.cat(predicted_expression, dim=0).numpy()  # [N, num_genes]
logger.info(f"Predicted expression shape: {predicted_expression.shape}")

# Create AnnData with predicted expression (decoder predicts log1p, convert back to counts)
gene_df = pd.read_csv(gene_list_path)
gene_names = gene_df["gene_name"].tolist()

predicted_counts = np.expm1(
    np.clip(predicted_expression, 0, 20)
)  # clip to avoid overflow
predicted_adata = anndata.AnnData(
    X=predicted_counts,
    obs=adata.obs.copy(),
    var=pd.DataFrame(index=gene_names),
)

# ── Stage 2: Classify cell types using fine-tuned Geneformer ─────────────────
logger.info("Classifying cell types from predicted expression...")

# Load the label mapping (training_cell_type → evaluation_cell_type)
label_mapping = pd.read_csv(label_mapping_csv)
num_training_classes = len(label_mapping)

# Load fine-tuned Geneformer classifier
import sys

# finetuning_eval is a local module under src/figures/notebooks/
_project_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_dir / "src" / "figures" / "notebooks"))
from finetuning_eval.models.geneformer import GeneformerCelltypeModel, GeneformerConfig

geneformer_model = GeneformerCelltypeModel(
    GeneformerConfig(),
    num_classes=num_training_classes,
    freeze=True,
)
geneformer_model.load_state_dict(torch.load(classifier_weights, map_location="cpu"))
geneformer_model = geneformer_model.to(device).eval()

# Tokenize predicted expression using Geneformer processor
from cellwhisperer.jointemb.geneformer_model import GeneformerTranscriptomeProcessor

geneformer_processor = GeneformerTranscriptomeProcessor(
    nproc=1,
    emb_label=[],
)

logger.info("Tokenizing predicted expression for Geneformer...")
tokenized = geneformer_processor(predicted_adata, return_tensors="pt")

expression_tokens = tokenized["expression_tokens"].to(device)
expression_token_lengths = tokenized["expression_token_lengths"].to(device)

# Run through classifier in batches
all_logits = []
with torch.no_grad():
    for i in range(0, expression_tokens.shape[0], batch_size):
        batch_tokens = expression_tokens[i : i + batch_size]
        batch_lengths = expression_token_lengths[i : i + batch_size]
        logits = geneformer_model(
            expression_tokens=batch_tokens,
            expression_token_lengths=batch_lengths,
        )
        all_logits.append(logits.cpu())

all_logits = torch.cat(all_logits, dim=0)  # [N, num_training_classes]
probs = torch.softmax(all_logits, dim=-1).numpy()

# ── Map training cell types → PathoCellBench CRC cell types ──────────────────
logger.info("Mapping training cell types to evaluation cell types...")

crc_cell_types = [
    c for c in adata.obsm["cell_type_counts_coarse"].columns if c.lower() != "background"
]

# Sum probabilities for all training types mapping to same eval type
predictions_df = pd.DataFrame(
    np.zeros((probs.shape[0], len(crc_cell_types))),
    columns=crc_cell_types,
    index=adata.obs.index,
)

for idx, row in label_mapping.iterrows():
    eval_ct = row["evaluation_cell_type"]
    if eval_ct in crc_cell_types:
        predictions_df[eval_ct] += probs[:, idx]

# ── Save scores ──────────────────────────────────────────────────────────────
logger.info(f"Saving scores to {snakemake.output.scores}")
predictions_df.to_csv(snakemake.output.scores, index=False)
logger.info("Done.")
