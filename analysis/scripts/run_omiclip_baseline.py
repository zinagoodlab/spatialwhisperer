#!/usr/bin/env python3
"""
Run OmiCLIP (Loki) zero-shot annotation on PathoCellBench patches.

Produces a single CSV of cosine-similarity logits, one row per patch across
all datasets, matching the schema of conch_logits_terms1.csv:
  source_image, spot_id, <class1>, <class2>, ...

OmiCLIP encodes H&E patches with its image encoder and marker-gene
"sentences" with its text encoder, then computes dot-product similarity
(embeddings are L2-normalised, so dot product = cosine similarity).

Marker genes for each PathoCellBench CRC cell type are defined below,
curated from established literature markers.

Usage (standalone):
    python run_omiclip_baseline.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_dir /path/to/pathocell/processed \
        --output /path/to/omiclip_logits.csv \
        --prediction_level patch

Usage (Snakemake): see omiclip_baseline rule in pathocell_benchmark.smk
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CRC cell-type marker gene sentences ────────────────────────────────────
# Each entry: (class_name, space-separated gene symbols)
# These are top marker genes for the 14 PathoCellBench CRC classes (including
# Background).  Gene selection follows standard scRNA-seq marker panels and
# CRC literature.
CRC_MARKER_GENES = {
    "Background": "COL1A1 COL1A2 COL3A1 FN1 VIM SPARC DCN LUM",
    "B cells": "CD79A CD79B MS4A1 CD19 PAX5 BANK1 CD22 BLK IGHM IGHD",
    "Macrophages/Monocytes": "CD68 CD163 CSF1R CD14 FCGR3A MSR1 MRC1 ITGAM AIF1 TYROBP",
    "Adipocytes": "FABP4 ADIPOQ LEP PPARG PLIN1 LIPE CEBPA ACSL1 APOE",
    "Dendritic cells": "ITGAX CLEC9A BATF3 IRF8 FLT3 LAMP3 CD1C FCER1A CLEC10A",
    "T cells": "CD3D CD3E CD4 CD8A CD8B IL7R TRAC TRBC2 LCK",
    "Granulocytes": "S100A8 S100A9 CEACAM8 FCGR3B CSF3R CXCR2 MMP9 ELANE MPO",
    "NK cells": "GNLY NKG7 KLRD1 KLRB1 NCAM1 NCR1 KLRK1 PRF1 GZMB",
    "Nerves": "S100B PLP1 MPZ MBP NEFH NEFL GAP43 TUBB3 SNAP25",
    "Plasma cells": "JCHAIN MZB1 XBP1 IGHG1 IGHA1 SDC1 PRDM1 IRF4",
    "Smooth muscle": "ACTA2 MYH11 TAGLN DES CNN1 MYLK ACTG2 MYOCD LMOD1",
    "Stroma": "FAP PDGFRA PDGFRB COL1A1 COL3A1 THY1 ACTA2 VIM FN1 DCN",
    "Tumor cells": "EPCAM KRT8 KRT18 KRT19 KRT20 CDH1 MUC2 CEACAM5 CDX2 TP53",
    "Vasculature/Lymphatics": "PECAM1 CDH5 VWF FLT1 KDR LYVE1 PROX1 PDPN FLT4",
    "Other cells": "KIT TPSAB1 CPA3 HDC CTSG MS4A2",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing processed PathoCellBench h5ad/tiff files",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--prediction_level", type=str, default="patch")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--gene_sentences_json",
        type=str,
        default=None,
        help="JSON file mapping class names to gene sentences. "
        "If not provided, uses built-in CRC_MARKER_GENES.",
    )
    return p.parse_args()


def load_omiclip(checkpoint: str, device: str):
    """Load OmiCLIP model, preprocess transform, and tokenizer."""
    from open_clip import create_model_from_pretrained, get_tokenizer

    # OmiCLIP checkpoint contains numpy scalars; need weights_only=False
    model, preprocess = create_model_from_pretrained(
        "coca_ViT-L-14",
        device=device,
        pretrained=checkpoint,
        load_weights_only=False,
    )
    tokenizer = get_tokenizer("coca_ViT-L-14")
    model.to(device).eval()
    return model, preprocess, tokenizer


def encode_marker_genes(model, tokenizer, marker_genes: dict, device: str):
    """Encode marker gene sentences → (N_classes, D) L2-normalised embeddings."""
    class_names = list(marker_genes.keys())
    texts = [marker_genes[c] for c in class_names]
    text_inputs = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(text_inputs)
    return class_names, F.normalize(feats, p=2, dim=-1)


def encode_patches_from_tiff(
    model,
    preprocess,
    tiff_path: str,
    spatial_coords: np.ndarray,
    patch_size: int,
    device: str,
    batch_size: int = 64,
):
    """
    Load a composite TIFF image and crop 224x224 patches at the given
    spatial coordinates (patch centres), then encode with OmiCLIP.
    Returns (N, D) L2-normalised embeddings.
    """
    from tifffile import imread

    full_img = imread(tiff_path)  # (H, W, 3)
    half = patch_size // 2

    embeddings = []
    batch = []
    for cx, cy in spatial_coords:
        cx, cy = int(cx), int(cy)
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        # Clamp to image bounds
        y0c, y1c = max(y0, 0), min(y1, full_img.shape[0])
        x0c, x1c = max(x0, 0), min(x1, full_img.shape[1])
        crop = full_img[y0c:y1c, x0c:x1c]
        pil_img = Image.fromarray(crop)
        batch.append(preprocess(pil_img))

        if len(batch) >= batch_size:
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch_tensor)
            embeddings.append(feats)
            batch = []

    if batch:
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
        embeddings.append(feats)

    all_emb = torch.cat(embeddings, dim=0)
    return F.normalize(all_emb, p=2, dim=-1)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = args.device if torch.cuda.is_available() else "cpu"
    pred_level = args.prediction_level

    logger.info(f"Loading OmiCLIP from {args.checkpoint}")
    model, preprocess, tokenizer = load_omiclip(args.checkpoint, device)

    # Load gene sentences: external JSON or built-in short markers
    if args.gene_sentences_json:
        import json

        with open(args.gene_sentences_json) as f:
            gene_sentences = json.load(f)
        logger.info(f"Loaded gene sentences from {args.gene_sentences_json}")
    else:
        gene_sentences = CRC_MARKER_GENES
        logger.info("Using built-in CRC_MARKER_GENES (short marker lists)")

    logger.info("Encoding marker gene sentences")
    class_names, text_emb = encode_marker_genes(
        model, tokenizer, gene_sentences, device
    )
    logger.info(f"  {len(class_names)} classes, embedding dim {text_emb.shape[1]}")

    # Discover all processed datasets
    import anndata as ad

    h5ad_files = sorted(data_dir.glob(f"*_{pred_level}.h5ad"))
    logger.info(f"Found {len(h5ad_files)} datasets")

    all_rows = []
    for h5ad_path in tqdm(h5ad_files, desc="Datasets"):
        dataset_id = h5ad_path.stem.replace(f"_{pred_level}", "")
        tiff_path = h5ad_path.with_suffix(".tiff")
        adata = ad.read_h5ad(h5ad_path)
        n_patches = adata.n_obs

        spatial_coords = adata.obsm["spatial"]  # (N, 2) — patch centres (x, y)
        logger.info(f"  {dataset_id}: {n_patches} patches")
        img_emb = encode_patches_from_tiff(
            model,
            preprocess,
            str(tiff_path),
            spatial_coords,
            224,
            device,
            args.batch_size,
        )

        # Cosine similarity (dot product of L2-normalised vectors)
        sim = (img_emb @ text_emb.T).cpu().numpy()  # (N, C)

        for i in range(n_patches):
            row = {
                "source_image": f"{dataset_id}_{pred_level}.tiff",
                "spot_id": f"patch_{i}",
            }
            for j, cls in enumerate(class_names):
                row[cls] = float(sim[i, j])
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    # Support both CLI and Snakemake invocation
    try:
        snakemake  # noqa: F821
        import sys

        sys.argv = [
            __file__,
            "--checkpoint",
            str(snakemake.input.checkpoint),
            "--data_dir",
            str(snakemake.params.data_dir),
            "--output",
            str(snakemake.output.logits_csv),
            "--prediction_level",
            snakemake.params.prediction_level,
            "--batch_size",
            str(snakemake.params.get("batch_size", 64)),
        ]
        gene_json = snakemake.params.get("gene_sentences_json", None)
        if gene_json:
            sys.argv.extend(["--gene_sentences_json", str(gene_json)])
    except NameError:
        pass
    main()
