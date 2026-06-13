"""
Dataset UMAP: Visualize overlap of Geneformer and trimodal embeddings
across cellxgene_census (scRNA-seq) and hest1k (Visium spatial transcriptomics).

Extracts both embedding types in a single forward pass per sample:
  - transcriptome_features: raw Geneformer pooled hidden states (pre-projection)
  - transcriptome_embeds: trimodal projected embeddings (post-projection head)

Produces UMAP plots colored by dataset source + CSV of coordinates.
"""

import pyarrow  # must be first (glibc compat on Sherlock)

import torch
import numpy as np
import pandas as pd
import anndata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style
from pathlib import Path
from tqdm import tqdm
import umap
import glob

from spatialwhisperer.utils.model_io import load_cellwhisperer_model
from spatialwhisperer.jointemb.dataset.inference import CellxGenePreparationLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]  # spatialwhisperer root
matplotlib.style.use(str(PROJECT_DIR / "src/plot_style/main.style"))

OUTPUT_DIR = PROJECT_DIR / "results/spatialwhisperer_eval/dataset_umap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CKPT = (
    PROJECT_DIR
    / "results/models/jointemb/spatialwhisperer_cellxgene_census__archs4_geo__hest1k.ckpt"
)
CELLXGENE_PATH = PROJECT_DIR / "results/cellxgene_census/full_data.h5ad"
HEST1K_GLOB = str(PROJECT_DIR / "results/hest1k/h5ads/full_data_*.h5ad")

N_SAMPLES_PER_DATASET = 5000
BATCH_SIZE = 64
SEED = 42


def strip_image_keys(adata):
    """Remove image-related uns keys so CellxGenePreparationLoader skips image processing."""
    for key in ["he_slide", "20x_slide", "image_path"]:
        adata.uns.pop(key, None)
    return adata


def extract_embeddings(model, loader, device, max_samples):
    """Extract both Geneformer and trimodal embeddings from a dataloader."""
    geneformer_embs = []
    trimodal_embs = []
    n_collected = 0

    for batch in tqdm(loader, desc="Extracting embeddings"):
        if n_collected >= max_samples:
            break

        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            transcriptome_features, transcriptome_embeds = (
                model.get_transcriptome_features(
                    expression_tokens=batch_device.get("expression_tokens"),
                    expression_token_lengths=batch_device.get(
                        "expression_token_lengths"
                    ),
                    normalize_embeds=True,
                )
            )

        geneformer_embs.append(transcriptome_features.cpu().float().numpy())
        trimodal_embs.append(transcriptome_embeds.cpu().float().numpy())
        n_collected += transcriptome_features.shape[0]

    geneformer_embs = np.concatenate(geneformer_embs, axis=0)[:max_samples]
    trimodal_embs = np.concatenate(trimodal_embs, axis=0)[:max_samples]
    return geneformer_embs, trimodal_embs


def load_subsampled_adata(path, n_samples, seed):
    """Load and subsample an AnnData object."""
    adata = anndata.read_h5ad(path)
    rng = np.random.RandomState(seed)
    if adata.n_obs > n_samples:
        idx = rng.choice(adata.n_obs, n_samples, replace=False)
        adata = adata[idx].copy()
    return strip_image_keys(adata)


def load_hest1k_subsampled(glob_pattern, n_samples, seed):
    """Load hest1k from multiple h5ad files, subsampling uniformly."""
    files = sorted(glob.glob(glob_pattern))
    # Randomly select files, then subsample within each
    rng = np.random.RandomState(seed)
    rng.shuffle(files)

    adatas = []
    n_collected = 0
    for f in tqdm(files, desc="Loading hest1k files"):
        if n_collected >= n_samples:
            break
        adata = anndata.read_h5ad(f)
        strip_image_keys(adata)
        remaining = n_samples - n_collected
        if adata.n_obs > remaining:
            idx = rng.choice(adata.n_obs, remaining, replace=False)
            adata = adata[idx].copy()
        n_collected += adata.n_obs
        adatas.append(adata)

    return anndata.concat(adatas)


print("Loading model...")
pl_model, tokenizer, transcriptome_processor, image_processor = (
    load_cellwhisperer_model(model_path=str(MODEL_CKPT), eval=True)
)
model = pl_model.model
device = next(model.parameters()).device
transcriptome_model_type = model.transcriptome_model.config.model_type

# --- cellxgene_census ---
print("Loading cellxgene_census...")
adata_cxg = load_subsampled_adata(CELLXGENE_PATH, N_SAMPLES_PER_DATASET, SEED)
print(f"  cellxgene_census: {adata_cxg.n_obs} samples")

loader_cxg = CellxGenePreparationLoader(
    read_count_table=adata_cxg,
    transcriptome_processor=transcriptome_model_type,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

gf_cxg, tri_cxg = extract_embeddings(model, loader_cxg, device, N_SAMPLES_PER_DATASET)
del adata_cxg, loader_cxg
torch.cuda.empty_cache()

# --- hest1k ---
print("Loading hest1k...")
adata_hest = load_hest1k_subsampled(HEST1K_GLOB, N_SAMPLES_PER_DATASET, SEED)
print(f"  hest1k: {adata_hest.n_obs} samples")

loader_hest = CellxGenePreparationLoader(
    read_count_table=adata_hest,
    transcriptome_processor=transcriptome_model_type,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

gf_hest, tri_hest = extract_embeddings(
    model, loader_hest, device, N_SAMPLES_PER_DATASET
)
del adata_hest, loader_hest
torch.cuda.empty_cache()

# --- UMAP ---
for emb_name, emb_cxg, emb_hest in [
    ("geneformer", gf_cxg, gf_hest),
    ("trimodal_projected", tri_cxg, tri_hest),
]:
    print(f"Computing UMAP for {emb_name}...")
    combined = np.concatenate([emb_cxg, emb_hest], axis=0)
    labels = ["cellxgene_census"] * emb_cxg.shape[0] + ["hest1k"] * emb_hest.shape[0]

    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.3, metric="cosine", random_state=SEED
    )
    coords = reducer.fit_transform(combined)

    df = pd.DataFrame(
        {
            "UMAP1": coords[:, 0],
            "UMAP2": coords[:, 1],
            "dataset": labels,
        }
    )
    df.to_csv(OUTPUT_DIR / f"umap_{emb_name}.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"cellxgene_census": "#9c7cb8", "hest1k": "#8c1515"}
    for dataset_name in ["cellxgene_census", "hest1k"]:
        mask = df["dataset"] == dataset_name
        ax.scatter(
            df.loc[mask, "UMAP1"],
            df.loc[mask, "UMAP2"],
            c=colors[dataset_name],
            label=dataset_name,
            s=3,
            alpha=0.4,
            rasterized=True,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{emb_name} embeddings")
    ax.legend(markerscale=4, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"umap_{emb_name}.pdf", dpi=150)
    fig.savefig(OUTPUT_DIR / f"umap_{emb_name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {emb_name} UMAP to {OUTPUT_DIR}")

print("Done.")
