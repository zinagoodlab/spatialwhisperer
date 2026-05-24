# Dataset UMAP: Cross-Platform Embedding Overlap

## Motivation
Reviewer concern: The transitive transfer relies on Geneformer producing a consistent embedding space across both HEST-1K (Visium spatial transcriptomics) and CellWhisperer (single-cell RNA-seq / cellxgene_census), despite different measurement platforms with known technical differences in sequencing depth and cell resolution. The reviewer requests empirical verification by embedding gene expression profiles from both datasets through the frozen Geneformer encoder and visualizing overlap via UMAP.

## Design
- **Datasets**: `cellxgene_census` (scRNA-seq) vs `hest1k` (Visium spatial transcriptomics)
- **Model**: `spotwhisperer_cellxgene_census__archs4_geo__hest1k.ckpt` (trimodal, no quilt1m)
- **Embeddings extracted** (simultaneously from single forward pass):
  1. **Geneformer embeddings** (`transcriptome_features`): raw frozen Geneformer pooled hidden states, before projection
  2. **Trimodal projected embeddings** (`transcriptome_embeds`): after the learned projection head in shared CLIP space
- **Visualization**: UMAP colored by dataset source (cellxgene_census vs hest1k)
- **Subsampling**: random 5k samples per dataset for tractable UMAP

## Implementation
- `scripts/dataset_umap.py`: loads model, iterates over both datasets, extracts both embedding types, computes UMAP, saves plot + CSV
- Run on Sherlock with 1 GPU (H100), sufficient memory

## How to run
```bash
# On Sherlock
sbatch --partition=cmackall -G 1 --cpus-per-task=8 --mem=128G --time=04:00:00 \
  --output=/scratch/users/moritzs/dataset_umap_%j.out \
  --error=/scratch/users/moritzs/dataset_umap_%j.err \
  --job-name=dataset_umap \
  --wrap='cd ~/cellwhisperer_private && conda run -n cellwhisperer python src/spotwhisperer_eval/scripts/dataset_umap.py'
```

## Results

**Both embedding spaces show highly disjoint clusters by dataset source.** cellxgene_census and hest1k samples form almost completely non-overlapping regions in UMAP for both the raw Geneformer embeddings and the trimodal projected embeddings.

This confirms the reviewer's concern: Geneformer does **not** produce a consistent shared embedding space across scRNA-seq and Visium data. The two platforms occupy separate manifolds, meaning the bridge modality assumption does not hold at the level of the frozen encoder.

Importantly, the disjunction is equally pronounced in the trimodal projected space, suggesting the learned projection head does not correct for this platform gap either — it aligns modalities within-dataset but does not bridge the scRNA-seq / Visium domain shift.

### Implications for the paper
This is an important qualifier on the main results: the transitive transfer works empirically (as shown by the benchmark numbers), but not because Geneformer provides a truly shared representation. The mechanism must instead be that the trimodal training objective forces the projection head to align the two domains indirectly via the shared text and image modalities, rather than relying on a pre-existing Geneformer embedding overlap. This is arguably a more interesting result — it shows the trimodal learning objective itself is doing the heavy lifting of cross-platform alignment, beyond what the frozen encoder provides.

This should be addressed honestly in the paper, e.g. in a limitations section or as a nuance in the discussion of the transitive transfer mechanism.

## Progress Log

### 2026-03-26: Completed
- Submitted SLURM job **19680187** on Sherlock (cmackall, 1 GPU, 128G, 4h)
- **Completed successfully**
- Output: `results/spotwhisperer_eval/dataset_umap/umap_{geneformer,trimodal_projected}.{pdf,png,csv}`
- **Finding**: clusters highly disjoint in both embedding spaces

## Status
- [x] Script implemented
- [x] Job completed (19680187)
- [x] Results reviewed
- [ ] Addressed in paper (discussion / limitations)
