# SpatialWhisperer

Accompanying repository for **Transitive Representation Learning Enhances
Histopathology Annotation** (Schaefer et al., ICML 2026).

Use this repo to:

- **Reproduce paper analyses** &mdash; use the `analysis/` Snakemake pipeline to
  reproduce our published results
- **Run the model on your own data** &mdash; load the trained checkpoint from
  HuggingFace and to perform zero-shot inference of histopathology images 
  in the shared trimodal space.

Released artifacts:

- Model weights: <https://huggingface.co/Good-Lab/spatialwhisperer>
- Paper: <https://openreview.net/forum?id=Ze7U293Zw4>

---

## Install

Tested on Linux x86_64 and aarch64 with glibc ≥ 2.28.

```bash
git clone git@github.com:zinagoodlab/spatialwhisperer.git
cd spatialwhisperer
# curl -fsSL https://pixi.sh/install.sh | bash    # if you don't have pixi yet
pixi install                                    # ~10 min first run; ~5 GB env
pixi shell                                      # activate
```

### Downloading the model

While spatialwhisperer's weights don't require approval for download, we use foundation-model weights (UNI2, Geneformer) that do. 

These are downloaded **on demand**
the first time `load_spatialwhisperer_model()` is called. UNI2
(`MahmoodLab/UNI2-h`) is a gated HuggingFace model &mdash; accept the terms
once at <https://huggingface.co/MahmoodLab/UNI2-h>, then make the token
visible to your pixi env (the loader checks `HF_TOKEN` / `HUGGINGFACE_TOKEN`
first, otherwise falls back to `huggingface_hub`'s cache):

```bash
huggingface-cli login    # paste your read token
```

---

## Use the model



```python
from spatialwhisperer import load_spatialwhisperer_model

model, tokenizer, transcriptome_processor, image_processor = load_spatialwhisperer_model()
# model: TranscriptomeTextDualEncoderLightning (frozen, eval mode, on CUDA if available)
```

The first time this is called with download the checkpoint plus UNI2 and Geneformer weights; subsequent calls load from cache.

Embed text prompts into the shared 512-D space and compute pairwise cosine similarity (verified working — see snippet below). Each `get_<modality>_features` call returns a tuple `(pooled_features, projected_embed_in_shared_space)`; the second element is what you compare across modalities.

```python
import torch

prompts = ["cytotoxic T cells", "plasma cells"]
text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    _, text_emb = model.model.get_text_features(normalize_embeds=True, **text_inputs)
print(text_emb.shape)                                                 # (2, 512)
print(torch.nn.functional.cosine_similarity(text_emb[0:1], text_emb[1:2]).item())
# ~0.20 — two distinct cell types live in different regions of the embedding space
```

Image and transcriptome embeddings follow the same pattern:

```python
from PIL import Image

# Image patch (224x224 H&E at ~0.5 um/px; Visium-compatible)
patch = Image.open("path/to/patch.png").convert("RGB")
image_inputs = image_processor(images=patch, return_tensors="pt").to(model.device)

# Transcriptome profile (counts dict: {gene_symbol: count})
counts = {"CD3D": 42, "CD8A": 17, "MS4A1": 0}
gene_inputs = transcriptome_processor([counts]).to(model.device)

with torch.no_grad():
    _, image_emb = model.model.get_image_features(normalize_embeds=True, **image_inputs)
    _, gene_emb  = model.model.get_transcriptome_features(normalize_embeds=True, **gene_inputs)

sim_it = torch.nn.functional.cosine_similarity(image_emb, text_emb[0:1]).item()
sim_ig = torch.nn.functional.cosine_similarity(image_emb, gene_emb).item()
```

## Reproduce paper analyses

The pipeline lives in `analysis/` (Snakemake 7.x). Outputs go to
`<repo>/results/`. From a fresh clone (after `pixi install` + the HF
login below), one command reproduces every table and figure in the
paper:

```bash
cd analysis
pixi run snakemake -j 4 paper_all
```

`paper_all` composes the per-benchmark aggregates listed below. To
reproduce only one benchmark, target the corresponding subtarget:

| Target | Produces |
|---|---|
| `pathocell_all` | PathoCell CRC + Lizard + PanNuke per-class metrics, plots, model comparisons |
| `seed_analysis_all` | Multi-seed Table 2 (our model, all seeds) + PLIP/CONCH baselines, both `terms1` and `terms2` prompt sets |
| `hest_benchmark_all` | HEST regression + retrieval benchmarks |
| `spatialwhisperer_all` | Lung tissue zero-shot predictions |
| `cellwhisperer_benchmark_all` | CellWhisperer benchmark summary + per-class analysis |
| `lambda_ablation_all` | λ ablation appendix table |
| `freezing_encoder_appendix_all` | Encoder-freezing appendix table |
| `model_interpretation_all` | LLM-based interpretation + correlation + disease-detectability plots |

### Eval data — auto-downloaded

All four eval benchmarks (PathoCell, Lizard, PanNuke, Kriegsmann skin)
are fetched + post-processed from their canonical sources by Snakemake
rules in `analysis/rules/{pathocell_benchmark,eval_dataset_download}.smk`.
No upstream MUSK bundle is needed.

| Dataset | Source | Rule | Disk |
|---|---|---|---|
| PathoCell CRC | [Kainmueller-Lab/PathoCell](https://huggingface.co/datasets/Kainmueller-Lab/PathoCell) on HF (gated) | `pathocell_download_dataset` | ~10 GB |
| Lizard | same HF repo (LMDB) → `convert_lmdb_to_hdf.py` (batch=1) | `download_lizard_lmdb` + `convert_lizard_lmdb_to_hdf` | ~3 GB |
| PanNuke | same HF repo (3-part LMDB, concat ~104 GB) → `convert_lmdb_to_hdf.py` (batch=50) | `download_pannuke_lmdb_parts` + `concat_pannuke_lmdb_parts` + `convert_pannuke_lmdb_to_hdf` | ~250 GB scratch |
| Kriegsmann skin | [heiDATA doi:10.11588/data/7QCR8S](https://doi.org/10.11588/data/7QCR8S) (data.zip) | `download_kriegsmann_skin` | ~4 GB |

**One-time access setup for Kainmueller-Lab/PathoCell**: the dataset is
gated. Visit the [dataset page](https://huggingface.co/datasets/Kainmueller-Lab/PathoCell),
request access, and make a token with read permission visible to the
pipeline via `HF_TOKEN` or `huggingface-cli login`. The Kriegsmann skin
dataset is fully open; no token needed for that one.

## Repository layout

```
spatialwhisperer/
├── analysis/              # paper pipeline (Snakemake)
│   ├── Snakefile          # entry point
│   ├── rules/             # 24 modular rule files
│   ├── scripts/           # rule bodies (table builders, score splitters, etc.)
│   ├── notebooks/         # plotting / per-class analysis notebooks
│   ├── experiments/       # per-ablation working dirs (lambda, freezing,
│   │                      #   seed-variance, ...)
│   └── static/            # vendored baseline logit CSVs (CONCH, PLIP)
├── src/spatialwhisperer/  # model + training code (vendored from cellwhisperer)
├── modules/Geneformer/    # forked Geneformer (CellWhisperer patches)
├── shared/                # dataset preprocessing rules shared with training
├── config.yaml            # path map for datasets and checkpoints
├── pixi.toml + pyproject.toml + pixi.lock  # reproducible env
└── README.md              # this file
```

---

## License

MIT. See `LICENSE`. Foundation-model components (UNI2, Geneformer) carry
their own licenses on HuggingFace.

## Citation

```bibtex
@inproceedings{schaefer2026spatialwhisperer,
  title     = {Transitive Representation Learning Enhances Histopathology Annotation},
  author    = {Schaefer, Moritz and Piran, Zoe and Walter, Nils Philipp and Awasthi, Animesh and Bock, Christoph and Leskovec, Jure and Good, Zinaida},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  % volume    = {}, %tbd
  publisher = {PMLR},
  address   = {Seoul, South Korea},
  month     = jul,
  year      = {2026},
  url       = {https://openreview.net/forum?id=Ze7U293Zw4} %tbd
}
```
