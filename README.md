# SpatialWhisperer

Accompanying repository for **Transitive Representation Learning Enhances
Histopathology Annotation** (Schaefer et al., ICML 2026).

Use this repo to:

- **Reproduce paper analyses** &mdash; Use the `analysis/` Snakemake pipeline to
  reproduce our published results
- **Run the model on your own data** &mdash; load the trained checkpoint from
  HuggingFace and embed image patches, transcriptome profiles, and free-text
  prompts in the shared trimodal space.

Released artifacts:

- Model weights: **<https://huggingface.co/Good-Lab/spatialwhisperer>**
- Paper: <https://openreview.net/forum?id=Ze7U293Zw4>

---

## Install

Tested on Linux x86_64 and aarch64 with glibc ≥ 2.28.

```bash
git clone git@github.com:snap-stanford/spatialwhisperer_paper.git
cd spatialwhisperer_paper
curl -fsSL https://pixi.sh/install.sh | bash    # if you don't have pixi yet
pixi install                                    # ~10 min first run; ~5 GB env
pixi shell                                      # activate
```

That's the whole setup. The first `pixi install` solves a fully-locked env
(see `pixi.lock`) and editable-installs the `spatialwhisperer` package.

Foundation-model weights (UNI2, Geneformer) are downloaded **on demand**
the first time `load_spatialwhisperer_model()` is called. UNI2
(`MahmoodLab/UNI2-h`) is a gated HuggingFace model &mdash; accept the terms
once at <https://huggingface.co/MahmoodLab/UNI2-h>, then make the token
visible to your pixi env (the loader checks `HF_TOKEN` / `HUGGINGFACE_TOKEN`
first, otherwise falls back to `huggingface_hub`'s cache):

```bash
huggingface-cli login    # paste your read token
```

On HPC clusters where `~/.cache` is sometimes redirected to per-node
`/tmp`, also set `HF_HOME` to a shared filesystem path before logging in
(e.g. `export HF_HOME=$HOME/.cache/huggingface`) so the token survives
across compute nodes.

**Older clusters (glibc 2.17, e.g. Sherlock)**: pixi binaries won't run.
Use the existing `cellwhisperer` conda env there (it has the same deps; the
snakemake rules' `conda:` directives already point at it).

---

## Use the model

Load SpatialWhisperer in three lines. The first call downloads the
checkpoint (~4.5 GB) plus UNI2 and Geneformer weights; subsequent calls
load from cache.

```python
from spatialwhisperer import load_spatialwhisperer_model

model, tokenizer, transcriptome_processor, image_processor = load_spatialwhisperer_model()
# model: TranscriptomeTextDualEncoderLightning (frozen, eval mode, on CUDA if available)
```

Embed text prompts into the shared 2048-D space and compute pairwise
cosine similarity (verified working — see snippet below). Each
`get_<modality>_features` call returns a tuple `(pooled_features,
projected_embed_in_shared_space)`; the second element is what you compare
across modalities.

```python
import torch

prompts = ["a sample of cytotoxic T cells", "a sample of plasma cells"]
text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    _, text_emb = model.model.get_text_features(normalize_embeds=True, **text_inputs)
print(text_emb.shape)                                                 # (2, 2048)
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

### Custom data / cache locations

- The downloads land in `<repo>/resources/{geneformer-12L-30M,uni2}/` by
  default (matches `config.yaml`'s `model_name_path_map`). Override by
  passing `target_dir=...` to `ensure_geneformer_weights` /
  `ensure_uni2_weights`, or by populating those directories before the first
  load call.
- The SpatialWhisperer checkpoint itself is cached under
  `~/.cache/huggingface/hub/` via `huggingface_hub`.

---

## Reproduce paper analyses

The pipeline lives in `analysis/` (Snakemake 7.x). Outputs go to
`<repo>/results/`. Snakemake is invoked from inside `analysis/` so the
`include:` paths resolve correctly:

```bash
cd analysis
pixi run snakemake --dry-run -j1 pathocell_terms1_vs_terms2_table   # preview
pixi run snakemake -j 4 pathocell_terms1_vs_terms2_table            # appendix tab:conch_plip_comparison
pixi run snakemake -j 4 subset_performance_trend_grid               # main-text fig:subsampling
pixi run snakemake -j 4 pathocell_all                                # tab:method_baselines_benchmark (PathoCell column)
pixi run snakemake -j 4 hest_benchmark_all                           # tab:method_baselines_benchmark (HEST column)
```

Standalone Python script for the multi-seed table 2 numbers:

```bash
cd <repo-root>
BASELINE_TERMS=terms1 pixi run python analysis/scripts/compute_baselines_table2_style.py
```

**Data dependencies.** Most downstream targets (tables, plots) only need
the cached per-model inference outputs in `results/pathocell_evaluation/`,
`results/spatialwhisperer_eval/`, `results/skin_benchmark/`, and
`results/experiments/`. Training and per-dataset inference targets require
GPUs and the raw datasets (HEST-1K, CELLxGENE Census, ARCHS4-GEO,
Quilt-1M, PathoCell, Lizard, PanNuke, Kriegsmann skin) &mdash; see
`config.yaml` and the per-experiment `analysis/experiments/<name>/SUMMARY.md`
files for download paths.

**Note on the `conda env export` warning.** Several rules carry a
`conda: "cellwhisperer"` directive (a historical Sherlock convention).
Snakemake 7 emits a `CalledProcessError: Command 'conda env export --name
'cellwhisperer''` traceback at the end of each rule when `conda` is not on
PATH (i.e. under pixi). The traceback is cosmetic &mdash; the rule's
outputs are produced correctly. To suppress it, either install a `conda`
shim (`echo '#!/bin/sh\nexit 0' > "$PIXI_HOME/bin/conda" && chmod +x ...`)
or run snakemake without invoking those rules' env-recording.

**On Sherlock / SLURM**: pixi doesn't run there (glibc 2.17). Use the
`cellwhisperer` conda env that already exists in `~/group_home/miniforge3/envs/`,
plus snakemake 7's legacy `slurm` resource string. Example controllers:
`analysis/experiments/*/sherlock_controller.sh`.

---

## Repository layout

```
spatialwhisperer_paper/
├── analysis/              # paper pipeline (Snakemake)
│   ├── Snakefile          # entry point
│   ├── rules/             # 24 modular rule files
│   ├── scripts/           # standalone Python tools (table builders, etc.)
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
  volume    = {306},
  publisher = {PMLR},
  address   = {Seoul, South Korea},
  month     = jul,
  year      = {2026},
  url       = {https://openreview.net/forum?id=Ze7U293Zw4}
}
```
