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
- Paper: [preprint.pdf](preprint.pdf); OpenReview (still hidden): <https://openreview.net/forum?id=Ze7U293Zw4>

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

The default environment installs the CPU build of PyTorch, which runs on any machine. On Linux x86_64 with an NVIDIA GPU, use the `cuda` environment instead, which installs the CUDA build: `pixi install -e cuda`, then `pixi shell -e cuda`. There, `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

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

The first call downloads the checkpoint plus the UNI2 and Geneformer weights; later calls load from cache.

SpatialWhisperer maps text, H&E image patches and gene-expression profiles into one shared 2048-dimensional space (`projection_dim=2048` in the checkpoint), where any two of them are compared by cosine similarity. Each `get_<modality>_features` call returns a tuple `(pooled_features, projected_embed_in_shared_space)`; the second element is the 2048-D embedding you compare across modalities.

**Text prompts**

```python
import torch

prompts = ["cytotoxic T cells", "plasma cells"]
text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    _, text_emb = model.model.get_text_features(normalize_embeds=True, **text_inputs)
print(text_emb.shape)  # torch.Size([2, 2048])
print(text_emb[0] @ text_emb[1])  # tensor(0.4042): cosine similarity of the two prompts
```

**H&E image patches.** The image tower takes 224 × 224 px RGB patches, converted to a tensor and normalized with the ImageNet mean and standard deviation (the transform used in training). Training patches were cropped from Visium H&E slides at their native resolution (median 0.46 µm/px, range 0.17–0.80 µm/px), so patches at a similar scale (around 0.5 µm/px, i.e. 20×) are closest to what the model saw. Tiles of another size can be resized and center-cropped to 224 px first, as `analysis/scripts/score_kriegsmann_skin.py` does for the paper's skin benchmark.

```python
from PIL import Image
from torchvision import transforms

to_input = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
patch = Image.open("path/to/patch.png").convert("RGB")  # 224 x 224 px
patches = to_input(patch).unsqueeze(0).to(model.device)

with torch.no_grad():
    _, image_emb = model.model.get_image_features(patches_ctx=patches, normalize_embeds=True)
print(image_emb.shape)  # torch.Size([1, 2048])

# Zero-shot annotation: cosine similarity between the patch and each prompt
print(image_emb @ text_emb.T)  # shape (1, 2)
```

**Gene expression.** `adata_to_embeds` takes an AnnData with raw counts in `.X` (one row per spot or cell) and gene symbols or Ensembl IDs as `var_names`, and returns unit-length embeddings:

```python
from spatialwhisperer.utils.processing import adata_to_embeds

gene_emb = adata_to_embeds(adata, model.model)  # torch.Size([n_obs, 2048])
```

With `use_image_data=True`, the same function embeds the image instead: a 224 × 224 px patch centered on each spot of a whole-slide image, given spot centers in `adata.obs["x_pixel"]` / `adata.obs["y_pixel"]` and the slide as an RGB array in `adata.uns["he_slide"]`. A file path in `adata.uns["image_path"]` works too; files other than `.svs` are opened with PIL, which refuses images above about 179 megapixels (most whole slides) unless `PIL.Image.MAX_IMAGE_PIXELS = None` is set first.

## Reproduce paper analyses

The pipeline lives in `analysis/` (Snakemake 7.x). Outputs go to
`<repo>/results/`. Snakemake is invoked from inside `analysis/`. E.g.

```bash
cd analysis
pixi run snakemake -j 4 pathocell_all                                # tab:method_baselines_benchmark (PathoCell column)
```

Standalone Python script for the multi-seed table 2 numbers:

```bash
cd <repo-root>
BASELINE_TERMS=terms1 pixi run python analysis/scripts/compute_baselines_table2_style.py
```

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
  % volume    = {}, %tbd
  publisher = {PMLR},
  address   = {Seoul, South Korea},
  month     = jul,
  year      = {2026},
  url       = {https://openreview.net/forum?id=Ze7U293Zw4} %tbd
}
```
