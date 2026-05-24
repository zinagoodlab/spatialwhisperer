"""Zero-shot classification of the Kriegsmann et al. (2022) skin H&E benchmark.

Native re-port of the MUSK harness path for the skin task: reads the test split
of `tiles-v2.csv` directly, runs each patch through the CellWhisperer image
tower, and scores against the 16 clinical class names defined in
`skin_labels.py`. Writes a per-patch logits CSV.

Snakemake invalidation: `skin_labels.py` is a declared input, so any edit to
the label list or prompt template re-triggers scoring (which the MUSK rule did
not do — the structural reason the prior relabel rerun was hard to trust).

Inputs (Snakemake):
  - snakemake.input.model: CellWhisperer .ckpt
  - snakemake.input.csv: tiles-v2.csv (dataset manifest)
  - snakemake.input.labels_module: skin_labels.py (single source of truth)

Params:
  - snakemake.params.dataset_root: directory the `file` column paths are relative to
  - snakemake.params.batch_size: image batch size
  - snakemake.params.preprocess: one of
        "resize_crop" — Resize(224, bicubic) + CenterCrop(224) + ToTensor + Normalize
                        (our native default; keeps full tile content, downsampled)
        "crop_only"   — CenterCrop(224) + ToTensor + Normalize
                        (mirrors the MUSK harness path for our model: no resize,
                         center crop from the raw 395x395 tile)

Outputs:
  - snakemake.output.scores: per-patch CSV with columns
        [patch_file, true_class_raw, true_class, <16 clinical-label score columns>]
"""

import importlib.util
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from cellwhisperer.utils.model_io import load_cellwhisperer_model

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spec = importlib.util.spec_from_file_location(
    "skin_labels", snakemake.input.labels_module
)
assert spec is not None and spec.loader is not None, snakemake.input.labels_module
skin_labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skin_labels)

RAW_TO_CLINICAL = skin_labels.RAW_TO_CLINICAL
CLASSES = skin_labels.CLASSES
PROMPT_TEMPLATE = skin_labels.PROMPT_TEMPLATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Loading model %s on %s", snakemake.input.model, device)
pl_model, _, _, _ = load_cellwhisperer_model(
    model_path=snakemake.input.model, eval=True, device=device
)
model = pl_model.model

df = pd.read_csv(snakemake.input.csv)
df = df[df["set"] == "Test"].reset_index(drop=True)
unknown = sorted(set(df["class"]) - set(RAW_TO_CLINICAL))
assert not unknown, f"Unknown raw classes in CSV: {unknown}"
logger.info("Test split: %d patches across %d classes", len(df), df["class"].nunique())

dataset_root = Path(snakemake.params.dataset_root)
batch_size = int(snakemake.params.batch_size)
preprocess = str(snakemake.params.preprocess)

prompts = [PROMPT_TEMPLATE.format(c=c) for c in CLASSES]
with torch.no_grad():
    text_embeds = model.embed_texts(prompts, chunk_size=len(prompts))
text_embeds = text_embeds.to(device)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
logger.info("Encoded %d text prompts, shape=%s", len(prompts), tuple(text_embeds.shape))

_normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
if preprocess == "resize_crop":
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _normalize,
    ])
elif preprocess == "crop_only":
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _normalize,
    ])
else:
    raise ValueError(f"Unknown preprocess: {preprocess!r}")
logger.info("Preprocess pipeline: %s", preprocess)

image_embeds_list = []
n = len(df)
with torch.no_grad():
    for i in range(0, n, batch_size):
        rows = df.iloc[i : i + batch_size]
        tensors = []
        for fp in rows["file"]:
            img = Image.open(dataset_root / fp).convert("RGB")
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device)
        _, img_emb = model.get_image_features(
            patches_ctx=batch, normalize_embeds=True
        )
        image_embeds_list.append(img_emb.detach().cpu())
        if (i // batch_size) % 10 == 0:
            logger.info("Encoded %d/%d patches", min(i + batch_size, n), n)

image_embeds = torch.cat(image_embeds_list, dim=0).to(device)
logits = (image_embeds @ text_embeds.T).cpu().numpy()

out = pd.DataFrame(logits, columns=CLASSES)
out.insert(0, "true_class", df["class"].map(RAW_TO_CLINICAL).values)
out.insert(0, "true_class_raw", df["class"].values)
out.insert(0, "patch_file", df["file"].values)
out.to_csv(snakemake.output.scores, index=False)
logger.info("Saved %d patch scores to %s", len(out), snakemake.output.scores)
