from __future__ import annotations

import argparse
import json
import io
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["plip", "conch"], required=True)
    parser.add_argument(
        "--caption-condition", choices=["original", "curated"], required=True
    )
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--original-h5ad-dir", type=Path, required=True)
    parser.add_argument("--curated-h5ad-dir", type=Path, required=True)
    parser.add_argument("--image-zip-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=128)
    return parser.parse_args()


def load_plip(device: torch.device):
    model = cast(Any, CLIPModel.from_pretrained("vinid/plip"))
    model = model.to(device)
    model.eval()
    processor = cast(Any, CLIPProcessor.from_pretrained("vinid/plip"))
    logit_scale = model.logit_scale.exp().detach().cpu()
    max_length = int(model.config.text_config.max_position_embeddings)

    def encode_images(images, batch_size):
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(
                device
            )
            with torch.inference_mode():
                embeds = model.get_image_features(**inputs)
            outputs.append(F.normalize(embeds, dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    def encode_texts(texts, batch_size):
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                embeds = model.get_text_features(**inputs)
            outputs.append(F.normalize(embeds, dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    return encode_images, encode_texts, float(logit_scale.item())


def load_conch(device: torch.device):
    from conch.open_clip_custom import (
        create_model_from_pretrained,
        get_tokenizer,
        tokenize,
    )

    hf_token = None
    if "HUGGINGFACE_TOKEN" in __import__("os").environ:
        hf_token = __import__("os").environ["HUGGINGFACE_TOKEN"]
    loaded = cast(
        Any,
        create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token
        ),
    )
    model, preprocess = loaded
    model = cast(Any, model).to(device)
    model.eval()
    tokenizer = get_tokenizer()
    logit_scale = model.logit_scale.exp().detach().cpu()

    def encode_images(images, batch_size):
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            image_tensor = torch.stack(
                [preprocess(image) for image in batch], dim=0
            ).to(device)
            with torch.inference_mode():
                embeds = model.encode_image(image_tensor)
            outputs.append(F.normalize(embeds, dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    def encode_texts(texts, batch_size):
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokenized = tokenize(texts=batch, tokenizer=tokenizer).to(device)
            with torch.inference_mode():
                embeds = model.encode_text(tokenized)
            outputs.append(F.normalize(embeds, dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    return encode_images, encode_texts, float(logit_scale.item())


def load_subset_records(manifest_df: pd.DataFrame, h5ad_dir: Path) -> list[dict]:
    patch_ids_by_stem = defaultdict(list)
    for row in manifest_df.itertuples(index=False):
        patch_ids_by_stem[getattr(row, "image_stem")].append(getattr(row, "patch_id"))

    records = {}
    for image_stem, patch_ids in patch_ids_by_stem.items():
        h5ad_path = h5ad_dir / f"full_data_{image_stem}.h5ad"
        adata = anndata.read_h5ad(h5ad_path)
        obs = (
            cast(pd.DataFrame, adata.obs)
            .loc[patch_ids, ["x_pixel", "y_pixel", "natural_language_annotation"]]
            .copy()
        )
        image_path = Path(adata.uns["image_path"])
        patch_size = int(
            adata.uns.get("patch_size", adata.uns.get("spot_diameter_fullres", 224))
        )
        for patch_id, row in obs.iterrows():
            records[patch_id] = {
                "patch_id": patch_id,
                "image_path": image_path,
                "image_stem": image_stem,
                "x_pixel": int(row["x_pixel"]),
                "y_pixel": int(row["y_pixel"]),
                "caption": row["natural_language_annotation"],
                "patch_size": patch_size,
            }

    return [records[patch_id] for patch_id in manifest_df["patch_id"]]


def build_image_member_map(
    image_zip_root: Path, image_stems: list[str]
) -> dict[str, tuple[Path, str]]:
    remaining = {f"{stem}.jpg" for stem in image_stems}
    member_map = {}
    for zip_path in sorted(image_zip_root.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                member_name = Path(member).name
                if member_name in remaining:
                    member_map[member_name] = (zip_path, member)
                    remaining.remove(member_name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(
            f"Missing {len(remaining)} image members from {image_zip_root}"
        )
    return member_map


@lru_cache(maxsize=16)
def get_zip_handle(zip_path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(zip_path)


def crop_patch(
    record: dict, image_member_map: dict[str, tuple[Path, str]]
) -> Image.Image:
    zip_path, member_name = image_member_map[f"{record['image_stem']}.jpg"]
    with get_zip_handle(str(zip_path)).open(member_name) as handle:
        image = Image.open(io.BytesIO(handle.read())).convert("RGB")
    x_pixel = record["x_pixel"]
    y_pixel = record["y_pixel"]
    patch_size = record["patch_size"]
    return image.crop((x_pixel, y_pixel, x_pixel + patch_size, y_pixel + patch_size))


def compute_image_embeddings(
    records: list[dict], batch_size: int, encode_images, image_member_map
):
    outputs = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_images = [
            crop_patch(record, image_member_map) for record in batch_records
        ]
        outputs.append(encode_images(batch_images, batch_size=len(batch_images)))
    return torch.cat(outputs, dim=0)


def compute_metrics(
    scores: torch.Tensor, left_embeds: torch.Tensor, right_embeds: torch.Tensor
):
    true_class_indices = list(range(scores.shape[1]))
    num_classes = scores.shape[1]
    preds = scores
    target = torch.tensor(true_class_indices, device=scores.device, dtype=torch.long)

    torchmetric_kwargs = {
        "preds": preds,
        "target": target,
        "num_classes": num_classes,
        "average": "none",
        "top_k": 1,
    }

    res_metrics = {
        "precision": torchmetrics.functional.classification.multiclass_precision(
            **torchmetric_kwargs
        ).detach(),
        "accuracy": torchmetrics.functional.classification.multiclass_accuracy(
            **torchmetric_kwargs
        ).detach(),
        "f1": torchmetrics.functional.classification.multiclass_f1_score(
            **torchmetric_kwargs
        ).detach(),
        "rocauc": torchmetrics.functional.classification.multiclass_auroc(
            **{k: v for k, v in torchmetric_kwargs.items() if k != "top_k"}
        ).detach(),
    }

    for k in [1, 5, 10, 50]:
        if num_classes >= k:
            torchmetric_kwargs["top_k"] = k
            res_metrics[f"recall_at_{k}"] = (
                torchmetrics.functional.classification.multiclass_recall(
                    **torchmetric_kwargs
                ).detach()
            )
            torchmetric_kwargs["top_k"] = 1
        else:
            res_metrics[f"recall_at_{k}"] = torch.full(
                (num_classes,), float("nan"), device=scores.device
            )

    return {
        f"{metric}_macroAvg": float(value.mean().cpu())
        for metric, value in res_metrics.items()
    }


def build_per_sample_table(
    manifest_df: pd.DataFrame,
    scores_text_image: torch.Tensor,
    scores_image_text: torch.Tensor,
):
    n = len(manifest_df)
    rows = []
    for idx in range(n):
        text_scores = scores_text_image[:, idx]
        image_scores = scores_image_text[:, idx]
        text_rank = int((text_scores > text_scores[idx]).sum().item() + 1)
        image_rank = int((image_scores > image_scores[idx]).sum().item() + 1)
        row = manifest_df.iloc[idx].to_dict()
        row.update(
            {
                "matched_score_image_to_text": float(
                    scores_text_image[idx, idx].item()
                ),
                "matched_score_text_to_image": float(
                    scores_image_text[idx, idx].item()
                ),
                "rank_image_to_text": text_rank,
                "rank_text_to_image": image_rank,
                "hit_at_1_image_to_text": int(text_rank <= 1),
                "hit_at_5_image_to_text": int(text_rank <= 5),
                "hit_at_10_image_to_text": int(text_rank <= 10),
                "hit_at_50_image_to_text": int(text_rank <= 50),
                "hit_at_1_text_to_image": int(image_rank <= 1),
                "hit_at_5_text_to_image": int(image_rank <= 5),
                "hit_at_10_text_to_image": int(image_rank <= 10),
                "hit_at_50_text_to_image": int(image_rank <= 50),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(args.subset_manifest)
    print(f"Loaded subset manifest with {len(manifest_df)} rows", flush=True)
    h5ad_dir = (
        args.original_h5ad_dir
        if args.caption_condition == "original"
        else args.curated_h5ad_dir
    )
    records = load_subset_records(manifest_df, h5ad_dir)
    print(
        f"Loaded {len(records)} records for {args.caption_condition} captions from {h5ad_dir}",
        flush=True,
    )
    image_member_map = build_image_member_map(
        args.image_zip_root,
        manifest_df["image_stem"].drop_duplicates().tolist(),
    )

    texts = [record["caption"] for record in records]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "plip":
        encode_images, encode_texts, logit_scale = load_plip(device)
    else:
        encode_images, encode_texts, logit_scale = load_conch(device)

    print(f"Encoding images on {device}", flush=True)
    image_embeds = compute_image_embeddings(
        records, args.image_batch_size, encode_images, image_member_map
    )
    print("Encoding texts", flush=True)
    text_embeds = encode_texts(texts, batch_size=args.text_batch_size)

    print("Computing similarity matrices", flush=True)
    scores_text_image = torch.matmul(text_embeds, image_embeds.T) * logit_scale
    scores_image_text = torch.matmul(image_embeds, text_embeds.T) * logit_scale

    print("Computing retrieval metrics", flush=True)
    image_to_text_metrics = compute_metrics(
        scores_text_image, image_embeds, text_embeds
    )
    text_to_image_metrics = compute_metrics(
        scores_image_text, text_embeds, image_embeds
    )

    metrics = {
        "model": args.model,
        "caption_condition": args.caption_condition,
        "n_samples": int(len(records)),
        "image_to_text": image_to_text_metrics,
        "text_to_image": text_to_image_metrics,
        "macro_auroc_mean": float(
            np.mean(
                [
                    image_to_text_metrics["rocauc_macroAvg"],
                    text_to_image_metrics["rocauc_macroAvg"],
                ]
            )
        ),
    }

    per_sample_df = build_per_sample_table(
        manifest_df, scores_text_image, scores_image_text
    )
    per_sample_df.to_csv(args.output_dir / "per_sample_scores.csv", index=False)

    np.savez_compressed(
        args.output_dir / "similarity_matrix.npz",
        scores_text_image=scores_text_image.numpy().astype(np.float16),
        scores_image_text=scores_image_text.numpy().astype(np.float16),
    )

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote metrics to {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
