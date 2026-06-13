#!/usr/bin/env python3
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-metadata", type=Path, required=True)
    parser.add_argument("--curated-metadata", type=Path, required=True)
    parser.add_argument("--image-zip-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default="vinid/plip")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    parser.add_argument("--bridge-original", type=float, default=0.645)
    parser.add_argument("--bridge-curated", type=float, default=0.695)
    return parser.parse_args()


def load_and_align_metadata(original_path: Path, curated_path: Path) -> pd.DataFrame:
    original = pd.read_csv(original_path)
    curated = pd.read_csv(curated_path)

    key_cols = ["Unnamed: 0", "image_path"]
    merged = original[key_cols + ["caption"]].merge(
        curated[key_cols + ["caption"]],
        on=key_cols,
        suffixes=("_original", "_curated"),
        how="inner",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"Unnamed: 0": "row_id"})
    return merged


def sample_pairs(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size > len(df):
        raise ValueError(
            f"Requested {sample_size} pairs, but only {len(df)} are available"
        )
    sampled = (
        df.sample(n=sample_size, random_state=seed)
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    sampled["pair_index"] = np.arange(len(sampled))
    return sampled


def build_image_member_map(
    image_zip_root: Path, image_names: list[str]
) -> dict[str, tuple[Path, str]]:
    remaining = set(image_names)
    member_map: dict[str, tuple[Path, str]] = {}
    zip_paths = sorted(image_zip_root.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No zip files found in {image_zip_root}")

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                member_name = Path(member).name
                if member_name in remaining:
                    member_map[member_name] = (zip_path, member)
                    remaining.remove(member_name)
            if not remaining:
                break

    if remaining:
        missing = ", ".join(sorted(list(remaining))[:10])
        raise FileNotFoundError(
            f"Could not resolve {len(remaining)} image files, e.g. {missing}"
        )

    return member_map


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_names: list[str],
    image_member_map: dict[str, tuple[Path, str]],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    zip_handles: dict[Path, zipfile.ZipFile] = {}
    embeddings = []

    try:
        for start in range(0, len(image_names), batch_size):
            batch_names = image_names[start : start + batch_size]
            images = []
            for image_name in batch_names:
                zip_path, member_name = image_member_map[image_name]
                if zip_path not in zip_handles:
                    zip_handles[zip_path] = zipfile.ZipFile(zip_path)
                with zip_handles[zip_path].open(member_name) as handle:
                    image = Image.open(io.BytesIO(handle.read())).convert("RGB")
                images.append(image)

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                if not isinstance(image_features, torch.Tensor):
                    image_features = image_features.pooler_output
                image_features = F.normalize(image_features, dim=-1)
            embeddings.append(image_features.cpu().numpy().astype(np.float32))
    finally:
        for handle in zip_handles.values():
            handle.close()

    return np.concatenate(embeddings, axis=0)


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeddings = []
    max_length = int(model.config.text_config.max_position_embeddings)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            text_features = F.normalize(text_features, dim=-1)
        embeddings.append(text_features.cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


def compute_scores(
    left_embeds: np.ndarray, right_embeds: np.ndarray, scale: float
) -> np.ndarray:
    return (left_embeds @ right_embeds.T) * scale


def compute_metrics(scores: np.ndarray) -> dict:
    """Compute retrieval metrics using torchmetrics, matching the SpotWhisperer pipeline."""
    scores_t = torch.from_numpy(scores).float()
    num_classes = scores_t.shape[1]
    target = torch.arange(scores_t.shape[0], dtype=torch.long)

    kwargs = {
        "preds": scores_t,
        "target": target,
        "num_classes": num_classes,
        "average": "none",
        "top_k": 1,
    }

    res = {
        "precision": torchmetrics.functional.classification.multiclass_precision(
            **kwargs
        ).detach(),
        "accuracy": torchmetrics.functional.classification.multiclass_accuracy(
            **kwargs
        ).detach(),
        "f1": torchmetrics.functional.classification.multiclass_f1_score(
            **kwargs
        ).detach(),
        "rocauc": torchmetrics.functional.classification.multiclass_auroc(
            **{k: v for k, v in kwargs.items() if k != "top_k"}
        ).detach(),
    }

    for k in [1, 5, 10, 50]:
        if num_classes >= k:
            kwargs["top_k"] = k
            res[f"recall_at_{k}"] = (
                torchmetrics.functional.classification.multiclass_recall(
                    **kwargs
                ).detach()
            )
            kwargs["top_k"] = 1
        else:
            res[f"recall_at_{k}"] = torch.full((num_classes,), float("nan"))

    return {
        f"{metric}_macroAvg": float(value.mean().cpu()) for metric, value in res.items()
    }


def load_or_compute_embeddings(path: Path, fn):
    if path.exists():
        return np.load(path)
    embeds = fn()
    np.save(path, embeds)
    return embeds


def write_score_chunks(
    scores: np.ndarray, out_dir: Path, prefix: str, chunk_size: int
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"target_{i:05d}" for i in range(scores.shape[1])]
    paths = []
    for start in range(0, scores.shape[0], chunk_size):
        end = min(start + chunk_size, scores.shape[0])
        df = pd.DataFrame(scores[start:end], columns=columns)
        df.insert(0, "query_index", np.arange(start, end))
        path = out_dir / f"{prefix}_{start:05d}_{end:05d}.csv.gz"
        df.to_csv(path, index=False)
        paths.append(path.name)
    return paths


def write_results_table(
    metrics_df: pd.DataFrame, args: argparse.Namespace, out_dir: Path
) -> None:
    plip_original = float(
        metrics_df.loc[metrics_df["condition"] == "original", "auroc_macro_mean"].iloc[
            0
        ]
    )
    plip_curated = float(
        metrics_df.loc[metrics_df["condition"] == "curated", "auroc_macro_mean"].iloc[0]
    )

    table_df = pd.DataFrame(
        [
            {
                "Model": "PLIP",
                "Original captions (AUROC)": plip_original,
                "Curated captions (AUROC)": plip_curated,
                "Delta": plip_curated - plip_original,
            },
            {
                "Model": "Our bridge model",
                "Original captions (AUROC)": args.bridge_original,
                "Curated captions (AUROC)": args.bridge_curated,
                "Delta": args.bridge_curated - args.bridge_original,
            },
        ]
    )
    table_df.to_csv(out_dir / "results_table.csv", index=False)

    md = table_df.copy()
    for col in md.columns[1:]:
        md[col] = md[col].map(lambda x: f"{x:.3f}")
    (out_dir / "results_table.md").write_text(md.to_markdown(index=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sampled = sample_pairs(
        load_and_align_metadata(args.original_metadata, args.curated_metadata),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    sampled.to_csv(args.output_dir / "sample_metadata.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.model_name)

    pair_image_embeds_path = args.output_dir / "image_embeddings.npy"
    if pair_image_embeds_path.exists():
        pair_image_embeds = np.load(pair_image_embeds_path)
    else:
        unique_images = sampled["image_path"].drop_duplicates().tolist()
        unique_image_embeds = load_or_compute_embeddings(
            args.output_dir / "unique_image_embeddings.npy",
            lambda: encode_images(
                model=model,
                processor=processor,
                image_names=unique_images,
                image_member_map=build_image_member_map(
                    args.image_zip_root, unique_images
                ),
                batch_size=args.image_batch_size,
                device=device,
            ),
        )
        image_index = {name: idx for idx, name in enumerate(unique_images)}
        pair_image_embeds = unique_image_embeds[
            sampled["image_path"].map(image_index).to_numpy()
        ]
        np.save(pair_image_embeds_path, pair_image_embeds)

    original_text_embeds = load_or_compute_embeddings(
        args.output_dir / "text_embeddings_original.npy",
        lambda: encode_texts(
            model=model,
            processor=processor,
            texts=sampled["caption_original"].tolist(),
            batch_size=args.text_batch_size,
            device=device,
        ),
    )
    curated_text_embeds = load_or_compute_embeddings(
        args.output_dir / "text_embeddings_curated.npy",
        lambda: encode_texts(
            model=model,
            processor=processor,
            texts=sampled["caption_curated"].tolist(),
            batch_size=args.text_batch_size,
            device=device,
        ),
    )

    scale = float(model.logit_scale.exp().detach().cpu())
    metrics = []
    score_manifest = {}
    for condition, text_embeds in [
        ("original", original_text_embeds),
        ("curated", curated_text_embeds),
    ]:
        scores_text_image = compute_scores(
            text_embeds, pair_image_embeds, scale=scale
        ).astype(np.float32)
        scores_image_text = compute_scores(
            pair_image_embeds, text_embeds, scale=scale
        ).astype(np.float32)
        np.savez_compressed(
            args.output_dir / f"similarity_matrix_{condition}.npz",
            scores_text_image=scores_text_image.astype(np.float16),
            scores_image_text=scores_image_text.astype(np.float16),
        )
        np.save(args.output_dir / f"scores_{condition}.npy", scores_image_text)
        score_manifest[condition] = write_score_chunks(
            scores=scores_image_text,
            out_dir=args.output_dir / f"scores_{condition}_csv",
            prefix=f"scores_{condition}",
            chunk_size=args.score_chunk_size,
        )

        image_to_text_metrics = compute_metrics(scores_text_image)
        text_to_image_metrics = compute_metrics(scores_image_text)
        metrics.append(
            {
                "model": "PLIP",
                "condition": condition,
                "n_pairs": len(sampled),
                "n_unique_images": int(sampled["image_path"].nunique()),
                "image_to_text_auroc_macro": image_to_text_metrics["rocauc_macroAvg"],
                "text_to_image_auroc_macro": text_to_image_metrics["rocauc_macroAvg"],
                "image_to_text_recall_at_1_macro": image_to_text_metrics[
                    "recall_at_1_macroAvg"
                ],
                "text_to_image_recall_at_1_macro": text_to_image_metrics[
                    "recall_at_1_macroAvg"
                ],
                "auroc_macro_mean": np.mean(
                    [
                        image_to_text_metrics["rocauc_macroAvg"],
                        text_to_image_metrics["rocauc_macroAvg"],
                    ]
                ),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df["delta_vs_original"] = metrics_df["auroc_macro_mean"] - float(
        metrics_df.loc[metrics_df["condition"] == "original", "auroc_macro_mean"].iloc[
            0
        ]
    )
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)

    manifest = {
        "model_name": args.model_name,
        "device": str(device),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "original_metadata": str(args.original_metadata),
        "curated_metadata": str(args.curated_metadata),
        "image_zip_root": str(args.image_zip_root),
        "score_scale": scale,
        "score_csv_chunks": score_manifest,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    write_results_table(metrics_df, args, args.output_dir)


if __name__ == "__main__":
    main()
