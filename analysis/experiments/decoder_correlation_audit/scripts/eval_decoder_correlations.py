"""
Robust correlation audit of the HEST-1K UNI2 → MLP gene-expression decoder
(wandb run dl047zmb, checkpoint dl047zmb.../decoder.ckpt) that reports
val/correlation = 0.680 in the ICML SpatialWhisperer manuscript methods.

We re-evaluate the same checkpoint on the same 95/5 random spot-level split
(seed=42, hest1k, 17,851 genes) but compute three additional metrics that
DeepSpot/Nonchev2025-style ST-from-H&E papers report:

  1. Per-spot Pearson across the 17,851-gene axis (reproduces the 0.68).
  2. Per-gene Pearson across val spots (the DeepSpot-style metric).
  3. A "predict train-mean" baseline under the same two metrics
     (constant-per-gene predictor; isolates the contribution of global
     gene-expression scale to (1)).

In addition we break down (1) by slide_id and by HEST-bench organ (CCRCC,
COAD, HCC, IDC, LUNG, LYMPH_IDC, PAAD, PRAD, READ, SKCM, or "OTHER" if a
HEST sample is outside the benchmark organ panel).

Outputs (parents created by snakemake or --out-dir):
  metrics_summary.json   - all aggregate numbers
  per_spot.parquet       - one row per val spot
  per_gene.parquet       - one row per gene (model + baseline correlations)
  per_slide.csv          - per-slide aggregates
  per_organ.csv          - per-HEST-bench-organ aggregates
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-ckpt", required=True)
    p.add_argument("--gene-list", required=True, help="hest_geneformer_genes.csv")
    p.add_argument("--uni2-weights-dir", required=True)
    p.add_argument(
        "--hest-bench-root",
        required=True,
        help="hest_bench_data/ root (used to build sample_id -> organ map)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--top-hvg",
        type=int,
        default=50,
        help="number of high-variance genes for the focused per-gene-r summary",
    )
    p.add_argument(
        "--limit-train-spots",
        type=int,
        default=100000,
        help="cap on number of train spots used to compute the train-mean baseline "
        "(100K random spots is plenty for a stable per-gene mean; pass 0 to disable; "
        "pass -1 for all ~875K)",
    )
    return p.parse_args()


def build_sample_to_organ(bench_root: Path) -> dict[str, str]:
    """Each {organ}/splits/{train,test}_*.csv lists the sample_ids in that organ."""
    mapping: dict[str, str] = {}
    for organ_dir in sorted(bench_root.iterdir()):
        if not organ_dir.is_dir():
            continue
        organ = organ_dir.name
        splits_dir = organ_dir / "splits"
        if not splits_dir.exists():
            continue
        for csv in splits_dir.glob("*.csv"):
            df = pd.read_csv(csv)
            for sid in df["sample_id"].unique():
                mapping[str(sid)] = organ
    logger.info("HEST-bench sample→organ map covers %d samples", len(mapping))
    return mapping


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Datamodule (same construction as train_two_stage_decoder.py) ─────────
    from spatialwhisperer.jointemb.dataset import JointEmbedDataModule
    from spatialwhisperer.jointemb.mlp_model import MLPTranscriptomeProcessor
    from spatialwhisperer.expression_decoder.raw_uni2_decoder_lightning import (
        RawUNI2DecoderLightning,
    )

    custom_mlp = MLPTranscriptomeProcessor(gene_list_path=args.gene_list)
    n_genes = custom_mlp.input_dim
    logger.info("Gene list: %d genes", n_genes)

    dm = JointEmbedDataModule(
        tokenizer="bert",
        transcriptome_processor="mlp",
        image_processor="uni2",
        dataset_names="hest1k",
        batch_size=args.batch_size,
        train_fraction=0.95,
        use_disk_loading=True,
    )
    dm.processor.transcriptome_processor = custom_mlp

    _orig_hash = dm._compute_hash

    def _custom_hash(i=None):
        return _orig_hash(i) + "_hestgf17851"

    dm._compute_hash = _custom_hash

    dm.prepare_data()
    dm.setup()

    logger.info(
        "val_datasets: %d slides, %d total val spots",
        len(dm.val_datasets),
        sum(len(ds.orig_ids) for ds in dm.val_datasets),
    )
    logger.info(
        "train_datasets: %d slides, %d total train spots",
        len(dm.train_datasets),
        sum(len(ds.orig_ids) for ds in dm.train_datasets),
    )

    # ── Load decoder checkpoint ──────────────────────────────────────────────
    # strict=False: the checkpoint contains the frozen UNI2 weights as part of
    # the state_dict (despite the `object.__setattr__` workaround in the Lightning
    # module), so we ignore the _uni2_model.* keys and lazy-load UNI2 from
    # --uni2-weights-dir at inference time.
    decoder = RawUNI2DecoderLightning.load_from_checkpoint(
        args.decoder_ckpt,
        uni2_weights_dir=args.uni2_weights_dir,
        strict=False,
    )
    decoder.eval().to(device)
    assert decoder.num_genes == n_genes, (
        f"Decoder gene count {decoder.num_genes} != gene list {n_genes}"
    )

    # ── Sample_id → organ map (from HEST-bench split CSVs) ───────────────────
    sample_to_organ = build_sample_to_organ(Path(args.hest_bench_root))

    # ── Train-mean baseline ──────────────────────────────────────────────────
    # Streaming sum of expression_expr across train spots. We parallelize disk
    # I/O via a DataLoader (collate just keeps expression_expr, drops patches).
    # 100K random train spots are more than enough for a stable per-gene mean.
    from torch.utils.data import ConcatDataset, Subset
    import random as _rnd

    def expr_only_collate(samples):
        return torch.stack([s["expression_expr"] for s in samples], dim=0)

    if args.limit_train_spots == 0:
        logger.info("Train mean baseline disabled (--limit-train-spots 0)")
        train_mean = torch.zeros(n_genes, dtype=torch.float32)
        train_n = 0
    else:
        full_train = ConcatDataset(dm.train_datasets)
        if args.limit_train_spots == -1 or args.limit_train_spots >= len(full_train):
            train_subset = full_train
        else:
            _rnd.seed(42)
            idx = _rnd.sample(range(len(full_train)), args.limit_train_spots)
            train_subset = Subset(full_train, idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn=expr_only_collate,
        )
        train_sum = torch.zeros(n_genes, dtype=torch.float64)
        train_n = 0
        for batch_expr in tqdm(train_loader, desc="train mean"):
            train_sum += batch_expr.to(torch.float64).sum(dim=0)
            train_n += batch_expr.shape[0]
        train_mean = (train_sum / max(train_n, 1)).to(torch.float32)
        logger.info(
            "Train mean computed over %d spots (parallel DataLoader, %d workers)",
            train_n,
            args.num_workers,
        )

    # ── Val pass: per-slide DataLoader, collect predictions + targets ────────
    all_pred = []
    all_tgt = []
    spot_records = []  # one dict per val spot

    for ds in tqdm(dm.val_datasets, desc="val slides"):
        slide_id = str(ds.i)
        organ = sample_to_organ.get(slide_id, "OTHER")
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                patches_ctx = batch["patches_ctx"].to(device, non_blocking=True)
                patches_cell = batch["patches_cell"].to(device, non_blocking=True)
                tgt = batch["expression_expr"].to(device, non_blocking=True)

                # Manually invoke UNI2 + decoder (mirrors RawUNI2DecoderLightning._step)
                uni2 = decoder._get_uni2()
                _, embeds = uni2(patches_ctx=patches_ctx, patches_cell=patches_cell)
                pred = decoder(embeds)

                all_pred.append(pred.detach().to(torch.float32).cpu())
                all_tgt.append(tgt.detach().to(torch.float32).cpu())

                n_in_batch = pred.shape[0]
                for k in range(n_in_batch):
                    spot_records.append(
                        {
                            "slide_id": slide_id,
                            "organ": organ,
                            "orig_id": str(ds.orig_ids[offset + k]),
                        }
                    )
                offset += n_in_batch

    pred_mat = torch.cat(all_pred, dim=0)  # [n_val, n_genes]
    tgt_mat = torch.cat(all_tgt, dim=0)
    n_val = pred_mat.shape[0]
    logger.info("Val matrix: %s", tuple(pred_mat.shape))

    # ── Metric helpers ────────────────────────────────────────────────────────
    def per_axis_pearson(X: torch.Tensor, Y: torch.Tensor, axis: int) -> torch.Tensor:
        """Pearson r along `axis`. axis=1 → per-row (per-spot); axis=0 → per-column (per-gene)."""
        Xc = X - X.mean(dim=axis, keepdim=True)
        Yc = Y - Y.mean(dim=axis, keepdim=True)
        num = (Xc * Yc).sum(dim=axis)
        den = torch.sqrt((Xc**2).sum(dim=axis) * (Yc**2).sum(dim=axis))
        return num / den.clamp_min(1e-12)

    per_spot_r_model = per_axis_pearson(pred_mat, tgt_mat, axis=1)  # [n_val]
    per_gene_r_model = per_axis_pearson(pred_mat, tgt_mat, axis=0)  # [n_genes]

    # Baselines: predict the train-set mean / val-set mean for every val spot.
    # Both are constant-per-gene predictors that carry zero spot-specific info.
    # If per-spot r is still high under them, the manuscript metric is being
    # carried by global gene-expression scale rather than spot-level signal.
    val_mean = tgt_mat.mean(dim=0)
    baseline_train = train_mean.unsqueeze(0).expand_as(tgt_mat)
    baseline_val = val_mean.unsqueeze(0).expand_as(tgt_mat)

    per_spot_r_base_train = per_axis_pearson(baseline_train, tgt_mat, axis=1)
    per_spot_r_base_val = per_axis_pearson(baseline_val, tgt_mat, axis=1)
    # per-gene r of any constant predictor is undefined (Var=0) → NaN below.
    per_gene_r_base_train = per_axis_pearson(baseline_train, tgt_mat, axis=0)

    # High-variance gene subset (variance computed on val targets — independent of model)
    tgt_var = tgt_mat.var(dim=0)
    top_idx = torch.argsort(tgt_var, descending=True)[: args.top_hvg]
    top_per_gene_r_model = per_gene_r_model[top_idx]
    top_per_gene_r_base = per_gene_r_base_train[top_idx]

    # ── Build per-spot dataframe ──────────────────────────────────────────────
    df_spot = pd.DataFrame(spot_records)
    df_spot["per_spot_r_model"] = per_spot_r_model.numpy()
    df_spot["per_spot_r_baseline_train_mean"] = per_spot_r_base_train.numpy()
    df_spot["per_spot_r_baseline_val_mean"] = per_spot_r_base_val.numpy()
    df_spot.to_parquet(out_dir / "per_spot.parquet", index=False)

    # ── Per-gene dataframe ────────────────────────────────────────────────────
    gene_list = pd.read_csv(args.gene_list)["gene_name"].tolist()
    df_gene = pd.DataFrame(
        {
            "gene_name": gene_list,
            "per_gene_r_model": per_gene_r_model.numpy(),
            "per_gene_r_baseline_train_mean": per_gene_r_base_train.numpy(),
            "val_mean": val_mean.numpy(),
            "val_var": tgt_var.numpy(),
            "train_mean": train_mean.numpy(),
        }
    )
    df_gene.to_parquet(out_dir / "per_gene.parquet", index=False)

    # ── Per-slide and per-organ aggregates ────────────────────────────────────
    df_slide = (
        df_spot.groupby(["slide_id", "organ"])
        .agg(
            n_spots=("orig_id", "count"),
            per_spot_r_model_mean=("per_spot_r_model", "mean"),
            per_spot_r_model_median=("per_spot_r_model", "median"),
            per_spot_r_baseline_train_mean=("per_spot_r_baseline_train_mean", "mean"),
            per_spot_r_baseline_val_mean=("per_spot_r_baseline_val_mean", "mean"),
        )
        .reset_index()
        .sort_values(["organ", "per_spot_r_model_mean"], ascending=[True, False])
    )
    df_slide.to_csv(out_dir / "per_slide.csv", index=False)

    df_organ = (
        df_spot.groupby("organ")
        .agg(
            n_slides=("slide_id", "nunique"),
            n_spots=("orig_id", "count"),
            per_spot_r_model_mean=("per_spot_r_model", "mean"),
            per_spot_r_model_median=("per_spot_r_model", "median"),
            per_spot_r_baseline_train_mean=("per_spot_r_baseline_train_mean", "mean"),
            per_spot_r_baseline_val_mean=("per_spot_r_baseline_val_mean", "mean"),
        )
        .reset_index()
        .sort_values("per_spot_r_model_mean", ascending=False)
    )
    df_organ.to_csv(out_dir / "per_organ.csv", index=False)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    summary = {
        "n_val_spots": int(n_val),
        "n_val_slides": int(df_spot["slide_id"].nunique()),
        "n_val_organs": int(df_organ.shape[0]),
        "n_genes": int(n_genes),
        "n_train_spots_for_baseline": int(train_n),
        "metric_definitions": {
            "per_spot_r": "Pearson r per val spot across all 17,851 genes (the manuscript metric)",
            "per_gene_r": "Pearson r per gene across all val spots (DeepSpot/Nonchev2025-comparable)",
            "baseline": "Predict train-set mean expression vector for every val spot",
        },
        "model": {
            "per_spot_r_mean": float(per_spot_r_model.mean()),
            "per_spot_r_median": float(per_spot_r_model.median()),
            "per_gene_r_mean": float(np.nanmean(per_gene_r_model.numpy())),
            "per_gene_r_median": float(np.nanmedian(per_gene_r_model.numpy())),
            f"per_gene_r_top{args.top_hvg}_mean": float(
                np.nanmean(top_per_gene_r_model.numpy())
            ),
            f"per_gene_r_top{args.top_hvg}_median": float(
                np.nanmedian(top_per_gene_r_model.numpy())
            ),
        },
        "baseline_predict_train_mean": {
            "per_spot_r_mean": float(per_spot_r_base_train.mean()),
            "per_spot_r_median": float(per_spot_r_base_train.median()),
            "per_gene_r_mean": float(np.nanmean(per_gene_r_base_train.numpy())),
            f"per_gene_r_top{args.top_hvg}_mean": float(
                np.nanmean(top_per_gene_r_base.numpy())
            ),
        },
        "baseline_predict_val_mean": {
            "per_spot_r_mean": float(per_spot_r_base_val.mean()),
            "per_spot_r_median": float(per_spot_r_base_val.median()),
        },
        "per_organ_per_spot_r_mean": (
            df_organ.set_index("organ")["per_spot_r_model_mean"].to_dict()
        ),
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Summary:")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
