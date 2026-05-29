"""
DeepSpot-comparable held-out evaluation of the HEST-1K UNI2→MLP decoder.

Iterates the 10 HEST-bench organs (hesteval_<organ>). For each organ:
  1. Loads the converted spatialwhisperer-format dataset via JointEmbedDataModule
     with train_fraction=0.0 (= all spots go to val).
  2. Filters to held-out samples only (i.e. those NOT present in our
     hest1k training set at results/hest1k/h5ads/full_data_<sample>.h5ad).
  3. Runs UNI2 + decoder forward to obtain predicted expression over our
     17,851-gene vocabulary.
  4. Subsets both predictions and targets to the organ's curated 50-gene
     panel (from results/hesteval_<organ>/dataset_metadata.json).
  5. Computes per-gene Pearson r across all held-out spots in the organ;
     bootstraps the median and a 95% CI (DeepSpot-style aggregation).

Produces:
  per_organ_hesteval_summary.json
  per_organ_hesteval_table.csv
  per_gene_hesteval.parquet   (one row per (organ, gene) pair)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


HEST_ORGANS = [
    "CCRCC",
    "COAD",
    "HCC",
    "IDC",
    "LUNG",
    "LYMPH_IDC",
    "PAAD",
    "PRAD",
    "READ",
    "SKCM",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-ckpt", required=True)
    p.add_argument("--gene-list", required=True)
    p.add_argument("--uni2-weights-dir", required=True)
    p.add_argument("--hesteval-root", required=True,
                   help="results/ root containing hesteval_<organ>/ folders")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--organs", nargs="+", default=HEST_ORGANS)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    return p.parse_args()


def per_gene_pearson(P: torch.Tensor, T: torch.Tensor) -> np.ndarray:
    """Per-column Pearson r between P and T (same shape, [n_spots, n_genes])."""
    Pc = P - P.mean(dim=0, keepdim=True)
    Tc = T - T.mean(dim=0, keepdim=True)
    num = (Pc * Tc).sum(dim=0)
    den = torch.sqrt((Pc**2).sum(dim=0) * (Tc**2).sum(dim=0))
    return (num / den.clamp_min(1e-12)).cpu().numpy()


def bootstrap_median(values: np.ndarray, n_iter: int, seed: int = 42) -> dict:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"median": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    medians = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        sample = rng.choice(values, size=len(values), replace=True)
        medians[i] = np.median(sample)
    return {
        "median": float(np.median(values)),
        "bootstrap_median_mean": float(medians.mean()),
        "ci_low": float(np.quantile(medians, 0.025)),
        "ci_high": float(np.quantile(medians, 0.975)),
        "n": int(len(values)),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    from spatialwhisperer.jointemb.dataset import JointEmbedDataModule
    from spatialwhisperer.jointemb.mlp_model import MLPTranscriptomeProcessor
    from spatialwhisperer.expression_decoder.raw_uni2_decoder_lightning import (
        RawUNI2DecoderLightning,
    )

    custom_mlp = MLPTranscriptomeProcessor(gene_list_path=args.gene_list)
    n_genes = custom_mlp.input_dim
    decoder_gene_list = pd.read_csv(args.gene_list)["gene_name"].tolist()
    gene_to_idx = {g: i for i, g in enumerate(decoder_gene_list)}
    logger.info("Decoder vocab: %d genes", n_genes)

    decoder = RawUNI2DecoderLightning.load_from_checkpoint(
        args.decoder_ckpt,
        uni2_weights_dir=args.uni2_weights_dir,
        strict=False,
    )
    decoder.eval().to(device)
    assert decoder.num_genes == n_genes

    hesteval_root = Path(args.hesteval_root)

    all_organ_summaries = {}
    all_rows = []
    per_gene_records = []

    for organ in args.organs:
        logger.info("=" * 60)
        logger.info("Organ: %s", organ)

        meta_path = hesteval_root / f"hesteval_{organ}" / "dataset_metadata.json"
        with open(meta_path) as f:
            organ_meta = json.load(f)
        panel_genes = organ_meta["genes"]
        # Symbols missing from our decoder vocabulary are silently dropped.
        panel_idx = [(gene_to_idx[g], g) for g in panel_genes if g in gene_to_idx]
        if not panel_idx:
            logger.warning("No panel-gene overlap with decoder vocab for %s, skipping", organ)
            continue
        panel_decoder_idx = torch.tensor([i for i, _ in panel_idx], device="cpu")
        panel_gene_names = [g for _, g in panel_idx]
        logger.info(
            "Panel size: %d, %d overlap with decoder vocab",
            len(panel_genes), len(panel_gene_names),
        )

        dm = JointEmbedDataModule(
            tokenizer="bert",
            transcriptome_processor="mlp",
            image_processor="uni2",
            dataset_names=f"hesteval_{organ}",
            batch_size=args.batch_size,
            train_fraction=0.0,
            use_disk_loading=True,
        )
        dm.processor.transcriptome_processor = custom_mlp
        _orig_hash = dm._compute_hash

        def _custom_hash(i=None, _h=_orig_hash):
            return _h(i) + "_hestgf17851"

        dm._compute_hash = _custom_hash
        dm.prepare_data()
        dm.setup()

        logger.info(
            "%d val sample datasets: %s",
            len(dm.val_datasets),
            ", ".join(str(ds.i) for ds in dm.val_datasets[:5])
            + ("..." if len(dm.val_datasets) > 5 else ""),
        )

        all_pred = []
        all_tgt = []
        sample_records = []
        for ds in tqdm(dm.val_datasets, desc=f"{organ} samples"):
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            n_in_sample = 0
            with torch.no_grad():
                for batch in loader:
                    patches_ctx = batch["patches_ctx"].to(device, non_blocking=True)
                    patches_cell = batch["patches_cell"].to(device, non_blocking=True)
                    tgt = batch["expression_expr"].to(device, non_blocking=True)
                    uni2 = decoder._get_uni2()
                    _, embeds = uni2(patches_ctx=patches_ctx, patches_cell=patches_cell)
                    pred = decoder(embeds)
                    all_pred.append(pred[:, panel_decoder_idx.to(device)].detach().cpu())
                    all_tgt.append(tgt[:, panel_decoder_idx.to(device)].detach().cpu())
                    n_in_sample += pred.shape[0]
            sample_records.append({"organ": organ, "sample_id": str(ds.i),
                                   "n_spots": n_in_sample})

        if not all_pred:
            logger.warning("No spots for %s, skipping", organ)
            continue
        pred_mat = torch.cat(all_pred, dim=0).to(torch.float32)
        tgt_mat = torch.cat(all_tgt, dim=0).to(torch.float32)
        logger.info("%s held-out matrix: %s spots × %d genes", organ,
                    pred_mat.shape[0], pred_mat.shape[1])

        per_gene_r = per_gene_pearson(pred_mat, tgt_mat)
        for g, r in zip(panel_gene_names, per_gene_r):
            per_gene_records.append({"organ": organ, "gene": g,
                                     "per_gene_r": float(r)})

        stats = bootstrap_median(per_gene_r, args.n_bootstrap)
        stats["n_samples"] = len(dm.val_datasets)
        stats["n_spots"] = int(pred_mat.shape[0])
        stats["n_panel_genes_used"] = len(panel_gene_names)
        stats["sample_ids"] = sorted(set(r["sample_id"] for r in sample_records))
        all_organ_summaries[organ] = stats
        all_rows.append({"organ": organ, **stats})

    # ── Save ─────────────────────────────────────────────────────────────────
    pd.DataFrame(per_gene_records).to_parquet(
        out_dir / "per_gene_hesteval.parquet", index=False
    )
    pd.DataFrame(all_rows).to_csv(out_dir / "per_organ_hesteval_table.csv", index=False)
    with open(out_dir / "per_organ_hesteval_summary.json", "w") as f:
        json.dump(all_organ_summaries, f, indent=2)

    logger.info("=" * 60)
    logger.info("Final per-organ summary:")
    logger.info(json.dumps(all_organ_summaries, indent=2))


if __name__ == "__main__":
    main()
