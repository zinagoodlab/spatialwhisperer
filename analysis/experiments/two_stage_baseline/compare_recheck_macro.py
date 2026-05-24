#!/usr/bin/env python3
"""Compare macro AUROCs between canonical and recheck score dirs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_table2 import (
    load_gt_for_dir, load_scores,
    auroc_crc, auroc_lizard, auroc_pannuke,
    RESULTS, H5AD_BASE,
)

crc_gt = load_gt_for_dir(H5AD_BASE)
liz_gt = load_gt_for_dir(H5AD_BASE / "lizard")
pan_gt = load_gt_for_dir(H5AD_BASE / "pannuke")

PAIRS = [
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k",           "crc",     crc_gt, auroc_crc),
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m",  "crc",     crc_gt, auroc_crc),
    ("spotwhisperer_quilt1m",                                        "crc",     crc_gt, auroc_crc),
    ("two_stage_baseline",                                           "crc",     crc_gt, auroc_crc),
    ("omiclip",                                                      "crc",     crc_gt, auroc_crc),
    ("omiclip_pseudobulk",                                           "crc",     crc_gt, auroc_crc),
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k",           "lizard",  liz_gt, auroc_lizard),
    ("spotwhisperer_cellxgene_census__archs4_geo__hest1k",           "pannuke", pan_gt, auroc_pannuke),
]

print(f"{'model':60s}  {'bench':7s}  {'canonical':>10s}  {'recheck':>10s}  {'|Δ|':>10s}")
for m, bench, gt, scorer in PAIRS:
    can_dir = RESULTS / m / ("" if bench == "crc" else bench)
    rec_dir = RESULTS / f"{m}_recheck" / ("" if bench == "crc" else bench)
    can_macro, _ = scorer(load_scores(can_dir), gt)
    rec_macro, _ = scorer(load_scores(rec_dir), gt)
    print(f"{m[:60]:60s}  {bench:7s}  {can_macro:10.4f}  {rec_macro:10.4f}  {abs(can_macro-rec_macro):10.6f}")
