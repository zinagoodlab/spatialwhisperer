#!/usr/bin/env python3
"""Compare Lizard/PanNuke per-class AUROC across seeds."""

import pandas as pd
import numpy as np

base = "/home/groups/zinaida/moritzs/cellwhisperer_private/results/pathocell_evaluation"

for bench, subdir in [("Lizard", "lizard_summary"), ("PanNuke", "pannuke_summary")]:
    print(f"=== {bench} ===")
    rows = {}
    for seed_label, model in [
        ("seed0_orig", "spotwhisperer_cellxgene_census__archs4_geo__hest1k"),
        ("seed1", "spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed1"),
        ("seed2", "spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed2"),
    ]:
        try:
            df = pd.read_csv(
                f"{base}/{model}/{subdir}/patch_per_class_metrics_from_scores.csv"
            )
            rows[seed_label] = df.set_index("class_label")["rocauc"]
        except Exception as e:
            print(f"  {seed_label}: {e}")

    combined = pd.DataFrame(rows)
    print(
        f"  {'Class':30s}  {'seed0_orig':>10s}  {'seed1':>10s}  {'seed2':>10s}  {'std':>8s}"
    )
    for cls in combined.index:
        vals = combined.loc[cls].values
        line = f"  {cls:30s}"
        for v in vals:
            line += f"  {v:10.4f}"
        line += f"  {np.std(vals):8.4f}"
        print(line)
    means = combined.mean()
    line = f"  {'Mean':30s}"
    for v in means:
        line += f"  {v:10.4f}"
    line += f"  {np.std(means):8.4f}"
    print(line)
    print()
