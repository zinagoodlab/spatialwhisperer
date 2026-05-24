#!/usr/bin/env python3
"""Quick comparison of original seed-0 model scores vs retrained seed-0."""

import pandas as pd
import numpy as np
from pathlib import Path

orig_dir = Path(
    "/home/groups/zinaida/moritzs/cellwhisperer_private/results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k"
)
new_dir = Path(
    "/home/groups/zinaida/moritzs/cellwhisperer_private/results/pathocell_evaluation/spotwhisperer_cellxgene_census__archs4_geo__hest1k_seed0"
)

# Find datasets present in both
orig_files = {f.name for f in orig_dir.glob("*_scores_seed0.csv")}
new_files = {f.name for f in new_dir.glob("*_scores_seed0.csv")}
common = sorted(orig_files & new_files)[:5]  # compare first 5

print(
    f"Common score files: {len(orig_files & new_files)} (comparing first {len(common)})\n"
)

for fn in common:
    orig = pd.read_csv(orig_dir / fn)
    new = pd.read_csv(new_dir / fn)
    ds = fn.replace("_patch_scores_seed0.csv", "")
    print(f"--- {ds} ---")
    print(f"  Shape: orig={orig.shape}, new={new.shape}")
    if orig.shape == new.shape and list(orig.columns) == list(new.columns):
        diff = orig.values.astype(float) - new.values.astype(float)
        print(f"  Max abs diff:  {np.max(np.abs(diff)):.8f}")
        print(f"  Mean abs diff: {np.mean(np.abs(diff)):.8f}")
        print(
            f"  Identical (atol=1e-5): {np.allclose(orig.values, new.values, atol=1e-5)}"
        )
        print(
            f"  Identical (atol=1e-2): {np.allclose(orig.values, new.values, atol=1e-2)}"
        )
    else:
        print(
            f"  Column mismatch! orig={list(orig.columns)[:3]}, new={list(new.columns)[:3]}"
        )
