"""
Collect aggregated metrics from all prompt templates + original baseline
into a single comparison CSV.

Inputs (Snakemake):
- snakemake.input.original: JSON with original (raw label) aggregated metrics
- snakemake.input.templates: list of JSONs, one per prompt template

Outputs:
- snakemake.output.csv: CSV with rows = prompt IDs, cols = metrics
"""

import json
from pathlib import Path

import pandas as pd

rows = []

# Original baseline
with open(snakemake.input.original) as f:
    d = json.load(f)
d["prompt_id"] = "original"
d["template"] = "{label}"
rows.append(d)

# Each template
for fp in snakemake.input.templates:
    fp = Path(fp)
    prompt_id = fp.parent.name.replace("_summary", "").replace("prompt_", "")
    with open(fp) as f:
        d = json.load(f)
    d["prompt_id"] = prompt_id
    rows.append(d)

df = pd.DataFrame(rows)
cols = ["prompt_id"] + [c for c in df.columns if c != "prompt_id"]
df = df[cols].sort_values("prompt_id")
df.to_csv(snakemake.output.csv, index=False)
