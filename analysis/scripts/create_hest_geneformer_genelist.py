"""
Create a gene list CSV containing the intersection of HEST-1K genes and
Geneformer's vocabulary. This gene list defines the target space for the
image→expression decoder in the two-stage baseline.
"""

import pyarrow  # needed

import anndata
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Geneformer's gene dictionaries
from geneformer.tokenizer import TranscriptomeTokenizer

tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={}, nproc=1)
geneformer_ensembl_ids = set(tokenizer.genelist_dict.keys())
logger.info(f"Geneformer vocabulary: {len(geneformer_ensembl_ids)} Ensembl gene IDs")

# Load the Ensembl ↔ symbol mapping
from cellwhisperer.config import get_path

annot_path = get_path(["paths", "ensembl_gene_symbol_map"])
annot = pd.read_csv(annot_path, index_col=0)

# Build symbol → ensembl mapping (uppercased symbols)
symbol_to_ensembl = {}
for symbol, row in annot.iterrows():
    symbol_to_ensembl[str(symbol).upper()] = row["ensembl_gene_id"]

# Scan HEST-1K h5ad files to collect all gene symbols
hest_h5ad_dir = Path(snakemake.input.hest_dir)
h5ad_files = sorted(hest_h5ad_dir.glob("*.h5ad"))
logger.info(f"Found {len(h5ad_files)} HEST h5ad files in {hest_h5ad_dir}")

all_genes = set()
for f in h5ad_files[:50]:  # sample subset for speed (genes are mostly shared)
    adata = anndata.read_h5ad(f, backed="r")
    gene_names = adata.var.index.astype(str).str.upper().tolist()
    all_genes.update(gene_names)
    adata.file.close()

logger.info(f"Total unique gene symbols across sampled HEST files: {len(all_genes)}")

# Map to Ensembl IDs and intersect with Geneformer vocabulary
valid_genes = []
for gene_symbol in sorted(all_genes):
    ensembl_id = symbol_to_ensembl.get(gene_symbol, "")
    if ensembl_id in geneformer_ensembl_ids:
        valid_genes.append({"gene_name": gene_symbol, "ensembl_id": ensembl_id})

logger.info(f"Genes in HEST ∩ Geneformer: {len(valid_genes)}")

# Save
df = pd.DataFrame(valid_genes)
df.to_csv(snakemake.output.gene_list, index=False)
logger.info(f"Saved gene list to {snakemake.output.gene_list}")
