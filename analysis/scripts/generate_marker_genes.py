#!/usr/bin/env python3
"""
Emit OmiCLIP marker-gene "sentences" JSON for a given benchmark and variant.

Variants:
  short      ~8-12 genes per class (compact marker sentence)
  pseudobulk ~25-30 genes per class (expanded sentence; matches Loki / OmiCLIP style)

Benchmarks:
  lizard   classes match obsm["cell_type_counts_coarse"] columns of
           resources/pathocell/processed/lizard/*_patch.h5ad:
             ['background', 'Neutrophil', 'Epithelial', 'Lymphocyte',
              'Plasma', 'Eosinophil', 'Connective tissue']
  pannuke  classes match obsm["cell_type_counts_coarse"] columns of
           resources/pathocell/processed/pannuke/*_patch.h5ad:
             ['Background', 'Epithelial', 'Dead Cells',
              'Connective/Soft tissue cells', 'Inflammatory', 'Neoplastic cells']

The CRC variant lives in run_omiclip_baseline.py (CRC_MARKER_GENES) and
generate_pseudobulk_genes.py (CRC_PSEUDOBULK) and is intentionally not duplicated
here.

Usage:
  python generate_marker_genes.py <benchmark> <variant> <out_path>
"""

import json
import sys


# ── Lizard ────────────────────────────────────────────────────────────────────
# Original Lizard classes: background + 6 cell types (Neutrophil, Epithelial,
# Lymphocyte, Plasma, Eosinophil, Connective tissue). Reduced 3-class scoring
# (Epithelial, Leukocyte={Neutro+Lympho+Eosino}, Fibroblast={Connective tissue})
# is performed downstream by compute_reduced_class_table2_style.py.
#
# Marker selection sources: PanglaoDB, Human Protein Atlas, Lee 2020 (CRC),
# Pelka 2021 (CRC), and the existing CRC_MARKER_GENES / CRC_PSEUDOBULK in this
# repo (consistent style with OmiCLIP's Visium training distribution).

LIZARD_SHORT = {
    "background": "COL1A1 COL1A2 COL3A1 FN1 VIM SPARC DCN LUM",
    "Neutrophil": "S100A8 S100A9 CSF3R CXCR2 FCGR3B MPO ELANE CTSG CEACAM8",
    "Epithelial": "EPCAM KRT8 KRT18 KRT19 KRT20 CDH1 VIL1 CDX2 MUC2 CEACAM5",
    "Lymphocyte": "CD3D CD3E CD2 IL7R LCK CD79A MS4A1 CD19 CD22 CD8A",
    "Plasma": "JCHAIN MZB1 XBP1 IGHG1 IGHA1 SDC1 PRDM1 IRF4 CD38",
    "Eosinophil": "EPX RNASE2 RNASE3 PRG2 IL5RA CCR3 SIGLEC8 CLC IL5",
    "Connective tissue": "FAP PDGFRA PDGFRB COL1A1 COL3A1 THY1 ACTA2 VIM DCN LUM",
}

LIZARD_PSEUDOBULK = {
    "background": "COL1A1 COL1A2 COL3A1 FN1 VIM SPARC DCN LUM BGN COL6A1 COL6A2 COL5A1 COL5A2 POSTN THBS2 CTGF MFAP5 FBLN1 FBLN2 AEBP1 COL14A1 PCOLCE MMP2 TIMP1 TIMP3 SERPINH1 FSTL1 COL4A1 COL4A2 NID1 LAMA4 LAMB1",
    "Neutrophil": "S100A8 S100A9 S100A12 CEACAM8 FCGR3B CSF3R CXCR2 MMP9 ELANE MPO CTSG AZU1 DEFA3 BPI CAMP LTF PRTN3 LCN2 CXCR1 FPR1 FPR2 SELL ITGAM CD177 ALPL CEBPE SPI1",
    "Epithelial": "EPCAM KRT8 KRT18 KRT19 KRT20 CDH1 MUC2 CEACAM5 CDX2 FABP1 PIGR TFF3 OLFM4 LGR5 ASCL2 REG4 DEFA5 DEFA6 AGR2 FCGBP ZG16 CLCA1 ITLN1 MUC13 SPINK4 GPA33 CEACAM6 CEACAM7 VIL1",
    "Lymphocyte": "CD3D CD3E CD3G CD2 TRAC TRBC1 TRBC2 IL7R LCK CD8A CD8B CD4 CD27 CD28 ICOS GZMK CCL5 NKG7 GZMA PRF1 IFNG CD79A CD79B MS4A1 CD19 BANK1 IGHM IGHD CD22 BLK PAX5",
    "Plasma": "JCHAIN MZB1 XBP1 IGHG1 IGHG2 IGHG3 IGHG4 IGHA1 IGHA2 SDC1 PRDM1 IRF4 SLAMF7 TNFRSF17 CD38 SSR4 DERL3 SEC11C FKBP11 TXNDC5 HSPA5 MYDGF IGKC IGLC2 IGLC3",
    "Eosinophil": "EPX RNASE2 RNASE3 PRG2 PRG3 IL5RA CCR3 SIGLEC8 CLC IL5 CD9 ALOX15 GATA2 GATA1 EPO MBP RETN HRH4 SLC18A2 LTC4S CHIT1 CSF2RB GPR44 IL3RA HDC IDO1",
    "Connective tissue": "FAP PDGFRA PDGFRB THY1 COL1A1 COL3A1 ACTA2 VIM FN1 DCN LUM BGN POSTN THBS2 MMP2 MMP3 CXCL12 CXCL14 CCL2 IL6 WNT2 WNT5A RSPO3 BMP4 SFRP2 SFRP4 DKK3",
}


# ── PanNuke ───────────────────────────────────────────────────────────────────
# Original PanNuke classes: Background + 5 cell types (Epithelial, Dead Cells,
# Connective/Soft tissue cells, Inflammatory, Neoplastic cells). Reduced 4-class
# scoring drops Dead Cells.
#
# Note: PanNuke spans 19 tissues, so "Epithelial" markers are kept broad
# (multi-tissue keratins) rather than colon-specific.

PANNUKE_SHORT = {
    "Background": "COL1A1 COL1A2 COL3A1 FN1 VIM SPARC DCN LUM",
    "Epithelial": "EPCAM KRT8 KRT18 KRT19 CDH1 KRT5 KRT14 KRT7 KRT17",
    "Dead Cells": "CASP3 CASP7 BAX PARP1 H2AFX HMGB1 BIRC5 ANXA5 BCL2",
    "Connective/Soft tissue cells": "FAP PDGFRA PDGFRB COL1A1 COL3A1 THY1 ACTA2 VIM DCN LUM",
    "Inflammatory": "PTPRC CD68 CD163 CD3D CD79A MS4A1 S100A8 S100A9 CD8A",
    "Neoplastic cells": "TP53 EPCAM KRAS EGFR MYC MKI67 KRT8 KRT19 CEACAM5 CDX2",
}

PANNUKE_PSEUDOBULK = {
    "Background": "COL1A1 COL1A2 COL3A1 FN1 VIM SPARC DCN LUM BGN COL6A1 COL6A2 COL5A1 COL5A2 POSTN THBS2 CTGF MFAP5 FBLN1 FBLN2 AEBP1 COL14A1 PCOLCE MMP2 TIMP1 TIMP3 SERPINH1 FSTL1 COL4A1 COL4A2 NID1 LAMA4 LAMB1",
    "Epithelial": "EPCAM KRT8 KRT18 KRT19 CDH1 KRT5 KRT14 KRT7 KRT17 KRT4 KRT13 KRT15 KRT6A KRT6B KRT16 KRT20 CDX2 VIL1 CLDN3 CLDN4 OCLN TJP1 MUC1 MUC2 SFN PERP CEBPB GRHL2 ELF3 EHF",
    "Dead Cells": "CASP3 CASP7 CASP6 CASP8 CASP9 BAX BAK1 BAD BID BIK BCL2 BCL2L1 PARP1 H2AFX HMGB1 BIRC5 ANXA5 ATG5 ATG7 BECN1 PUMA NOXA TP53AIP1 ENDOG AIFM1 CYCS DIABLO CFLAR FAS FASLG TNFRSF10A",
    "Connective/Soft tissue cells": "FAP PDGFRA PDGFRB THY1 COL1A1 COL3A1 ACTA2 VIM FN1 DCN LUM BGN POSTN THBS2 MMP2 MMP3 CXCL12 CXCL14 CCL2 IL6 WNT2 WNT5A RSPO3 BMP4 SFRP2 SFRP4 DKK3 S100A4 SPARC TAGLN",
    "Inflammatory": "PTPRC CD68 CD163 CD14 ITGAM AIF1 TYROBP CSF1R CD3D CD3E CD2 TRAC IL7R LCK CD79A CD79B MS4A1 CD19 BANK1 CD8A CD8B CD4 GNLY NKG7 KLRD1 KLRB1 NCAM1 S100A8 S100A9 CXCR2 CCR2 CXCL10",
    "Neoplastic cells": "TP53 EPCAM KRAS EGFR MYC MKI67 PCNA TOP2A CCND1 CCNE1 BIRC5 SURVIVIN AURKA AURKB CEACAM5 CEACAM6 CEACAM7 CDX2 KRT8 KRT18 KRT19 KRT20 CDH1 MUC1 MUC2 BRCA1 BRCA2 PIK3CA AKT1 PTEN APC",
}


REGISTRY = {
    ("lizard", "short"): LIZARD_SHORT,
    ("lizard", "pseudobulk"): LIZARD_PSEUDOBULK,
    ("pannuke", "short"): PANNUKE_SHORT,
    ("pannuke", "pseudobulk"): PANNUKE_PSEUDOBULK,
}


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    benchmark = sys.argv[1]
    variant = sys.argv[2]
    out_path = sys.argv[3]
    if (benchmark, variant) not in REGISTRY:
        print(
            f"Unknown (benchmark, variant) = ({benchmark!r}, {variant!r}). "
            f"Choices: {sorted(REGISTRY.keys())}",
            file=sys.stderr,
        )
        sys.exit(2)

    genes = REGISTRY[(benchmark, variant)]

    from open_clip import get_tokenizer

    tokenizer = get_tokenizer("coca_ViT-L-14")
    for name, sentence in genes.items():
        tokens = tokenizer([sentence])
        n_nonzero = (tokens != 0).sum().item()
        n_genes = len(sentence.split())
        truncated = " (TRUNCATED!)" if n_nonzero >= 76 else ""
        print(
            f"  {name:35s}: {n_genes:3d} genes, {n_nonzero:3d}/76 tokens{truncated}"
        )

    with open(out_path, "w") as f:
        json.dump(genes, f, indent=2)
    print(f"Saved {len(genes)} classes for ({benchmark}, {variant}) to {out_path}")


if __name__ == "__main__":
    main()
