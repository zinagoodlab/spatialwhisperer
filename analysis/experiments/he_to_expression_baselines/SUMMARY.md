# H&E → Expression → Cell Type Baselines (GHIST, Loki, sCellST)

## Reviewer Comment (Round 2)
> Given the goal of the paper, I believe there is still an important alternative paradigm that
> deserves clearer discussion and, where possible, comparison: predicting gene expression /
> spatial transcriptomics from H&E images, and then using those predictions for downstream
> cell-type annotation or molecular interpretation. Since the present work, such as GHIST,
> Loki, and sCellST, ultimately also aims to recover fine-grained cellular semantics from
> histology, this is a highly relevant point of comparison. [...] Even if a full empirical
> comparison is beyond the scope of the revision, the paper would benefit from a clearer
> discussion of these related approaches, their differences, and the practical trade-offs.

## Context: What We Already Have

The **two-stage baseline** (`experiments/two_stage_baseline/`) already addresses the paradigm
of H&E → expression → cell type. It uses UNI2 → MLP decoder → 17,851 predicted genes →
Geneformer → cell type classification and achieves **AUROC 0.550** on 13 CRC cell types,
vs. our trimodal model at **0.630** and CONCH at **0.545**. This reviewer wants us to
additionally contextualize GHIST, Loki, and sCellST specifically.

---

## Paper Summaries

### 1. GHIST (Fu et al., Nature Methods, Sep 2025)
- **DOI**: 10.1038/s41592-025-02795-z
- **Code**: https://github.com/SydneyBioX/GHIST
- **What it does**: Multitask DL framework that predicts single-cell-resolution gene
  expression from H&E, trained on subcellular spatial transcriptomics (10x Xenium).
  Four prediction heads: (i) cell type, (ii) neighborhood composition, (iii) nuclei
  morphology, (iv) gene expression. Validated on breast cancer Xenium data.
- **Resolution**: Single-cell (trained on Xenium subcellular data, ~280 genes)
- **Training**: Requires matched H&E + subcellular ST (Xenium/MERFISH) for training.
  No pretrained weights available. Per-dataset training required.
- **Input**: H&E WSI + paired subcellular ST data for training; H&E only at inference.
- **Key result**: Cell-type accuracy 0.75 (breast, 8 classes), PCC 0.6-0.7 for top SVGs.

### 2. Loki/OmiCLIP (Chen et al., Nature Methods, May 2025)
- **DOI**: 10.1038/s41592-025-02707-1
- **Code**: https://github.com/GuangyuWangLab2021/Loki
- **What it does**: Visual-omics foundation model using CLIP-style contrastive learning
  between H&E patches and gene expression "sentences" (top-expressed gene symbols).
  Trained on 2.2M paired patches across 32 organs. Platform with 5 modules:
  tissue alignment, annotation (bulk RNA-seq / marker genes), cell-type decomposition
  (via scRNA-seq reference), image-transcriptomics retrieval, ST prediction.
- **Resolution**: Spot-level (Visium, ~55μm)
- **Training**: Pretrained weights available on HuggingFace. Zero-shot capable.
- **Input**: H&E patches (224×224); for decomposition needs scRNA-seq reference.
- **Key result**: Compared against 22 SOTA methods across 5 tasks on 28 datasets.

### 3. sCellST (Chadoutaud et al., Nature Communications, Jan 2026)
- **DOI**: 10.1038/s41467-025-67965-1
- **Code**: https://github.com/sysbio-curie/sCellST
- **What it does**: MIL-based weakly supervised framework. Segments nuclei (CellViT),
  extracts cell crops, learns SSL embeddings (MoCo v3), then uses MIL to predict
  per-cell expression from spot-level supervision. Spot = bag, cells = instances.
- **Resolution**: Single-cell (via MIL deconvolution from Visium spots)
- **Training**: Per-slide/dataset training. No pretrained weights. Uses HEST database.
- **Input**: H&E WSI + Visium data for training; H&E only at inference.
- **Key result**: Competitive with spot-level methods on kidney/prostate cancer,
  PCC ~0.15-0.20 for top HVGs.

---

## Applicability Assessment for PathoCellBench CRC Comparison

Our benchmark: 88,014 cell-level image patches from 109 CRC tissue images (35 patients),
13 cell types annotated by CODEX. Zero-shot classification task.

### GHIST — NOT FEASIBLE for empirical comparison

| Criterion | Assessment |
|-----------|------------|
| Pretrained model | **No** — must train from scratch |
| Training data needed | Matched H&E + subcellular ST (Xenium/MERFISH) for CRC tissue |
| CRC Xenium data available? | No public CRC Xenium dataset with matched H&E and cell-type annotations |
| Resolution match | Single-cell → matches our patch level, but wrong training data |
| Zero-shot capable | **No** — entirely supervised, per-dataset |
| Estimated effort | Weeks (find data, preprocess, train, evaluate) |
| **Verdict** | **Discussion only** |

**Why it's not a fair comparison**: GHIST requires paired H&E + Xenium data for training,
which we don't have for CRC. It cannot be applied zero-shot. Training it on breast Xenium
and applying to CRC would be out-of-domain and not representative of the method's intended use.
Furthermore, GHIST predicts ~280 genes (Xenium panel), not the full transcriptome.

### Loki/OmiCLIP — MOST FEASIBLE (but caveats)

| Criterion | Assessment |
|-----------|------------|
| Pretrained model | **Yes** — HuggingFace weights available |
| Training data needed | None for zero-shot; scRNA-seq reference for decomposition |
| Resolution match | Spot-level (Visium) → mismatch with cell-level patches |
| Zero-shot capable | **Partially** — annotation by marker genes or bulk RNA-seq possible |
| Estimated effort | ~2-3 days to set up and run |
| **Verdict** | **Feasible for empirical comparison, with important caveats** |

**Plan for Loki comparison**:
1. Install Loki + download pretrained OmiCLIP weights
2. Apply "Loki Annotate" (marker gene annotation) to PathoCellBench CRC patches
   - Provide CRC cell type marker gene lists
   - Let Loki score each patch against marker-defined cell types
3. Apply "Loki Decompose" with a CRC scRNA-seq reference
   - Use a CRC scRNA-seq reference (e.g., from CellXGene) 
   - Get per-patch cell type decomposition scores
4. Compare AUROC against our trimodal model

**Caveats**:
- Loki was trained on Visium spots (~10-50 cells per patch). Our PathoCellBench patches
  are single-cell level. This is a resolution mismatch that disadvantages Loki.
- Loki's "annotation" uses cosine similarity between patch embeddings and gene-sentence
  embeddings. At single-cell resolution, there's much less transcriptomic signal per patch.
- This comparison is somewhat unfair to Loki but addresses the reviewer's question directly.

### sCellST — NOT FEASIBLE for empirical comparison

| Criterion | Assessment |
|-----------|------------|
| Pretrained model | **No** — must train per dataset |
| Training data needed | H&E + Visium for CRC tissue (for training) |
| CRC Visium data available? | Some in HEST-1K, but not matched to PathoCellBench |
| Resolution match | Single-cell (via MIL) → conceptually matches |
| Zero-shot capable | **No** — fully supervised MIL |
| Architecture compatibility | Expects spots (bags) containing cells (instances), not individual cell patches |
| Estimated effort | ~1 week (find CRC Visium data, train, design evaluation protocol) |
| **Verdict** | **Discussion only** |

**Why it's not a fair comparison**: sCellST's MIL architecture fundamentally requires
spot-level bags containing multiple cells. Our PathoCellBench evaluation provides individual
cell patches. Even if we trained sCellST on CRC Visium data, applying it to individual cell
patches would require a completely different inference pipeline. Also, sCellST has no
pretrained model, so any comparison would be on a different (Visium-based) CRC dataset,
not on PathoCellBench, making it apples-to-oranges.

---

## Summary Table

| Method | Empirical comparison? | Rationale |
|--------|----------------------|-----------|
| **Two-stage baseline** | **DONE** (AUROC 0.550) | Already implemented and reported |
| **Loki/OmiCLIP** | **FEASIBLE** (2-3 days) | Pretrained model, zero-shot capable |
| **GHIST** | Discussion only | No pretrained model, needs Xenium CRC data |
| **sCellST** | Discussion only | No pretrained model, MIL architecture mismatch |

---

## Execution Plan

### Phase 1: Loki Empirical Comparison (2-3 days)

1. **Setup** (~0.5 day)
   - Clone Loki repo, install in conda env on Sherlock
   - Download OmiCLIP pretrained weights from HuggingFace
   - Verify basic functionality with their tutorial

2. **PathoCellBench inference** (~1 day)
   - Load PathoCellBench CRC patches (same 88K patches used in paper)
   - Option A: **Loki Annotate** — define CRC cell type marker gene lists, compute
     cosine similarity between OmiCLIP image embeddings and gene-sentence embeddings
     for each cell type
   - Option B: **Loki Decompose** — provide a CRC scRNA-seq reference, get cell-type
     decomposition scores per patch
   - Option C: **Loki PredEx** — predict ST gene expression per patch, then classify
     (essentially another two-stage variant, but using Loki's trained decoder)

3. **Evaluation** (~0.5 day)
   - Compute per-class and macro-average AUROC on 13 CRC cell types
   - Compare with trimodal (0.630), two-stage (0.550), CONCH (0.545)
   - Save results to `results/pathocell_evaluation/loki/`

4. **Snakemake integration** (~0.5 day)
   - Add rules to `rules/` (or extend `two_stage_baseline.smk`)
   - Document in this SUMMARY.md

### Phase 2: Discussion for Paper (concurrent with Phase 1)

Draft a ~300-word paragraph for the paper discussion/rebuttal covering:

1. **Paradigm distinction**: Our trimodal approach vs. the H&E→expression→cell-type pipeline
   - We align modalities contrastively; they predict expression and then classify
   - Our approach avoids cascading errors and information bottleneck
   - Our approach is zero-shot; theirs requires per-dataset training (GHIST, sCellST)
     or at minimum a reference dataset (Loki)

2. **Practical trade-offs**:
   - H&E→expression methods can predict continuous gene expression (richer output)
   - Our method provides direct cell-type scores via text prompts (more flexible)
   - H&E→expression methods require matched training data at appropriate resolution
   - Our transitive learning avoids the need for paired H&E-text data

3. **Resolution considerations**:
   - GHIST/sCellST achieve single-cell resolution but need corresponding training data
   - Loki operates at spot level (like us)
   - All methods face the fundamental challenge that morphology only partially predicts expression

4. **Evidence from our two-stage baseline**:
   - AUROC 0.550 vs trimodal 0.630 demonstrates the cascading-error problem
   - Works for morphologically distinctive types but fails for immune subtypes
   - (If Loki results available: add those numbers)

### Key Discussion Points to Emphasize

- The reviewer's suggested paradigm (H&E→expression→cell type) is exactly what our
  two-stage baseline tests. Result: 7.2pp worse than trimodal.
- GHIST/sCellST/Loki are complementary, not competing: they excel at predicting
  spatial gene expression (their primary goal), while our model excels at zero-shot
  semantic annotation via text.
- The fundamental issue with H&E→expression→cell type is that gene expression
  prediction from morphology is inherently noisy (best reported: PCC ~0.6-0.7 for
  top SVGs), and this noise cascades into classification errors.
- Our contrastive approach sidesteps this by learning discriminative features directly,
  avoiding the intermediate expression prediction step.

---

## Execution Log

### 2026-04-02: Initial assessment
- Identified and read all three papers (GHIST, Loki/OmiCLIP, sCellST)
- Assessed feasibility for PathoCellBench comparison
- Loki is the only method feasible for empirical comparison (pretrained + zero-shot)
- GHIST and sCellST require per-dataset training with specific paired data we don't have
- Created execution plan: Loki comparison + discussion section

### 2026-04-02: Implementation
- Created `scripts/run_omiclip_baseline.py`: scores PathoCellBench patches using
  OmiCLIP pretrained weights + marker-gene sentences (same paradigm as Loki Annotate)
- Created `rules/omiclip_baseline.smk`: 3 rules (download checkpoint, score, split)
- Output reuses existing `pathocell_metrics_from_scores` pipeline (model name = `omiclip`)
- Added `include` to main Snakefile

### How to run
```bash
# From Sherlock (within SLURM job):
PD=/home/groups/zinaida/moritzs/cellwhisperer_private

# Full pipeline (download + score + split + aggregate metrics):
conda run -n cellwhisperer snakemake --snakefile src/spotwhisperer_eval/Snakefile \
    --profile sm7_slurm \
    ${PD}/results/pathocell_evaluation/omiclip/summary/patch_metrics_from_scores_aggregated.json
```

Note: `open_clip` and `tifffile` must be installed in the `cellwhisperer` conda env.
Install if missing: `conda run -n cellwhisperer pip install open_clip_torch tifffile`

---

## Reviewer Response Draft

We thank the reviewer for the thoughtful follow-up. We agree that this intuitive baseline needs to be discussed in a dedicated subsection in the revised paper, which we will do.

*When and why should our approach be preferred?* The core issue with the H&E → expression → cell type paradigm is related to the fact that histology images are underspecified in representing the transcriptome state.
Therefore, in a two-step prediction scenario, the intermediary transcriptome exhibits substantial noise, which the downstream classifier ingests blindly.
Our approach mitigates this issue through a shared embedding space, where the encoders jointly learn representations that are relevant and focused for downstream tasks.

To answer the reviewer's question directly: Our approach is better suited for predictions at the cellular level (e.g. cell types), as opposed to the prediction of one or few specific markers.

In addition to our two-stage baseline *we now also include the performance of OmiCLIP*. OmiCLIP's zero-shot annotation pipeline (Loki Annotate) perform zero-shot cell type annotation by scoring similarity of H&E patches against cell type marker genes.

Despite testing two sets of representative marker genes, following their pipeline and recommendations, OmiCLIP's approach only recognized B cells at meaningful performance (0.646 AUROC) whereas other cell types were near random, reinforcing the importance of learning all three aspects jointly: image, expression state, and annotation.

| Model                        | AUROC |
|------------------------------|-------|
| Trimodal (ours)              | 0.630 |
| Two-stage baseline (UNI2→GF) | 0.550 |
| CONCH                        | 0.545 |
| OmiCLIP (short marker list)  | 0.488 |
| OmiCLIP (expanded list)      | 0.480 |


The revised manuscript will include this and the two-stage baseline, scored for all three benchmarks (PathoCell, PanNuke, Lizard), alongside our guiding interpretations.


We will include this analysis and interpretations in the revised manuscript


**Why does OmiCLIP underperform?** 
OmiCLIP is a visual-omics foundation model trained on 2.2M paired H&E patches and gene expression "sentences" via contrastive learning (CoCa ViT-L-14). For annotation, it encodes cell-type-specific gene lists through its text encoder and computes cosine similarity with image patch embeddings.

We verified that our implementation performed correct patch cropping, model loading, and used Loki's own code patterns.
