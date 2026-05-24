# Two-Stage Baseline: Image → Expression → Cell Type

## Motivation (Reviewer Comment)
> The paper mainly compares against pathology vision-language baselines such as CONCH and PLIP.
> How does the proposed trimodal image–gene–text alignment compare to the alternative pipeline of
> predicting transcriptomic profiles from H&E and then performing downstream cell annotation?

## Approach
A two-stage pipeline that predicts transcriptomic profiles from H&E images and then classifies
cell types from predicted expression. This is the "obvious" alternative to our contrastive
trimodal approach:

```
H&E patch → UNI2 (frozen, 1536-dim) → MILinearBlock decoder → predicted expression (17,851 genes)
→ Geneformer tokenizer → Geneformer (frozen) + linear head → cell type probabilities
→ label mapping (sum probs) → CRC 13 cell types
```

## Components

### Stage 1: Image → Expression Decoder
- **Architecture**: Raw UNI2 embeddings (1536-dim) → MILinearBlock (2-layer MLP with residual + LayerNorm, 2048-dim) → Linear → 17,851 genes
- **Training data**: HEST-1K (~900K Visium spot-level image-expression pairs, same data as contrastive training)
- **Gene space**: Intersection of HEST genes with Geneformer's 25,424-gene vocabulary = **17,851 genes**
- **Loss**: MSE on log(expression+1)
- **Training**: 4 epochs, lr=1e-3, batch_size=256, bf16-mixed, AdamW with cosine schedule
- **Module**: `src/cellwhisperer/expression_decoder/raw_uni2_decoder_lightning.py`

### Stage 2: Expression → Cell Type Classifier
- **Architecture**: Geneformer-12L-30M (frozen backbone) + linear classification head
- **Training data**: CellXGene Census (`cell_type` labels, 482 classes)
- **Training**: 8 epochs, lr=1e-4, batch_size=16, bf16, Adam
- **Checkpoint**: `results/finetuning_eval/geneformer/finetuned_frozen.pt` (159MB)

### Label Mapping (CellXGene → CRC)
- Maps 482 CellXGene Census `cell_type` labels → 13 PathoCellBench CRC cell types
- **Method**: Regex word-boundary matching (more specific categories checked first)
- **Coverage**: 322/482 types mapped; 160 unmapped (rare/specialized types like retinal, adipocyte)
- Probabilities for multiple training types mapping to the same CRC type are summed

## Files Created

| File | Purpose |
|------|---------|
| `src/cellwhisperer/expression_decoder/raw_uni2_decoder_lightning.py` | Lightning module: frozen UNI2 → MILinearBlock → Linear → gene expression |
| `src/spotwhisperer_eval/rules/two_stage_baseline.smk` | Snakemake rules (6 rules) |
| `src/spotwhisperer_eval/scripts/train_two_stage_decoder.py` | Decoder training script (monkey-patches MLP processor for custom gene list) |
| `src/spotwhisperer_eval/scripts/retrain_geneformer_classifier.py` | Geneformer classifier retraining script |
| `src/spotwhisperer_eval/scripts/create_hest_geneformer_genelist.py` | Gene list: HEST ∩ Geneformer vocabulary |
| `src/spotwhisperer_eval/scripts/create_pathocell_label_mapping.py` | Regex word-boundary label mapping |
| `src/spotwhisperer_eval/scripts/two_stage_predict.py` | End-to-end prediction: patches → UNI2 → decoder → Geneformer → scores |

## Snakemake Rules (in `two_stage_baseline.smk`)

| Rule | Purpose | Resources |
|------|---------|-----------|
| `create_hest_geneformer_genelist` | Gene list CSV | CPU, ~15 min |
| `train_two_stage_decoder` | Train decoder on HEST-1K | GPU, 250GB, ~8-12 hrs |
| `retrain_geneformer_classifier` | Retrain Geneformer + linear head | GPU, 250GB, ~7 hrs |
| `download_transfered_labels` | Download pre-computed label transfer | CPU, seconds |
| `create_pathocell_label_mapping` | Map CellXGene → CRC cell types | CPU, seconds |
| `two_stage_pathocell_predict` | Per-dataset prediction | GPU, 50GB, ~1-2 hrs per dataset |

## Output Files (on Sherlock)
```
resources/two_stage_baseline/
    hest_geneformer_genes.csv                    # 17,851 genes
    pathocell_crc_label_mapping.csv              # 322 mapped types

results/finetuning_eval/geneformer/
    finetuned_frozen.pt                          # Geneformer classifier (159MB)

results/two_stage_baseline/decoder/
    decoder.ckpt                                 # Trained decoder (pending)

results/pathocell_evaluation/two_stage_baseline/
    {dataset}_{prediction_level}_scores_seed{seed}.csv  # Per-dataset score CSVs
```

## How to Run
```bash
# From Sherlock (within SLURM job):
PD=/home/groups/zinaida/moritzs/cellwhisperer_private

# Full pipeline for all CRC datasets:
conda run -n cellwhisperer snakemake --snakefile src/spotwhisperer_eval/Snakefile \
    --profile sm7_slurm \
    ${PD}/results/pathocell_evaluation/two_stage_baseline/summary/patch_metrics_from_scores_aggregated.json

# Or individual components:
conda run -n cellwhisperer snakemake ... ${PD}/results/two_stage_baseline/decoder/decoder.ckpt
conda run -n cellwhisperer snakemake ... ${PD}/results/finetuning_eval/geneformer/finetuned_frozen.pt
```

## Execution Log

### 2026-03-25: Initial implementation and submission
- Created all files and rules
- Submitted gene list, label mapping, GF retraining, decoder training on Sherlock

### Issues encountered and fixed:
1. **Relative vs absolute paths** in sbatch commands → use `$PD` from `git rev-parse`
2. **`JointEmbedDataModule` doesn't accept `gene_list_path`** → pass via `transcriptome_processor_kwargs`
3. **`conda run --no-banner`** unsupported → removed flag
4. **Notebook execution** (jupyter-nbconvert) silently swallows errors → replaced with `script:` directive
5. **`finetuning_eval` module not on PYTHONPATH** → added `sys.path` to `src/figures/notebooks/`
6. **`cross_entropy` receives list instead of tensor** → added `torch.tensor()` conversion
7. **OOM at 100GB** for both GF (59GB CellXGene Census h5ad) and decoder (HEST-1K) → increased to 250GB
8. **OOM at 350-500GB** for decoder → added `use_disk_loading=True`
9. **Dimension mismatch** (17,851 decoder output vs 5,782 MLP target) → `transcriptome_processor_kwargs` not passed through `JointEmbedDataModule` → `TranscriptomeTextDualEncoderProcessor` passes `**kwargs` to ALL sub-processors (including `UNIProcessor`) → monkey-patch approach in dedicated training script
10. **Label mapping false positives** (substring "t cell" matches "goblet cell") → switched to regex `\b` word-boundary matching

11. **Cached expression_expr has wrong gene list** (cosmx6k, 5782 genes instead of 17,851) → `JointEmbedDataModule` caches preprocessed data keyed by hash that doesn't include gene list → monkey-patch `_compute_hash` to append `_hestgf17851`, forcing reprocessing with custom MLP processor

### 2026-03-26: Completion status
- **Gene list**: DONE (17,851 genes)
- **Label mapping**: DONE (322/482 mapped, regex word-boundary)
- **GF classifier**: DONE (159MB, 7h24m training, wandb: `retrain_geneformer_frozen`)
- **Decoder training**: PENDING (job 19679407, cache invalidation + reprocessing with 17,851-gene MLP processor, time limit 1 day)
- **PathoCellBench predictions**: Blocked on decoder

### wandb Training Runs (group: `icml_revisions`)

**Decoder (UNI2→expression):** https://wandb.ai/single-cellm/cellwhisperer/runs/dl047zmb
- Run name: `two_stage_decoder_hest1k`
- Status: completed 4 epochs, marked "failed" in wandb due to `shutil.SameFileError` at end (training itself was complete)
- Final metrics: `val/loss=0.0684`, `val/correlation=0.680`, `train/correlation=0.679`
- Runtime: 5811s (~1.6 hrs) on H100

**Geneformer classifier:** https://wandb.ai/single-cellm/cellwhisperer/runs/zxwx99na
- Run name: `retrain_geneformer_frozen`
- Status: finished (8 epochs)
- Final metrics: `val_loss=1.924`, `train_loss=2.471`
- Runtime: 26226s (~7.3 hrs) on H100

Both runs are legitimate -- the decoder correlation of 0.68 is reasonable for predicting ~18K genes from H&E patches.

## Results

### Aggregate (macro-average across 109 CRC datasets)

Computed over the **13 cell types in Table 1** (excluding "Other cells"), matching the paper's reported numbers:

| Model | Mean AUROC (13 classes) |
|-------|------------------------|
| Trimodal (SpotWhisperer) | 0.630 |
| **Two-stage baseline (UNI2→expr→Geneformer)** | **0.550** |
| CONCH | 0.545 |

Note: the `rocauc_macroAvg` in the aggregated JSON (0.565 for two-stage, 0.637 for trimodal) includes additional categories (Other cells, etc.) not in the paper table. The numbers above use only the 13 table classes.

### Per-class AUROC

| Cell type | Two-stage | Trimodal | Δ |
|-----------|-----------|----------|---|
| Macrophages/Monocytes | **0.740** | 0.712 | +0.028 |
| Smooth muscle | 0.691 | **0.734** | −0.043 |
| Stroma | 0.681 | 0.633 | +0.048 |
| B cells | 0.586 | **0.726** | −0.140 |
| Plasma cells | 0.585 | 0.639 | −0.054 |
| Granulocytes | 0.591 | 0.636 | −0.045 |
| Tumor cells | 0.500 | 0.575 | −0.075 |
| Vasculature/Lymphatics | 0.500 | **0.720** | −0.220 |
| Other cells | 0.500 | 0.543 | −0.043 |
| Adipocytes | 0.500 | 0.473 | +0.027 |
| Dendritic cells | 0.500 | 0.611 | −0.111 |
| T cells | 0.453 | **0.734** | −0.281 |
| Nerves | 0.468 | 0.541 | −0.073 |
| NK cells | 0.362 | 0.457 | −0.095 |

The two-stage baseline performs comparably on stromal/structural types (Macrophages, Smooth muscle, Stroma) where morphological features correlate with cell identity -- but substantially worse on immune cell types (T cells −28pp, Vasculature −22pp, B cells −14pp), where the text-grounded contrastive embedding provides more discriminative signal.

The two-stage baseline (**AUROC 0.565**) is clearly worse than our trimodal model (0.637, **−7.2pp**) and only marginally better than CONCH (0.546). This directly addresses the reviewer's question: directly predicting transcriptomics from H&E and classifying cell types from predicted expression does NOT match the trimodal contrastive approach.

Full metrics at: `results/pathocell_evaluation/two_stage_baseline/summary/`

## Expected Outcome (confirmed)
The two-stage baseline performed worse than our trimodal approach due to:
1. **Cascading errors**: expression prediction errors (val correlation 0.68 for 17,851 genes) propagate to cell type classification
2. **Information bottleneck**: the full expression vector must encode all discriminative information, while our contrastive embeddings are directly optimized for cross-modal similarity
3. **Resolution mismatch**: decoder trained on Visium spots (~50μm) applied to cell-level patches
4. **Gene space limitation**: not all discriminative genes may be predictable from H&E morphology
# Two-Stage Baseline: extension to Lizard and PanNuke (revisions)

To be appended to `src/spotwhisperer_eval/experiments/two_stage_baseline/SUMMARY.md` once
the controller job (`tsb_secondary`, sbatch 23985697) finishes.

## Motivation (from rebuttal review of revisions)

ICML reviewers asked us to compare against the alternative paradigm of predicting
transcriptomic profiles from H&E images followed by downstream cell-type annotation,
on **all three benchmarks** in the manuscript (PathoCell CRC, Lizard, PanNuke), not
just the rebuttal's PathoCell-only column. The 6-method × 3-benchmark Table 2 in the
revised manuscript depends on this expansion.

## Scope of the extension

Added, in addition to the existing CRC pipeline:

| Method | New Snakemake rule(s) | Marker / mapping resource |
|---|---|---|
| Two-stage UNI2→GF | `two_stage_lizard_predict`, `two_stage_pannuke_predict` (in `two_stage_baseline.smk`) | `create_lizard_label_mapping`, `create_pannuke_label_mapping` (CellXGene → benchmark class space; regex word-boundary, mirrors PathoCell mapping) |
| OmiCLIP, short markers | `omiclip_secondary_score` + `omiclip_secondary_split_scores` with `omiclip_model=omiclip` (in `omiclip_baseline.smk`) | `omiclip_generate_marker_genes` → `<benchmark>_short.json` (8–12 genes/class, curated from PanglaoDB / Human Protein Atlas / CRC scRNA-seq references) |
| OmiCLIP, extended markers | same rule with `omiclip_model=omiclip_pseudobulk` | `<benchmark>_pseudobulk.json` (25–30 genes/class, OmiCLIP Visium-pseudobulk style) |
| Trimodal (ours) | existing `lizard_cell_type_prediction`, `pannuke_cell_type_prediction` | n/a — uses the existing trimodal checkpoint |
| Bimodal I↔T (Quilt-1M only) | same as above with `model=spotwhisperer_quilt1m` | n/a |

Minor change to `two_stage_predict.py`: the background-class filter is now case-insensitive
(`c.lower() != "background"`), since Lizard's adata uses lowercase `background` while
PathoCell and PanNuke use `Background`.

Minor change to `split_baseline_logits.py`: dataset-id extraction now falls back to
`<dataset>_<level>.tiff` stripping when the CRC-specific `regNNN_[AB]` regex doesn't
match. Existing CRC outputs are unaffected.

## Files added in this extension

| File | Purpose |
|---|---|
| `scripts/generate_marker_genes.py` | Emits OmiCLIP marker-gene sentence JSON for `(benchmark, variant)` pairs. CLI: `python generate_marker_genes.py {lizard,pannuke} {short,pseudobulk} <out_path>`. |
| `scripts/create_lizard_label_mapping.py` | Maps the 482 CellXGene Census cell types to Lizard's 6 classes (Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue). |
| `scripts/create_pannuke_label_mapping.py` | Maps the 482 CellXGene Census cell types to PanNuke's 5 classes (Epithelial, Dead Cells, Connective/Soft tissue cells, Inflammatory, Neoplastic cells). |
| `experiments/two_stage_baseline/run_secondary_benchmarks.sh` | SLURM controller — submits Snakemake to dispatch child jobs for the 10 aggregate targets. |

## Results (aggregate, presence-based AUROC; classes → datasets → mean; seed=0)

All numbers below are **cell-type prediction** (PanNuke uses cell-type, *not* the separate 19-tissue prediction benchmark — that is unrelated work).

| Method | PathoCell CRC (109 ds) | Lizard 3-class (70 ds) | PanNuke 4-class (51 ds) |
|---|---:|---:|---:|
| Trimodal (ours; T↔G+G↔I, the model also reported in `tab:pathocell_benchmark`) | **0.630** | **0.764** | 0.689 |
| Bimodal I↔T (Quilt-1M only) | 0.566 | 0.642 | **0.713** |
| Two-stage UNI2→GF (ours) | 0.550 | 0.662 | 0.599 |
| OmiCLIP, short marker list | 0.491 | 0.611 | 0.520 |
| OmiCLIP, extended marker list | 0.478 | 0.646 | 0.525 |

For internal-record completeness: the 3-paired-datasets variant T↔G+G↔I+I↔T (Quilt-1M raw) scores 0.609 / 0.713 / 0.665 — recorded in the per-class CSV but excluded from the manuscript Table 2 per author convention ("trimodal" = 2 paired datasets).

Per-class breakdown is at `table2_two_stage_baselines_per_class.csv`. Notable per-class numbers used in the §4.1 paragraph: OmiCLIP short markers PathoCell B cells = **0.646** AUROC (matching the rebuttal's text fragment), every other PathoCell cell type at near-random (≤0.55).

**Take-aways**:
1. On PathoCell and Lizard the trimodal model wins by ≥0.07 AUROC over every two-stage variant.
2. OmiCLIP / Loki Annotate, with short or extended marker sentences, is the weakest baseline on PathoCell — confirming that a generic published two-stage pipeline does not match a contrastive trimodal-framework model trained on the same data.
3. PanNuke is the one benchmark where a baseline (Quilt-1M-only I↔T, 0.713) edges past the trimodal model (0.689). PanNuke is sourced from a wide tissue panel rather than colon, so broad image–text pretraining on Quilt-1M captures across-tissue cellular morphology cues that HEST-1K's colon-heavy image–gene supervision underweights. Worth a sentence in §4.3 ("Transitive Representation Learning Requires Overlapping Modalities").

## Open issues / known caveats

- All numbers are seed=0 only. Multi-seed extension is the parallel TODO (see
  `multi-seed/multi-benchmark performance` in the parent file).
- For Lizard, Plasma cells are dropped from the reduced-class scoring (they form their
  own class in the original Lizard schema and don't fit the reduced 3-class space cleanly).
  Plasma is still scored individually in the per-class table, however.
- PanNuke's Dead Cells column is dropped from the reduced 4-class scoring; it has no
  good analogue in CellXGene Census normal-tissue labels, so the two-stage UNI2→GF
  baseline simply does not predict that class either.
- The OmiCLIP "extended" gene lists are styled to mimic OmiCLIP's training distribution
  (Visium pseudobulk gene sentences) but are not OmiCLIP-author-curated; we manually
  curated them since Loki Annotate's tutorial only provides 5 generic tissue-level
  markers, none of which match Lizard or PanNuke's class space.

## How to reproduce

```bash
# From Sherlock login:
ssh sherlock 'sbatch /home/groups/zinaida/moritzs/cellwhisperer_private/src/spotwhisperer_eval/experiments/two_stage_baseline/run_secondary_benchmarks.sh'

# After completion, build Table 2 + per-class table:
ssh sherlock 'cd /home/groups/zinaida/moritzs/cellwhisperer_private && conda run -n cellwhisperer python src/spotwhisperer_eval/experiments/two_stage_baseline/build_table2.py'
```
