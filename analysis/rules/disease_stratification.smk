# Disease-stratified evaluation of PanNuke cell type prediction, and
# zero-shot tissue type (organ) prediction on PanNuke.
#
# See: analysis/experiments/disease_stratification/SUMMARY.md

# ── Disease-stratified cell type prediction ──────────────────────────────────

rule pannuke_disease_stratified_metrics:
    """
    Compute disease-stratified cell type prediction metrics on PanNuke.

    Uses PLIP's classification: malignant (>=10 neoplastic cells, >=30% fraction)
    vs benign (0 neoplastic cells). Ambiguous patches are discarded.
    Dead Cells class is dropped; Neoplastic cells is additionally dropped for
    the benign stratum (no ground-truth positives).
    """
    input:
        scores=expand(
            PATHOCELL_RESULTS / "{{model}}" / "pannuke/{dataset}_patch_scores_seed0.csv",
            dataset=PANNUKE_DATASETS,
        ),
        adatas=expand(
            PATHOCELL_DATA / "processed/pannuke/{dataset}_patch.h5ad",
            dataset=PANNUKE_DATASETS,
        ),
    output:
        per_class=PATHOCELL_RESULTS / "{model}" / "pannuke_summary/disease_stratified_per_class.csv",
        summary=PATHOCELL_RESULTS / "{model}" / "pannuke_summary/disease_stratified_summary.csv",
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/compute_disease_stratified_metrics.py"


# ── Tissue type prediction ───────────────────────────────────────────────────

PANNUKE_TYPES_NPY = PATHOCELL_DATA / "pannuke_fold2_types.npy"
PANNUKE_SPLITS_CSV = PATHOCELL_DATA / "raw/data/pannuke/splits/train_test_val_split.csv"

rule build_pannuke_tissue_mapping:
    """Build (batch, patch) -> (sample, tissue_type) mapping for PanNuke test set."""
    input:
        splits_csv=PANNUKE_SPLITS_CSV,
        types_npy=PANNUKE_TYPES_NPY,
        adatas=expand(
            PATHOCELL_DATA / "processed/pannuke/{dataset}_patch.h5ad",
            dataset=PANNUKE_DATASETS,
        ),
    output:
        mapping=PATHOCELL_DATA / "processed/pannuke/tissue_mapping.csv",
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/build_pannuke_tissue_mapping.py"


rule score_pannuke_tissue_type:
    """Score a single PanNuke batch for tissue type prediction (zero-shot)."""
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        adata=PATHOCELL_DATA / "processed/pannuke/{dataset}_patch.h5ad",
        image=PATHOCELL_DATA / "processed/pannuke/{dataset}_patch.tiff",
    output:
        scores=PATHOCELL_RESULTS / "{model}" / "pannuke_tissue/{dataset}_tissue_scores.csv",
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=8, time="1:00:00")
    script:
        "../scripts/score_pannuke_tissue_type.py"


rule compute_tissue_type_metrics:
    """
    Aggregate tissue type scores across batches, compute per-tissue AUROC + confusion matrix.
    Runs twice: once on all patches, once on benign-only patches.
    """
    input:
        scores=expand(
            PATHOCELL_RESULTS / "{{model}}" / "pannuke_tissue/{dataset}_tissue_scores.csv",
            dataset=PANNUKE_DATASETS,
        ),
        mapping=rules.build_pannuke_tissue_mapping.output.mapping,
        adatas=expand(
            PATHOCELL_DATA / "processed/pannuke/{dataset}_patch.h5ad",
            dataset=PANNUKE_DATASETS,
        ),
    output:
        per_class=PATHOCELL_RESULTS / "{model}" / "pannuke_summary/tissue_type_{subset}_per_class.csv",
        summary=PATHOCELL_RESULTS / "{model}" / "pannuke_summary/tissue_type_{subset}_summary.csv",
        confusion=PATHOCELL_RESULTS / "{model}" / "pannuke_summary/tissue_type_{subset}_confusion.csv",
    params:
        benign_only=lambda wildcards: wildcards.subset == "benign_only",
    wildcard_constraints:
        subset="(all|benign_only)",
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/compute_tissue_type_metrics.py"
