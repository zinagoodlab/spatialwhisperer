# PathoCellBench evaluation pipeline for cell type classification
# This pipeline downloads PathoCell dataset, processes it into CellWhisperer format,
# and evaluates cell type prediction performance

from pathlib import Path as _Path

# TODO all processed files should use PATHOCELL_RESULTS (currently most use PATHOCELL_DATA)
PATHOCELL_RESULTS = PROJECT_DIR / "results/pathocell_evaluation"
PATHOCELL_DATA = PROJECT_DIR / "resources/pathocell"  # this is for downloaded files only
BASELINES_DIR = _Path(srcdir("../static/baselines_animesh"))  # PLIP/CONCH/MUSK logit CSVs
PATHOCELL_MODEL_RESULTS = PATHOCELL_RESULTS / "{model}"

# _hdf_dir = PATHOCELL_DATA / "raw/pathocell_hdf"
# DATASETS = sorted([p.stem for p in _hdf_dir.glob("*.hdf")])
DATASETS = ["reg006_B", "reg014_B", "reg022_B", "reg030_A", "reg037_B", "reg046_A", "reg056_A", "reg007_A", "reg015_A", "reg023_A", "reg030_B", "reg038_A", "reg047_A", "reg058_A", "reg007_B", "reg016_A", "reg023_B", "reg031_A", "reg039_A", "reg048_A", "reg059_A", "reg001_A", "reg008_A", "reg016_B", "reg024_B", "reg031_B", "reg039_B", "reg048_B", "reg059_B", "reg001_B", "reg008_B", "reg017_A", "reg025_A", "reg032_A", "reg040_A", "reg049_A", "reg060_A", "reg002_A", "reg009_A", "reg017_B", "reg025_B", "reg032_B", "reg040_B", "reg050_A", "reg060_B", "reg002_B", "reg009_B", "reg018_A", "reg026_A", "reg033_A", "reg041_A", "reg050_B", "reg061_A", "reg003_A", "reg010_A", "reg018_B", "reg026_B", "reg033_B", "reg041_B", "reg051_A", "reg062_A", "reg003_B", "reg010_B", "reg019_A", "reg027_A", "reg034_A", "reg042_A", "reg051_B", "reg063_A", "reg004_A", "reg011_A", "reg020_A", "reg027_B", "reg035_A", "reg042_B", "reg052_A", "reg064_A", "reg004_B", "reg011_B", "reg020_B", "reg028_A", "reg035_B", "reg043_A", "reg052_B", "reg065_A", "reg005_A", "reg012_A", "reg021_A", "reg028_B", "reg036_A", "reg044_A", "reg053_A", "reg066_A", "reg005_B", "reg012_B", "reg021_B", "reg029_A", "reg036_B", "reg045_A", "reg054_A", "reg067_A", "reg006_A", "reg013_B", "reg022_A", "reg029_B", "reg037_A", "reg045_B", "reg055_A", "reg068_A"]

# Lizard benchmark datasets (H&E colon histology, 6 cell types).
# Authoritative list at parse time = the test split shipped in static/, so
# downstream rules can be planned before convert_lizard_lmdb_to_hdf has run.
# After conversion, the sample_list.txt produced by that rule matches verbatim.
_lizard_hdf_dir = PATHOCELL_DATA / "converted/lizard_hdf"
_lizard_sample_list = _lizard_hdf_dir / "sample_list.txt"
_lizard_static_list = PROJECT_DIR / "analysis/static/eval_test_splits/lizard_test_samples.txt"
if _lizard_sample_list.exists():
    LIZARD_DATASETS = sorted(_lizard_sample_list.read_text().strip().split("\n"))
elif _lizard_hdf_dir.exists():
    LIZARD_DATASETS = sorted([p.stem for p in _lizard_hdf_dir.glob("*.hdf")])
elif _lizard_static_list.exists():
    LIZARD_DATASETS = sorted(_lizard_static_list.read_text().strip().split("\n"))
else:
    LIZARD_DATASETS = []

# PanNuke benchmark datasets (H&E multi-tissue histology, 5 cell types + background).
# Static list = 51 batch names (batch_00000, batch_00050, ..., batch_02500) deterministic
# from the 2523-row test split + batch_size=50 in convert_lmdb_to_hdf.py.
_pannuke_hdf_dir = PATHOCELL_DATA / "converted/pannuke_hdf"
_pannuke_sample_list = _pannuke_hdf_dir / "sample_list.txt"
_pannuke_static_list = PROJECT_DIR / "analysis/static/eval_test_splits/pannuke_test_batches.txt"
if _pannuke_sample_list.exists():
    PANNUKE_DATASETS = sorted(_pannuke_sample_list.read_text().strip().split("\n"))
elif _pannuke_hdf_dir.exists():
    PANNUKE_DATASETS = sorted([p.stem for p in _pannuke_hdf_dir.glob("*.hdf")])
elif _pannuke_static_list.exists():
    PANNUKE_DATASETS = sorted(_pannuke_static_list.read_text().strip().split("\n"))
else:
    PANNUKE_DATASETS = []


# NOTE: LUL_identity still had the adapter for `geneformer`. It's also the first one to load the "correct" temperature from the start
CONCH_MODEL_CONFIGS = ["frozen", "LLL", "LUL", "LUL_identity"] # TODO could also try others (NOTE might need to freeze things)

# ── Optimized label remaps ─────────────────────────────────────────────────────
# Determined via quick label variant experiment (quick_label_variants.py) on
# 5 samples per benchmark, testing 12 text-query alternatives.
#
# Lizard: original labels are near-optimal; only "Lymphocyte"→"T cells" helps
#   AUC (+0.08) but hurts F1.  We keep original labels (no remap).
# PanNuke: renaming three classes gives large gains (F1 0.22→0.33, AUC 0.66→0.71):
#   "Neoplastic cells"→"cancer cells", "Inflammatory"→"leukocytes",
#   "Connective/Soft tissue cells"→"fibroblasts"
#
# These remaps are applied to the main prediction rules below via
# filter_background=True and label_remap params.

BEST_LABEL_REMAP = {
    "lizard": {
        "Neutrophil": "neutrophil",
        "Epithelial": "epithelial",
        "Lymphocyte": "lymphocyte",
        "Plasma": "plasma",
        "Eosinophil": "eosinophil",
        "Connective tissue": "fibroblast",
    },
    "pannuke": {
        "Epithelial": "epithelial",
        "Dead Cells": "dead cells",
        "Neoplastic cells": "cancer cells",
        "Inflammatory": "leukocytes",
        "Connective/Soft tissue cells": "fibroblasts",
    },
}

# ── Label variant experiments (for ablation) ──────────────────────────────────
# Each variant defines a label_remap dict per benchmark. Keys are original class
# names (from the AnnData), values are alternative text queries.
# All variants also set filter_background=True (background is never informative).
#
# Lizard original (no background): Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue
# PanNuke original (no background): Epithelial, Dead Cells, Connective/Soft tissue cells, Inflammatory, Neoplastic cells

LABEL_VARIANTS = {
    # v0: just drop background, keep original labels
    "v0_nobg": {
        "lizard": {},
        "pannuke": {},
    },
    # v2: Lymphocyte → T cells (Lizard only change)
    "v2_tcells": {
        "lizard": {"Lymphocyte": "T cells"},
        "pannuke": {},
    },
    # v3: connective tissue → stroma
    "v3_stroma": {
        "lizard": {"Connective tissue": "stroma"},
        "pannuke": {"Connective/Soft tissue cells": "stroma"},
    },
    # v4: connective tissue → stromal cells
    "v4_stromal_cells": {
        "lizard": {"Connective tissue": "stromal cells"},
        "pannuke": {"Connective/Soft tissue cells": "stromal cells"},
    },
    # v5: connective tissue → fibroblasts
    "v5_fibroblasts": {
        "lizard": {"Connective tissue": "fibroblasts"},
        "pannuke": {"Connective/Soft tissue cells": "fibroblasts"},
    },
    # v6a: Inflammatory → inflamed cells
    "v6a_inflamed": {
        "lizard": {},
        "pannuke": {"Inflammatory": "inflamed cells"},
    },
    # v6b: Inflammatory → immune cells
    "v6b_immune": {
        "lizard": {},
        "pannuke": {"Inflammatory": "immune cells"},
    },
    # v6c: Inflammatory → leukocytes
    "v6c_leukocytes": {
        "lizard": {},
        "pannuke": {"Inflammatory": "leukocytes"},
    },
    # v7a: Neoplastic cells → cancer cells
    "v7a_cancer": {
        "lizard": {},
        "pannuke": {"Neoplastic cells": "cancer cells"},
    },
    # v7b: Neoplastic cells → tumor cells
    "v7b_tumor": {
        "lizard": {},
        "pannuke": {"Neoplastic cells": "tumor cells"},
    },
    # v8: combined best labels (= BEST_LABEL_REMAP for pannuke, best-guess for lizard)
    "v8_combined": {
        "lizard": {
            "Lymphocyte": "T cells",
            "Connective tissue": "stromal cells",
        },
        "pannuke": {
            "Connective/Soft tissue cells": "fibroblasts",
            "Inflammatory": "leukocytes",
            "Neoplastic cells": "cancer cells",
        },
    },
}

def _get_label_remap(wildcards):
    """Return the label_remap dict for a given benchmark and terms_id."""
    return LABEL_VARIANTS[wildcards.terms_id].get(wildcards.benchmark, {})



rule train_conch:
    """
    Train a SpotWhisperer model for a dataset_combo.
    Uses the base config and overrides dataset names; outputs a checkpoint.
    Ensures subsampled datasets exist when dataset_combo includes *thsub suffixes.

    TODO could still check learning rate...
    """
    input:
        base_config=PROJECT_DIR / "src/experiments/conch_finetuning_testing/finetune_conch_adapters.yaml"
    output:
        model=protected(PROJECT_DIR / config["paths"]["jointemb_models"] / "conch_{locking_mode}{identity_projection}.ckpt")
    wildcard_constraints:
        identity_projection="(|_identity)",
        locking_mode="(...|frozen)"
    params:
        test_run_config="--trainer.limit_train_batches 500 --trainer.max_epochs 2" if config.get("fast", False) else "",
        identity_projection_config=lambda wildcards: "--model.model_config.identity_projection true" if wildcards.identity_projection == "_identity" else "",  # TODO this one does not work
        seed=SEEDS[0],
        project_dir=PROJECT_DIR,
    resources:
        mem_mb=lambda wildcards: 150000,
        slurm=slurm_gres("large", num_cpus=12, time="70:00:00", num_gpus=1)
    shell: """
        cd {params.project_dir}

        cellwhisperer fit \
            --config {input.base_config} \
            {params.test_run_config} \
            {params.identity_projection_config} \
            --seed_everything {params.seed} \
            --last_model_path {output.model} \
            --omit_validation_functions \
            --wandb conch_finetuning_{wildcards.train_config} \
            --model.model_config.locking_mode {wildcards.locking_mode}
    """


rule pathocell_download_dataset:
    """
    Download PathoCell dataset from HuggingFace.
    Downloads the HDF5 format which contains images, masks, and cell type annotations.
    """
    output:
        dataset_marker=touch(PATHOCELL_DATA / "download_complete.marker"),
        data_dir=directory(PATHOCELL_DATA / "raw")
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=2"
    log:
        "logs/pathocell_download_dataset.log"
    shell: """
        LOG=$(realpath {log})
        mkdir -p {output.data_dir}
        
        # Download PathoCell dataset using huggingface-cli
        echo "Downloading PathoCell dataset from HuggingFace..." > $LOG
        
        # Download the dataset files
        huggingface-cli download \
            Kainmueller-Lab/PathoCell \
            --repo-type dataset \
            --local-dir {output.data_dir} \
            --local-dir-use-symlinks False \
            2>&1 | tee -a $LOG
        
        echo "Download complete" >> $LOG
    """


rule pathocell_process_data:
    """
    Process PathoCell data into CellWhisperer format.
    Converts a PathoCell HDF file to AnnData with spatial coordinates.

    The raw files (`CT_mapping*.txt`, `{dataset}.hdf`) live inside the
    `data_dir` produced by `pathocell_download_dataset` (declared as a
    Snakemake `directory()`), so per-file outputs aren't tracked. We
    therefore depend on the directory output as the build trigger and
    pass the specific file paths via `params:` — keeps the DAG planable
    without enumerating 109 hdf outputs in the download rule.
    """
    input:
        dataset_marker=rules.pathocell_download_dataset.output.dataset_marker,
        data_dir=rules.pathocell_download_dataset.output.data_dir,
    output:
        adata=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.tiff",
        metadata=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}_metadata.json"
    params:
        patch_level=lambda wildcards: wildcards.prediction_level=="patch",
        ct_mapping_fine=lambda wildcards: PATHOCELL_DATA / "raw/pathocell_hdf/CT_mapping.txt",
        ct_mapping_coarse=lambda wildcards: PATHOCELL_DATA / "raw/pathocell_hdf/CT_coarse_mapping.txt",
        hdf_file=lambda wildcards: PATHOCELL_DATA / f"raw/pathocell_hdf/{wildcards.dataset}.hdf",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=2 partition=cmackall"
    log:
        notebook="logs/pathocell_process_data_{dataset}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/pathocell_process_data.py.ipynb"



rule pathocell_cell_type_prediction:
    """
    Run cell type prediction and evaluation using CellWhisperer model.
    Uses score_left_vs_right() or get_performance_metrics_left_vs_right()
    for evaluation. Supports both cell-level and patch-level prediction.

    TODO should be refactored to only export scores
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        adata=rules.pathocell_process_data.output.adata,
        image=rules.pathocell_process_data.output.image
    output:
        results=PATHOCELL_MODEL_RESULTS / "{dataset}_{prediction_level}_prediction_seed{seed}.json",
        per_class_metrics=PATHOCELL_MODEL_RESULTS / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
        confusion_matrix=PATHOCELL_MODEL_RESULTS / "{dataset}_{prediction_level}_confusion_seed{seed}.csv",
        scores=PATHOCELL_MODEL_RESULTS / "{dataset}_{prediction_level}_scores_seed{seed}.csv",  # only valid for patches so far
    params:
        prediction_level="{prediction_level}",
    threads: 8
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+",
        seed="\\d+",
        model="[^/]+",   # exclude subdir paths so this rule doesn't match lizard/<ds> or pannuke/<ds>
    resources:
        mem_mb=150000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    log:
        notebook="logs/pathocell_cell_type_prediction_{model}_{dataset}_{prediction_level}_seed{seed}.ipynb"
    notebook:
        "../notebooks/pathocell_cell_type_prediction.py.ipynb"


rule pathocell_metrics_from_scores:
    """
    Aggregate metrics across datasets from stored scores and AnnData.
    Returns: aggregated JSON, per-class CSV (mean across datasets/seeds), per-dataset CSV (seed-averaged).
    """
    input:
        scores=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
            model=wildcards.model,
            dataset=DATASETS,
            prediction_level=wildcards.prediction_level,
            seed=SEEDS,
            allow_missing=True,
        ),
        adatas=lambda wildcards: expand(
            PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.h5ad",
            dataset=DATASETS,
            prediction_level=wildcards.prediction_level,
        ),
    output:
        aggregated=PATHOCELL_RESULTS / "{model}" / "summary" / "{prediction_level}_metrics_from_scores_aggregated.json",
        per_class=PATHOCELL_RESULTS / "{model}" / "summary" / "{prediction_level}_per_class_metrics_from_scores.csv",
        per_dataset=PATHOCELL_RESULTS / "{model}" / "summary" / "{prediction_level}_per_dataset_metrics_from_scores.csv",
        per_class_by_dataset=PATHOCELL_RESULTS / "{model}" / "summary" / "{prediction_level}_per_class_by_dataset_metrics_from_scores.csv",
    params:
        prediction_level="{prediction_level}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model="[^/]+",
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1 partition=cmackall"
    script:
        "../scripts/compute_pathocell_metrics_from_scores.py"


rule pathocell_aggregate_results:
    """
    Aggregate results across multiple seeds for a model.
    """
    input:
        results=lambda wildcards: expand(
            rules.pathocell_cell_type_prediction.output.results,
            model=wildcards.model,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS
        ),
    output:
        summary=PATHOCELL_MODEL_RESULTS / "summary" / "{prediction_level}_classification_summary.json"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    log:
        notebook="logs/pathocell_aggregate_results_{model}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/pathocell_aggregate_results.py.ipynb"


rule aggregate_pathocell_results:
    """
    Copy aggregated PathoCellBench summaries into the benchmarks directory for dataset_combo.
    This rule is used by the spider plot to access PathoCellBench results.

    """
    input:
        performance_summary=lambda wildcards: expand(
            rules.pathocell_aggregate_results.output.summary,
            model="spatialwhisperer_{}".format(wildcards.dataset_combo),
            prediction_level="cell",  # Default to cell-level for backwards compatibility
            allow_missing=True,
        )
    output:
        aggregated_pathocell=BENCHMARKS_DIR / "pathocell" / "{dataset_combo}" / "performance_summary.json"
    wildcard_constraints:
        dataset_combo="[^/]+"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    shell: """
        mkdir -p $(dirname {output.aggregated_pathocell})
        cp {input.performance_summary} {output.aggregated_pathocell}
    """


# ============================================================================
# Lizard benchmark (PathoCellBench benchmark 2/3)
# H&E colon histology with 6 cell types + background (7 total):
#   background, Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue
# Data converted from LMDB to: resources/pathocell/converted/lizard_hdf/*.hdf
# (downloaded together with the CRC data via pathocell_download_dataset)
# ============================================================================

rule lizard_process_data:
    """
    Process Lizard dataset HDF file into CellWhisperer format.
    Converts a Lizard HDF file to AnnData with spatial coordinates.
    Cell types: background, Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue

    Inputs the `sample_list.txt` produced by `convert_lizard_lmdb_to_hdf` as a
    build trigger (its `output_dir` is a `directory()` so per-file outputs
    aren't tracked); the actual hdf path is passed via `params`.
    """
    input:
        sample_list=_lizard_sample_list,
    output:
        adata=PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}.tiff",
        metadata=PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}_metadata.json"
    params:
        patch_level=lambda wildcards: wildcards.prediction_level == "patch",
        hdf_file=lambda wildcards: PATHOCELL_DATA / f"converted/lizard_hdf/{wildcards.dataset}.hdf",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=2 partition=cmackall"
    log:
        notebook="logs/lizard_process_data_{dataset}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/lizard_process_data.py.ipynb"


rule lizard_cell_type_prediction:
    """
    Run cell type prediction and evaluation on Lizard dataset using CellWhisperer model.
    Uses optimized label remap (BEST_LABEL_REMAP) and filters background class.
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        adata=rules.lizard_process_data.output.adata,
        image=rules.lizard_process_data.output.image
    output:
        results=PATHOCELL_RESULTS / "{model}" / "lizard/{dataset}_{prediction_level}_prediction_seed{seed}.json",
        per_class_metrics=PATHOCELL_RESULTS / "{model}" / "lizard/{dataset}_{prediction_level}_per_class_seed{seed}.csv",
        confusion_matrix=PATHOCELL_RESULTS / "{model}" / "lizard/{dataset}_{prediction_level}_confusion_seed{seed}.csv",
        scores=PATHOCELL_RESULTS / "{model}" / "lizard/{dataset}_{prediction_level}_scores_seed{seed}.csv",
    params:
        prediction_level="{prediction_level}",
        filter_background=True,
        label_remap=BEST_LABEL_REMAP["lizard"],
    threads: 8
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+",
        seed="\\d+"
    resources:
        mem_mb=150000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    log:
        notebook="logs/lizard_cell_type_prediction_{model}_{dataset}_{prediction_level}_seed{seed}.ipynb"
    notebook:
        "../notebooks/pathocell_cell_type_prediction.py.ipynb"


rule lizard_metrics_from_scores:
    """
    Aggregate Lizard metrics across datasets from stored scores and AnnData.
    Returns: aggregated JSON, per-class CSV (mean across datasets/seeds), per-dataset CSV (seed-averaged).
    """
    input:
        scores=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "lizard/{dataset}_{prediction_level}_scores_seed{seed}.csv",
            model=wildcards.model,
            dataset=LIZARD_DATASETS,
            prediction_level=wildcards.prediction_level,
            seed=SEEDS,
            allow_missing=True,
        ),
        adatas=lambda wildcards: expand(
            PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}.h5ad",
            dataset=LIZARD_DATASETS,
            prediction_level=wildcards.prediction_level,
        ),
    output:
        aggregated=PATHOCELL_RESULTS / "{model}" / "lizard_summary" / "{prediction_level}_metrics_from_scores_aggregated.json",
        per_class=PATHOCELL_RESULTS / "{model}" / "lizard_summary" / "{prediction_level}_per_class_metrics_from_scores.csv",
        per_dataset=PATHOCELL_RESULTS / "{model}" / "lizard_summary" / "{prediction_level}_per_dataset_metrics_from_scores.csv",
        per_class_by_dataset=PATHOCELL_RESULTS / "{model}" / "lizard_summary" / "{prediction_level}_per_class_by_dataset_metrics_from_scores.csv",
    params:
        prediction_level="{prediction_level}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model="[^/]+",
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/compute_pathocell_metrics_from_scores.py"


rule lizard_aggregate_results:
    """
    Aggregate Lizard results across multiple seeds for a model.
    """
    input:
        results=lambda wildcards: expand(
            rules.lizard_cell_type_prediction.output.results,
            model=wildcards.model,
            prediction_level=wildcards.prediction_level,
            dataset=LIZARD_DATASETS,
            seed=SEEDS
        ),
    output:
        summary=PATHOCELL_MODEL_RESULTS / "lizard_summary" / "{prediction_level}_classification_summary.json"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    log:
        notebook="logs/lizard_aggregate_results_{model}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/pathocell_aggregate_results.py.ipynb"


# ============================================================================
# PanNuke benchmark (PathoCellBench benchmark 3/3)
# H&E multi-tissue histology with 5 cell types (+background):
#   Background, Epithelial, Dead Cells, Connective/Soft tissue cells, Inflammatory, Neoplastic cells
# Data converted from LMDB to: resources/pathocell/converted/pannuke_hdf/*.hdf
# (downloaded together with the CRC data via pathocell_download_dataset)
# ============================================================================

rule pannuke_process_data:
    """
    Process PanNuke dataset HDF file into CellWhisperer format.
    Converts a PanNuke HDF file to AnnData with spatial coordinates.
    Cell types: Background, Epithelial, Dead Cells, Connective/Soft tissue cells, Inflammatory, Neoplastic cells

    Same `directory()` indirection as `lizard_process_data` — depend on the
    convert rule's sample_list as a build trigger; pass the actual hdf path
    via `params`.
    """
    input:
        sample_list=_pannuke_sample_list,
    output:
        adata=PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}.tiff",
        metadata=PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}_metadata.json"
    params:
        patch_level=lambda wildcards: wildcards.prediction_level == "patch",
        hdf_file=lambda wildcards: PATHOCELL_DATA / f"converted/pannuke_hdf/{wildcards.dataset}.hdf",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=2 partition=cmackall"
    log:
        notebook="logs/pannuke_process_data_{dataset}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/pannuke_process_data.py.ipynb"


rule pannuke_cell_type_prediction:
    """
    Run cell type prediction and evaluation on PanNuke dataset using CellWhisperer model.
    Uses optimized label remap (BEST_LABEL_REMAP) and filters background class.
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        adata=rules.pannuke_process_data.output.adata,
        image=rules.pannuke_process_data.output.image
    output:
        results=PATHOCELL_RESULTS / "{model}" / "pannuke/{dataset}_{prediction_level}_prediction_seed{seed}.json",
        per_class_metrics=PATHOCELL_RESULTS / "{model}" / "pannuke/{dataset}_{prediction_level}_per_class_seed{seed}.csv",
        confusion_matrix=PATHOCELL_RESULTS / "{model}" / "pannuke/{dataset}_{prediction_level}_confusion_seed{seed}.csv",
        scores=PATHOCELL_RESULTS / "{model}" / "pannuke/{dataset}_{prediction_level}_scores_seed{seed}.csv",
    params:
        prediction_level="{prediction_level}",
        filter_background=True,
        label_remap=BEST_LABEL_REMAP["pannuke"],
    threads: 8
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+",
        seed="\\d+"
    resources:
        mem_mb=150000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    log:
        notebook="logs/pannuke_cell_type_prediction_{model}_{dataset}_{prediction_level}_seed{seed}.ipynb"
    notebook:
        "../notebooks/pathocell_cell_type_prediction.py.ipynb"


rule pannuke_metrics_from_scores:
    """
    Aggregate PanNuke metrics across datasets from stored scores and AnnData.
    Returns: aggregated JSON, per-class CSV (mean across datasets/seeds), per-dataset CSV (seed-averaged).
    """
    input:
        scores=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "pannuke/{dataset}_{prediction_level}_scores_seed{seed}.csv",
            model=wildcards.model,
            dataset=PANNUKE_DATASETS,
            prediction_level=wildcards.prediction_level,
            seed=SEEDS,
            allow_missing=True,
        ),
        adatas=lambda wildcards: expand(
            PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}.h5ad",
            dataset=PANNUKE_DATASETS,
            prediction_level=wildcards.prediction_level,
        ),
    output:
        aggregated=PATHOCELL_RESULTS / "{model}" / "pannuke_summary" / "{prediction_level}_metrics_from_scores_aggregated.json",
        per_class=PATHOCELL_RESULTS / "{model}" / "pannuke_summary" / "{prediction_level}_per_class_metrics_from_scores.csv",
        per_dataset=PATHOCELL_RESULTS / "{model}" / "pannuke_summary" / "{prediction_level}_per_dataset_metrics_from_scores.csv",
        per_class_by_dataset=PATHOCELL_RESULTS / "{model}" / "pannuke_summary" / "{prediction_level}_per_class_by_dataset_metrics_from_scores.csv",
    params:
        prediction_level="{prediction_level}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model="[^/]+",
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/compute_pathocell_metrics_from_scores.py"


rule pannuke_aggregate_results:
    """
    Aggregate PanNuke results across multiple seeds for a model.
    """
    input:
        results=lambda wildcards: expand(
            rules.pannuke_cell_type_prediction.output.results,
            model=wildcards.model,
            prediction_level=wildcards.prediction_level,
            dataset=PANNUKE_DATASETS,
            seed=SEEDS
        ),
    output:
        summary=PATHOCELL_MODEL_RESULTS / "pannuke_summary" / "{prediction_level}_classification_summary.json"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    log:
        notebook="logs/pannuke_aggregate_results_{model}_{prediction_level}.ipynb"
    notebook:
        "../notebooks/pathocell_aggregate_results.py.ipynb"


# ── Label variant experiment rules ─────────────────────────────────────────────
# These rules run the same prediction pipeline but with remapped text queries
# and background filtering. Output paths include the terms_id wildcard.

def _label_variant_adata(wildcards):
    """Return the processed AnnData path for a benchmark dataset."""
    return PATHOCELL_DATA / f"processed/{wildcards.benchmark}/{wildcards.dataset}_{{prediction_level}}.h5ad".format(
        prediction_level=wildcards.prediction_level
    )

def _label_variant_image(wildcards):
    """Return the processed image path for a benchmark dataset."""
    return PATHOCELL_DATA / f"processed/{wildcards.benchmark}/{wildcards.dataset}_{{prediction_level}}.tiff".format(
        prediction_level=wildcards.prediction_level
    )

rule label_variant_prediction:
    """
    Run cell type prediction with alternative text labels for a benchmark dataset.
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        adata=_label_variant_adata,
        image=_label_variant_image,
    output:
        results=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}/{dataset}_{prediction_level}_prediction_seed{seed}.json",
        per_class_metrics=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}/{dataset}_{prediction_level}_per_class_seed{seed}.csv",
        confusion_matrix=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}/{dataset}_{prediction_level}_confusion_seed{seed}.csv",
        scores=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}/{dataset}_{prediction_level}_scores_seed{seed}.csv",
    params:
        prediction_level="{prediction_level}",
        filter_background=True,
        label_remap=_get_label_remap,
    threads: 8
    wildcard_constraints:
        prediction_level="(cell|patch)",
        dataset="[^/]+",
        seed="\\d+",
        benchmark="(lizard|pannuke)",
        terms_id="[a-z0-9_]+",
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    log:
        notebook="logs/label_variant_{benchmark}_{terms_id}_{model}_{dataset}_{prediction_level}_seed{seed}.ipynb"
    notebook:
        "../notebooks/pathocell_cell_type_prediction.py.ipynb"


rule label_variant_metrics:
    """
    Aggregate metrics for a label variant experiment.
    """
    input:
        scores=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}/{dataset}_{prediction_level}_scores_seed{seed}.csv",
            model=wildcards.model,
            benchmark=wildcards.benchmark,
            terms_id=wildcards.terms_id,
            dataset=LIZARD_DATASETS if wildcards.benchmark == "lizard" else PANNUKE_DATASETS,
            prediction_level=wildcards.prediction_level,
            seed=SEEDS,
        ),
        adatas=lambda wildcards: expand(
            PATHOCELL_DATA / "processed/{benchmark}/{dataset}_{prediction_level}.h5ad",
            benchmark=wildcards.benchmark,
            dataset=LIZARD_DATASETS if wildcards.benchmark == "lizard" else PANNUKE_DATASETS,
            prediction_level=wildcards.prediction_level,
        ),
    output:
        aggregated=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}_summary" / "{prediction_level}_metrics_from_scores_aggregated.json",
        per_class=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}_summary" / "{prediction_level}_per_class_metrics_from_scores.csv",
        per_dataset=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}_summary" / "{prediction_level}_per_dataset_metrics_from_scores.csv",
        per_class_by_dataset=PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}_summary" / "{prediction_level}_per_class_by_dataset_metrics_from_scores.csv",
    params:
        prediction_level="{prediction_level}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model="[^/]+",
        benchmark="(lizard|pannuke)",
        terms_id="[a-z0-9_]+",
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/compute_pathocell_metrics_from_scores.py"


rule label_variant_all:
    """
    Target rule to run all label variant experiments for a model and benchmark.
    """
    input:
        metrics=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{benchmark}_terms_{terms_id}_summary" / "patch_metrics_from_scores_aggregated.json",
            model=wildcards.model,
            benchmark=wildcards.benchmark,
            terms_id=list(LABEL_VARIANTS.keys()),
        ),
    output:
        touch(PATHOCELL_RESULTS / "{model}" / "{benchmark}_label_variants_done.flag"),
    wildcard_constraints:
        benchmark="(lizard|pannuke)",


# Main rule to run all PathoCellBench evaluation
rule pathocell_compare_models:
    """
    Statistical comparison between two models based on aggregated *_values and per-class metrics.
    Performs t-tests and effect sizes; outputs CSVs and a summary JSON.
    """
    input:
        model_a_summary=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_a}/summary/{wildcards.prediction_level}_classification_summary.json",
        model_b_summary=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_b}/summary/{wildcards.prediction_level}_classification_summary.json",
        per_class_a=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_a,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
        per_class_b=lambda wildcards: expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_{prediction_level}_per_class_seed{seed}.csv",
            model=wildcards.model_b,
            prediction_level=wildcards.prediction_level,
            dataset=DATASETS,
            seed=SEEDS,
            allow_missing=True,
        ),
    output:
        metrics_comparison_csv=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "{model_a}_vs_{model_b}_metrics.csv",
        per_class_comparison_csv=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "{model_a}_vs_{model_b}_per_class.csv",
        summary_json=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "{model_a}_vs_{model_b}_summary.json",
    params:
        model_a="{model_a}",
        model_b="{model_b}",
        prediction_level="{prediction_level}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model_a="[^/]+",
        model_b="[^/]+",
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/compare_pathocell_models.py"

rule pathocell_per_class:
    """
    Create per-class plots of delta-score (model_a - model_b) for a given metric.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        per_class_by_dataset_a=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_a}/summary/{wildcards.prediction_level}_per_class_by_dataset_metrics_from_scores.csv",
        per_class_by_dataset_b=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_b}/summary/{wildcards.prediction_level}_per_class_by_dataset_metrics_from_scores.csv",
        # Ensure statistical comparison CSV exists for significance coloring
        comparison_csv=lambda wildcards: PATHOCELL_RESULTS / "comparison" / f"{wildcards.prediction_level}" / f"{wildcards.model_a}_vs_{wildcards.model_b}_per_class.csv",
    output:
        plot=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "plots" / "per_class__{metric}__{model_a}_vs_{model_b}.svg",
    params:
        model_a="{model_a}",
        model_b="{model_b}",
        prediction_level="{prediction_level}",
        metric="{metric}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model_a="[^/]+",
        model_b="[^/]+",
        # Limit metric to avoid overlap with scatter outputs (no underscores)
        metric="[A-Za-z0-9@]+|soft_rocauc",
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/plot_pathocell_perclass.py"

rule pathocell_performance_overview:
    """
    Scatterplot of per-class mean F1 between two models.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        # Aggregated per-class CSVs from metrics_from_scores
        per_class_a=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_a}/summary/{wildcards.prediction_level}_per_class_metrics_from_scores.csv",
        per_class_b=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_b}/summary/{wildcards.prediction_level}_per_class_metrics_from_scores.csv",
        # Per-dataset CSVs from metrics_from_scores for patch-level distribution metrics
        results_a=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_a}/summary/{wildcards.prediction_level}_per_dataset_metrics_from_scores.csv",
        results_b=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_b}/summary/{wildcards.prediction_level}_per_dataset_metrics_from_scores.csv",
    output:
        plot=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "plots" / "performance_overview_{model_a}_vs_{model_b}__{metric}.svg",
    params:
        model_a="{model_a}",
        model_b="{model_b}",
        prediction_level="{prediction_level}",
        plot_type="scatter",
        scatter_unit="dataset",  # 'class' or 'dataset'
        metric="{metric}"  #  "mean_cross_entropy" also looks good, but not as much
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model_a="[^/]+",
        model_b="[^/]+",
    resources:
        mem_mb=6000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/pathocell_performance_overview.py"

rule pathocell_baselines_vs_trimodal:
    """
    Per-class metrics comparison for a single metric: Trimodal vs Quilt1m vs CONCH variants using metrics_from_scores per-class CSVs.
    Metric selected via wildcard 'metric'.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        # Use per-class metrics (seed/dataset aggregated) from metrics_from_scores
        bibridge_per_class=PATHOCELL_RESULTS / "spatialwhisperer_cellxgene_census__archs4_geo__hest1k/summary/patch_per_class_metrics_from_scores.csv",
        quilt_per_class=PATHOCELL_RESULTS / "spatialwhisperer_quilt1m/summary/patch_per_class_metrics_from_scores.csv",
        conch_LLL_per_class=PATHOCELL_RESULTS / "conch_LLL/summary/patch_per_class_metrics_from_scores.csv",
        conch_LUL_per_class=PATHOCELL_RESULTS / "conch_LUL_identity/summary/patch_per_class_metrics_from_scores.csv",
        conch_frozen_per_class=PATHOCELL_RESULTS / "conch_frozen/summary/patch_per_class_metrics_from_scores.csv",
        # Baselines (terms1) aggregated per-class metrics computed implicitly via metrics_from_scores
        conch_terms1_per_class=PATHOCELL_RESULTS / "conch_terms1/summary/patch_per_class_metrics_from_scores.csv",
        plip_terms1_per_class=PATHOCELL_RESULTS / "plip_terms1/summary/patch_per_class_metrics_from_scores.csv",
    output:
        plot=PATHOCELL_RESULTS / "comparison" / "patch" / "plots" / "per_class__{metric}__trimodal_vs_quilt_conch.svg",
        csv_table=PATHOCELL_RESULTS / "comparison" / "patch" / "tables" / "per_class_{metric}_trimodal_vs_quilt_conch_terms1_plip_terms1.csv",
    params:
        prediction_level="patch",
        metric="{metric}",
    wildcard_constraints:
        metric="(f1|rocauc|soft_rocauc|precision|accuracy|recall_at_5|mae_prob|mse_prob)",
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/plot_pathocell_baselines_vs_trimodal.py"

rule pathocell_terms1_vs_terms2_table:
    """
    Build AUROC per-class comparison table for CONCH and PLIP baselines: terms1 vs terms2.
    Uses metrics_from_scores per-class CSVs (seed/dataset aggregated) and writes a CSV table.
    """
    input:
        conch_terms1_per_class=PATHOCELL_RESULTS / "conch_terms1/summary/patch_per_class_metrics_from_scores.csv",
        conch_terms2_per_class=PATHOCELL_RESULTS / "conch_terms2/summary/patch_per_class_metrics_from_scores.csv",
        plip_terms1_per_class=PATHOCELL_RESULTS / "plip_terms1/summary/patch_per_class_metrics_from_scores.csv",
        plip_terms2_per_class=PATHOCELL_RESULTS / "plip_terms2/summary/patch_per_class_metrics_from_scores.csv",
    output:
        csv_table=PATHOCELL_RESULTS / "comparison" / "patch" / "tables" / "per_class_rocauc_conch_terms1_vs_terms2_plip_terms1_vs_terms2.csv",
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/plot_pathocell_terms1_vs_terms2_table.py"

rule pathocell_split_baseline_scores:
    """
    Split baseline logits (terms1) into a per-dataset score CSV matching
    pathocell_cell_type_prediction.output.scores schema.
    Uses wildcards for baseline (conch|plip), dataset, prediction_level, and seed.
    """
    input:
        baseline_csv=BASELINES_DIR / "{baseline}_logits_{terms_id}.csv",
    output:
        score=PATHOCELL_RESULTS / "{baseline}_{terms_id}" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        baseline="(conch|plip)",
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
        terms_id="(terms1|terms2)"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/split_baseline_logits.py"

ruleorder: pathocell_split_baseline_scores > pathocell_cell_type_prediction

# NOTE: These rules are great, but not debugged yet :). Animesh ran them manually
# # Baseline runners using local scripts and project data structure
# rule pathocell_conch_baseline:
#     output:
#         logits_terms1=PATHOCELL_DATA / "baselines_animesh_computed" / "conch_logits_terms1.csv",
#         logits_terms2=PATHOCELL_DATA / "baselines_animesh_computed" / "conch_logits_terms2.csv",
#     params:
#         data_dir=PATHOCELL_DATA / "processed",
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=32000,
#         slurm=slurm_gres("medium", num_cpus=4, time="4:00:00")
#     script:
#         "../scripts/run_conch_baseline.py"

# rule pathocell_plip_baseline:
#     output:
#         logits_terms1=PATHOCELL_DATA / "baselines_animesh_computed" / "plip_logits_terms1.csv",
#         logits_terms2=PATHOCELL_DATA / "baselines_animesh_computed" / "plip_logits_terms2.csv",
#     params:
#         data_dir=PATHOCELL_DATA / "processed",
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=32000,
#         slurm=slurm_gres("medium", num_cpus=4, time="4:00:00")
#     script:
#         "../scripts/run_plip_baseline.py"


rule pathocell_violin_deltas:
    """
    Violin plots of per-dataset metric deltas (model_a - model_b), one violin per metric.
    Uses per-dataset CSVs from metrics_from_scores.
    """
    input:
        mpl_style=ancient(PROJECT_DIR / config["plot_style"]),
        a_per_dataset=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_a}/summary/{wildcards.prediction_level}_per_dataset_metrics_from_scores.csv",
        b_per_dataset=lambda wildcards: PATHOCELL_RESULTS / f"{wildcards.model_b}/summary/{wildcards.prediction_level}_per_dataset_metrics_from_scores.csv",
    output:
        plot=PATHOCELL_RESULTS / "comparison" / "{prediction_level}" / "plots" / "violin_deltas__{model_a}_vs_{model_b}.svg",
    params:
        model_a="{model_a}",
        model_b="{model_b}",
        prediction_level="{prediction_level}",
    wildcard_constraints:
        prediction_level="(cell|patch)",
        model_a="[^/]+",
        model_b="[^/]+",
    resources:
        mem_mb=6000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/plot_pathocell_violin_deltas.py"



TRIMODAL_ABLATION_MODELS = {
    "Bimodal bridge": "spatialwhisperer_cellxgene_census__archs4_geo__hest1k",
    "Trimodal": "spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m",
    "Trimodal (curated)": "spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m_curated",
}

rule pathocell_trimodal_comparison_table:
    """
    Build per-class AUROC comparison table for trimodal ablation models on CRC benchmark.
    Produces a CSV and a LaTeX snippet matching tab:pathocell_benchmark format.
    """
    input:
        per_class_csvs=[
            PATHOCELL_RESULTS / f"{model}/summary/patch_per_class_metrics_from_scores.csv"
            for model in TRIMODAL_ABLATION_MODELS.values()
        ],
    output:
        csv_table=PATHOCELL_RESULTS / "comparison" / "patch" / "tables" / "trimodal_ablation_rocauc.csv",
        latex_table=PATHOCELL_RESULTS / "comparison" / "patch" / "tables" / "trimodal_ablation_rocauc.tex",
    params:
        model_labels=list(TRIMODAL_ABLATION_MODELS.keys()),
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/pathocell_trimodal_comparison_table.py"


rule pathocell_all:
    """
    Run complete PathoCellBench evaluation for cell type classification.
    TODO: most of these request __quilt1m (although we are usually interested in the bimodal bridge model)
    """
    input:
        # Per-class delta plots (core comparisons)
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            model_b=["spatialwhisperer_quilt1m"],
            prediction_level=["patch"],
            metric=["auroc", "f1"],
        ),
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k"],
            model_b=["spatialwhisperer_quilt1m"],
            prediction_level=["patch"],
            metric=["auroc", "f1"],
        ),
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["conch_frozen"],
            model_b=["conch_LUL_identity"],
            prediction_level=["patch"],
            metric=["auroc", "f1"],
        ),
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            model_b=["conch_frozen"],
            prediction_level=["patch"],
            metric=["auroc", "f1"],
        ),
        # Requested per-class delta plots (extras including soft_rocauc)
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["conch_LUL_identity"],
            model_b=["conch_frozen"],
            prediction_level=["patch"],
            metric=["f1", "auroc", "soft_rocauc"],
        ),
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k"],
            model_b=["spatialwhisperer_quilt1m"],
            prediction_level=["patch"],
            metric=["auroc", "f1", "soft_rocauc"],
        ),
        expand(
            rules.pathocell_per_class.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k"],
            model_b=["conch_frozen"],
            prediction_level=["patch"],
            metric=["f1", "auroc", "soft_rocauc"],
        ),
        # Per-class F1 scatter
        # expand(
        #     rules.pathocell_performance_overview.output.plot,
        #     model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
        #     model_b=["spatialwhisperer_quilt1m"],
        #     prediction_level=["patch"],
        # ),
        # Baselines vs Trimodal bar plots (single-metric axes)
        expand(
            rules.pathocell_baselines_vs_trimodal.output.plot,
            metric=[
                "f1",
                "rocauc",
                "soft_rocauc",
                "precision",
                "accuracy",
                "recall_at_5",
                "mae_prob",
                "mse_prob",
            ],
        ),
        # Requested violin deltas (per-dataset metric deltas)
        expand(
            rules.pathocell_violin_deltas.output.plot,
            model_a=["conch_frozen"],
            model_b=["conch_LUL_identity"],
            prediction_level=["patch"],
        ),
        expand(
            rules.pathocell_violin_deltas.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            model_b=["conch_frozen"],
            prediction_level=["patch"],
        ),
        expand(
            rules.pathocell_violin_deltas.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            model_b=["spatialwhisperer_quilt1m"],
            prediction_level=["patch"],
        ),
        # Requested KL divergence scatter (trimodal vs PLIP/CONCH terms1)
        expand(
            rules.pathocell_performance_overview.output.plot,
            model_a=["spatialwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            model_b=["plip_terms1", "conch_terms1"],
            prediction_level=["patch"],
            metric=["mean_kl_divergence"],
        ),
    default_target: True
