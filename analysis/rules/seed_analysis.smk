# Seed variance analysis for reviewer response (Table 2)
#
# Trains additional seeded models alongside the seed-0 model produced by
# `train_spatialwhisperer`, runs CRC/Lizard/PanNuke eval for each, and
# aggregates the resulting per-class score CSVs into the per-seed tables
# reported in the manuscript Table 2 footprint.

VARIANCE_SEEDS = [1, 2]  # seeds to train; seed 0 = existing model
VARIANCE_DATASET_COMBO = "cellxgene_census__archs4_geo__hest1k"
VARIANCE_MODEL_BASE = f"spatialwhisperer_{VARIANCE_DATASET_COMBO}"

# seed-0 model = bare VARIANCE_MODEL_BASE (no _seed suffix; canonical training path).
# Other seeds = VARIANCE_MODEL_BASE + _seed{N}.
VARIANCE_MODELS = [VARIANCE_MODEL_BASE] + [
    f"{VARIANCE_MODEL_BASE}_seed{s}" for s in VARIANCE_SEEDS
]
VARIANCE_SEED_LABELS = ["seed0"] + [f"seed{s}" for s in VARIANCE_SEEDS]

TERMS_IDS = ["terms1", "terms2"]
TABLE2_BASELINES = ["conch", "plip"]  # MUSK dropped (not part of the published paper)

SEED_TRAINING_CONFIG = srcdir("../seed_training_config.yaml")

ruleorder: train_spatialwhisperer_seeded > train_spatialwhisperer

rule train_spatialwhisperer_seeded:
    """Train SpotWhisperer with an explicit seed encoded in filename."""
    input:
        base_config=ancient(SEED_TRAINING_CONFIG),
        subsamples=_dataset_requirements,
    output:
        model=protected(PROJECT_DIR / config["paths"]["jointemb_models"] / "spatialwhisperer_{dataset_combo}_seed{seed}.ckpt"),
    params:
        dataset_names=lambda wildcards: wildcards.dataset_combo.replace("__", ","),
        test_run_config="--trainer.limit_train_batches 500 --trainer.max_epochs 2" if config.get("fast", False) else "",
        project_dir=PROJECT_DIR,
    wildcard_constraints:
        seed="\\d+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=lambda wildcards: 250000 if "archs4_geo" in wildcards.dataset_combo else 150000,
        slurm=slurm_gres("large", num_cpus=12, time="70:00:00", num_gpus=1),
    shell: """
        cd {params.project_dir}
        cellwhisperer fit \
            --config {input.base_config} \
            --data.dataset_names {params.dataset_names} \
            {params.test_run_config} \
            --seed_everything {wildcards.seed} \
            --last_model_path {output.model} \
            --omit_validation_functions \
            --wandb spatialwhisperer_seed{wildcards.seed}_{wildcards.dataset_combo}
    """

rule seed_variance_table:
    """Aggregate per-class CRC metrics across seeds (0, 1, 2) into mean +/- std table."""
    input:
        # Seed 0: existing model (no _seed suffix)
        seed0_per_class=PATHOCELL_RESULTS / VARIANCE_MODEL_BASE / "summary/patch_per_class_metrics_from_scores.csv",
        # Seeds 1, 2: seeded models
        seeded_per_class=expand(
            PATHOCELL_RESULTS / "{model}_seed{seed}/summary/patch_per_class_metrics_from_scores.csv",
            model=VARIANCE_MODEL_BASE,
            seed=VARIANCE_SEEDS,
        ),
    output:
        table=PATHOCELL_RESULTS / "seed_variance/seed_variance_table.csv",
    params:
        seed0=0,
        extra_seeds=VARIANCE_SEEDS,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1",
    script:
        "../scripts/compute_seed_variance_table.py"


# ----- Multi-seed Table 2-style reporting --------------------------------------
# Helpers gather per-dataset score CSVs and h5ad ground truth across all
# variance models (seed 0 plus VARIANCE_SEEDS).

def _all_crc_scores():
    return expand(
        PATHOCELL_RESULTS / "{model}" / "{dataset}_patch_scores_seed0.csv",
        model=VARIANCE_MODELS, dataset=DATASETS,
    )

def _all_lizard_scores():
    return expand(
        PATHOCELL_RESULTS / "{model}" / "lizard/{dataset}_patch_scores_seed0.csv",
        model=VARIANCE_MODELS, dataset=LIZARD_DATASETS,
    )

def _all_pannuke_scores():
    return expand(
        PATHOCELL_RESULTS / "{model}" / "pannuke/{dataset}_patch_scores_seed0.csv",
        model=VARIANCE_MODELS, dataset=PANNUKE_DATASETS,
    )

def _crc_gt_h5ads():
    return expand(PATHOCELL_DATA / "processed/{dataset}_patch.h5ad", dataset=DATASETS)

def _lizard_gt_h5ads():
    return expand(PATHOCELL_DATA / "processed/lizard/{dataset}_patch.h5ad", dataset=LIZARD_DATASETS)

def _pannuke_gt_h5ads():
    return expand(PATHOCELL_DATA / "processed/pannuke/{dataset}_patch.h5ad", dataset=PANNUKE_DATASETS)


rule reduced_class_table2_style:
    """Our model, all seeds: classes→datasets→mean AUROC reproducing Table 2 exactly."""
    input:
        crc_scores=_all_crc_scores(),
        lizard_scores=_all_lizard_scores(),
        pannuke_scores=_all_pannuke_scores(),
        crc_gt=_crc_gt_h5ads(),
        lizard_gt=_lizard_gt_h5ads(),
        pannuke_gt=_pannuke_gt_h5ads(),
    output:
        macro=PATHOCELL_RESULTS / "seed_variance/reduced_class_table2_style.csv",
        per_class=PATHOCELL_RESULTS / "seed_variance/reduced_class_table2_style_per_class.csv",
    params:
        models=VARIANCE_MODELS,
        seed_labels=VARIANCE_SEED_LABELS,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1",
    script:
        "../scripts/compute_reduced_class_table2_style.py"


rule reduced_class_table2_style_seed0:
    """Seed-0-only variant for the public reproducer (uses only the published checkpoint)."""
    input:
        crc_scores=expand(
            PATHOCELL_RESULTS / "{model}" / "{dataset}_patch_scores_seed0.csv",
            model=[VARIANCE_MODEL_BASE], dataset=DATASETS,
        ),
        lizard_scores=expand(
            PATHOCELL_RESULTS / "{model}" / "lizard/{dataset}_patch_scores_seed0.csv",
            model=[VARIANCE_MODEL_BASE], dataset=LIZARD_DATASETS,
        ),
        pannuke_scores=expand(
            PATHOCELL_RESULTS / "{model}" / "pannuke/{dataset}_patch_scores_seed0.csv",
            model=[VARIANCE_MODEL_BASE], dataset=PANNUKE_DATASETS,
        ),
        crc_gt=_crc_gt_h5ads(),
        lizard_gt=_lizard_gt_h5ads(),
        pannuke_gt=_pannuke_gt_h5ads(),
    output:
        macro=PATHOCELL_RESULTS / "seed_variance/reduced_class_table2_style_seed0.csv",
        per_class=PATHOCELL_RESULTS / "seed_variance/reduced_class_table2_style_seed0_per_class.csv",
    params:
        models=[VARIANCE_MODEL_BASE],
        seed_labels=["seed0"],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1",
    script:
        "../scripts/compute_reduced_class_table2_style.py"


rule reduced_class_seed_variance:
    """Same scores as table2_style but pooled globally across samples (different aggregation)."""
    input:
        crc_scores=_all_crc_scores(),
        lizard_scores=_all_lizard_scores(),
        pannuke_scores=_all_pannuke_scores(),
        crc_gt=_crc_gt_h5ads(),
        lizard_gt=_lizard_gt_h5ads(),
        pannuke_gt=_pannuke_gt_h5ads(),
    output:
        table=PATHOCELL_RESULTS / "seed_variance/reduced_class_seed_variance.csv",
    params:
        models=VARIANCE_MODELS,
        seed_labels=VARIANCE_SEED_LABELS,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1",
    script:
        "../scripts/compute_reduced_class_seed_variance.py"


rule reduced_class_per_dataset_averaged:
    """Per-dataset macro-averaged variant of the Table 2 aggregation."""
    input:
        crc_scores=_all_crc_scores(),
        lizard_scores=_all_lizard_scores(),
        pannuke_scores=_all_pannuke_scores(),
        crc_gt=_crc_gt_h5ads(),
        lizard_gt=_lizard_gt_h5ads(),
        pannuke_gt=_pannuke_gt_h5ads(),
    output:
        table=PATHOCELL_RESULTS / "seed_variance/reduced_class_per_dataset_averaged.csv",
    params:
        models=VARIANCE_MODELS,
        seed_labels=VARIANCE_SEED_LABELS,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1",
    script:
        "../scripts/compute_reduced_class_per_dataset_averaged.py"


rule baselines_table2_style:
    """PLIP/CONCH AUROC with Table 2 methodology, parameterised by terms set."""
    input:
        crc_csvs=expand(
            BASELINES_DIR / "{baseline}_logits_{{terms_id}}.csv",
            baseline=TABLE2_BASELINES,
        ),
        lizard_csvs=expand(
            BASELINES_DIR / "lizard/{baseline}_logits_lizard_{{terms_id}}.csv",
            baseline=TABLE2_BASELINES,
        ),
        pannuke_csvs=expand(
            BASELINES_DIR / "pannuke/{baseline}_logits_pannuke_{{terms_id}}.csv",
            baseline=TABLE2_BASELINES,
        ),
        crc_gt=_crc_gt_h5ads(),
        lizard_gt=_lizard_gt_h5ads(),
        pannuke_gt=_pannuke_gt_h5ads(),
    output:
        macro=PATHOCELL_RESULTS / "seed_variance/baselines_table2_style_{terms_id}.csv",
        per_class=PATHOCELL_RESULTS / "seed_variance/baselines_table2_style_per_class_{terms_id}.csv",
    params:
        baselines=TABLE2_BASELINES,
    wildcard_constraints:
        terms_id="(terms1|terms2)",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=1",
    script:
        "../scripts/compute_baselines_table2_style.py"


rule seed_analysis_all:
    """Target: train seeded models, run CRC/Lizard/PanNuke eval, produce every Table 2 artefact."""
    input:
        rules.seed_variance_table.output.table,
        rules.reduced_class_table2_style.output.macro,
        rules.reduced_class_table2_style.output.per_class,
        rules.reduced_class_seed_variance.output.table,
        rules.reduced_class_per_dataset_averaged.output.table,
        expand(rules.baselines_table2_style.output.macro, terms_id=TERMS_IDS),
        expand(rules.baselines_table2_style.output.per_class, terms_id=TERMS_IDS),
