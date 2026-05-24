# Seed variance analysis for reviewer response (Table 2)
#
# Reuses existing seed-0 model (spotwhisperer_cellxgene_census__archs4_geo__hest1k)
# and trains two additional seeds (1, 42). Runs CRC pathocell eval for each seeded
# model, then aggregates per-class metrics into a mean +/- std table.

VARIANCE_SEEDS = [1, 2]  # seeds to train; seed 0 = existing model
VARIANCE_DATASET_COMBO = "cellxgene_census__archs4_geo__hest1k"
VARIANCE_MODEL_BASE = f"spotwhisperer_{VARIANCE_DATASET_COMBO}"

SEED_TRAINING_CONFIG = srcdir("../seed_training_config.yaml")

ruleorder: train_spotwhisperer_seeded > train_spotwhisperer

rule train_spotwhisperer_seeded:
    """Train SpotWhisperer with an explicit seed encoded in filename."""
    input:
        base_config=ancient(SEED_TRAINING_CONFIG),
        subsamples=_dataset_requirements,
    output:
        model=protected(PROJECT_DIR / config["paths"]["jointemb_models"] / "spotwhisperer_{dataset_combo}_seed{seed}.ckpt"),
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
            --wandb spotwhisperer_seed{wildcards.seed}_{wildcards.dataset_combo}
    """

rule seed_variance_table:
    """Aggregate per-class CRC metrics across seeds (0, 1, 42) into mean +/- std table."""
    input:
        # Seed 0: existing model (no _seed suffix)
        seed0_per_class=PATHOCELL_RESULTS / VARIANCE_MODEL_BASE / "summary/patch_per_class_metrics_from_scores.csv",
        # Seeds 1, 42: seeded models
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

rule seed_analysis_all:
    """Target: train seeded models, run CRC pathocell eval, aggregate variance."""
    input:
        rules.seed_variance_table.output.table,
