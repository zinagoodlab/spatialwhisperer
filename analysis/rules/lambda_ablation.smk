# Lambda ablation: approximate loss-weight sensitivity via dataset duplication
#
# Reviewer question: "How sensitive is the model to the choice of λ₁, λ₂, λ₃?"
# We approximate different λ ratios by varying dataset composition while fixing
# the total number of gradient steps (max_steps: 4000 ≈ 1 epoch of the baseline).
#
# Configs are discovered from delta_config/*.yaml in the experiments directory.

from pathlib import Path as _Path

LAMBDA_ABLATION_DIR = PROJECT_DIR / "src/spotwhisperer_eval/experiments/lambda_ablation"
LAMBDA_DELTA_DIR = LAMBDA_ABLATION_DIR / "delta_config"
LAMBDA_CONFIGS = sorted(p.stem for p in LAMBDA_DELTA_DIR.glob("*.yaml"))
LAMBDA_FIXED_STEP_CONFIGS = sorted(
    p.stem for p in LAMBDA_DELTA_DIR.glob("*.yaml")
    if not p.stem.endswith("_full_epoch") and not p.stem.startswith("perpair_")
)
LAMBDA_FULL_EPOCH_CONFIGS = sorted(
    p.stem for p in LAMBDA_DELTA_DIR.glob("*_full_epoch.yaml")
)
LAMBDA_PERPAIR_CONFIGS = sorted(
    p.stem for p in LAMBDA_DELTA_DIR.glob("perpair_*.yaml")
)
LAMBDA_RESULTS = PATHOCELL_RESULTS  # reuse existing pathocell results structure
LAMBDA_BASE_CONFIG = PROJECT_DIR / "src/spotwhisperer_eval/lambda_training_config.yaml"


rule train_spotwhisperer_lambda:
    """Train a SpotWhisperer model with a lambda ablation delta config."""
    input:
        base_config=ancient(LAMBDA_BASE_CONFIG),
        delta_config=ancient(LAMBDA_DELTA_DIR / "{lambda_config}.yaml"),
    output:
        model=protected(PROJECT_DIR / config["paths"]["jointemb_models"] / "spotwhisperer_lambda_{lambda_config}.ckpt"),
    params:
        seed=SEEDS[0],
        project_dir=PROJECT_DIR,
    wildcard_constraints:
        lambda_config="|".join(LAMBDA_CONFIGS),
    conda:
        "cellwhisperer"
    resources:
        mem_mb=250000,
        slurm=slurm_gres("large", num_cpus=12, time="24:00:00", num_gpus=1),
    shell: """
        cd {params.project_dir}
        cellwhisperer fit \
            --config {input.base_config} \
            --config {input.delta_config} \
            --seed_everything {params.seed} \
            --last_model_path {output.model} \
            --omit_validation_functions \
            --wandb lambda_ablation_{wildcards.lambda_config}
    """

ruleorder: train_spotwhisperer_lambda > train_spotwhisperer


rule lambda_ablation_all:
    """Target: train all lambda configs and run CRC PathoCellBench eval."""
    input:
        # CRC per-class metrics for each lambda config
        expand(
            LAMBDA_RESULTS / "spotwhisperer_lambda_{lc}/summary/patch_per_class_metrics_from_scores.csv",
            lc=LAMBDA_FIXED_STEP_CONFIGS,
        ),


rule lambda_ablation_full_epoch_all:
    """Target: train the full-epoch lambda rerun configs and run CRC PathoCellBench eval."""
    input:
        expand(
            LAMBDA_RESULTS / "spotwhisperer_lambda_{lc}/summary/patch_per_class_metrics_from_scores.csv",
            lc=LAMBDA_FULL_EPOCH_CONFIGS,
        ),


rule lambda_ablation_perpair_all:
    """Target: per-pair lambda sweep (direct loss weighting in ClipLoss).

    Uses identical data (archs4_geo,cellxgene_census,hest1k) and step count
    across all configs; only the per-pair lambdas vary, so the result
    isolates loss-weighting sensitivity from data-coverage / step-budget
    effects that confounded the earlier dataset-duplication runs.
    """
    input:
        expand(
            LAMBDA_RESULTS / "spotwhisperer_lambda_{lc}/summary/patch_per_class_metrics_from_scores.csv",
            lc=LAMBDA_PERPAIR_CONFIGS,
        ),
