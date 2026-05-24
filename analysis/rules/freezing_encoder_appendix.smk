# Freezing/encoder-choice ablation appendix.
#
# Reviewer commitment: SpatialWhisperer_ICML/rebuttal/final.org lines 1051-1053
# ("additional ablations on encoder modules and freezing configurations").
#
# Two axes, single seed, 1 epoch each:
#   freezing axis (geneformer fixed): baseline (LUL), lll, llu, ull
#   encoder axis (LUL fixed):         baseline (geneformer), uce
# baseline is shared between the axes -> 5 configs total + 1 smoke config.
#
# Configs are discovered from delta_config/*.yaml in the experiments directory.

from pathlib import Path as _Path

FE_ABLATION_DIR = PROJECT_DIR / "analysis/experiments/freezing_encoder_appendix"
FE_DELTA_DIR = FE_ABLATION_DIR / "delta_config"
# Production configs evaluated on PathoCellBench. Excludes smoke configs.
FE_CONFIGS = sorted(
    p.stem for p in FE_DELTA_DIR.glob("*.yaml") if not p.stem.endswith("_smoke")
)
FE_SMOKE_CONFIGS = sorted(
    p.stem for p in FE_DELTA_DIR.glob("*_smoke.yaml")
)
FE_RESULTS = PATHOCELL_RESULTS  # reuse existing pathocell results structure
FE_BASE_CONFIG = PROJECT_DIR / "analysis/freezing_encoder_training_config.yaml"


rule train_spotwhisperer_fe:
    """Train a SpotWhisperer model with a freezing/encoder delta config."""
    input:
        base_config=ancient(FE_BASE_CONFIG),
        delta_config=ancient(FE_DELTA_DIR / "{fe_config}.yaml"),
    output:
        model=protected(PROJECT_DIR / config["paths"]["jointemb_models"] / "spotwhisperer_fe_{fe_config}.ckpt"),
    params:
        seed=SEEDS[0],
        project_dir=PROJECT_DIR,
    wildcard_constraints:
        fe_config="|".join(FE_CONFIGS + FE_SMOKE_CONFIGS),
    conda:
        "cellwhisperer"
    resources:
        # Sized so the 3 Geneformer configs (lll, llu, ull) can run in
        # parallel on sh04-02n05 (64 CPUs / 1000 GB / 4 GPUs):
        # 3 × 16 CPU = 48; 3 × 200 GB = 600 GB; 3 × 1 GPU = 3 → fits.
        # UCE needs more RAM for first-pass prep on full datasets; bump to
        # 700 GB for that one config only.
        mem_mb=lambda wildcards: 700000 if wildcards.fe_config == "uce" else 200000,
        slurm=lambda wildcards: slurm_gres("large", num_cpus=16, time="24:00:00", num_gpus=1),
    shell: """
        cd {params.project_dir}
        cellwhisperer fit \
            --config {input.base_config} \
            --config {input.delta_config} \
            --seed_everything {params.seed} \
            --last_model_path {output.model} \
            --omit_validation_functions \
            --wandb fe_appendix_{wildcards.fe_config}
    """

ruleorder: train_spotwhisperer_fe > train_spotwhisperer


rule freezing_encoder_appendix_smoke:
    """Smoke target: train the *_smoke configs only (no eval). Confirms the
    UCE preprocessing pipeline + training loop work end-to-end before the
    full sweep is launched."""
    input:
        expand(
            PROJECT_DIR / config["paths"]["jointemb_models"] / "spotwhisperer_fe_{fc}.ckpt",
            fc=FE_SMOKE_CONFIGS,
        ),


rule freezing_encoder_appendix_all:
    """Full sweep: train + CRC PathoCellBench eval for each production config."""
    input:
        expand(
            FE_RESULTS / "spotwhisperer_fe_{fc}/summary/patch_per_class_metrics_from_scores.csv",
            fc=FE_CONFIGS,
        ),


# Configs we actually trained successfully (ULL failed to fit on H100 80 GB
# even at bs=16; reported as N/A with asterisk in the appendix table).
FE_TRAINED_CONFIGS = [c for c in FE_CONFIGS if c != "ull"]


rule freezing_encoder_appendix_eval_trained:
    """Run CRC PathoCellBench eval on the configs we trained (excluding ull,
    which didn't fit in GPU memory). Mirrors `_all` but skips the missing
    ULL ckpt so the eval phase can launch without that dependency."""
    input:
        expand(
            FE_RESULTS / "spotwhisperer_fe_{fc}/summary/patch_per_class_metrics_from_scores.csv",
            fc=FE_TRAINED_CONFIGS,
        ),


rule freezing_encoder_appendix_train_remaining:
    """Train just the non-baseline production ckpts (lll, llu, ull, uce). No eval.
    Useful when baseline.ckpt already exists and we want training to take GPU
    priority before kicking off the ~545-job pathocell eval fan-out."""
    input:
        expand(
            PROJECT_DIR / config["paths"]["jointemb_models"] / "spotwhisperer_fe_{fc}.ckpt",
            fc=[c for c in FE_CONFIGS if c != "baseline"],
        ),
