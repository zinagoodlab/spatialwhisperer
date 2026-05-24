# Native Kriegsmann et al. (2022) skin H&E benchmark.
#
# Re-port of the MUSK skin-classification path into our own eval harness.
# Mirrors the pattern of score_pannuke_tissue_type / compute_tissue_type_metrics
# in disease_stratification.smk: a per-patch logits CSV produced by a GPU rule,
# consumed by a CPU metrics rule.
#
# Critical Snakemake invalidation note: scripts/skin_labels.py (the 16 raw->
# clinical class mapping + prompt template) is declared as an input of
# `score_skin`, so any edit to the label list re-triggers scoring. The MUSK
# rule did not have this guarantee, which is the root cause of the cache-
# trust problem prompting this re-port.

# MUSK CLI translates the "skin" dataset name to <root>/skincancer/ internally
# (clip_benchmark/cli.py); the SkinDataset reads `data/tiles-v2.csv` relative
# to that. Mirror that path here so we hit the same files.
SKIN_DATASET_ROOT = PROJECT_DIR / config["paths"]["musk"]["datasets"] / "skincancer"
SKIN_RESULTS = PROJECT_DIR / "results/skin_benchmark"
SKIN_SCRIPTS = PROJECT_DIR / "analysis/scripts"

# Each labels file is a declared input → edit auto-invalidates scores.
# clinical: Kriegsmann et al. 2022 strings (the relabel we want to ship).
# snakecase: original MUSK path-component strings (pre-relabel baseline; used
# to check whether the prior MUSK-harness 0.922 was actually using clinical
# labels or whether a stale snake_case cache leaked through).
SKIN_LABEL_FILES = {
    "clinical": SKIN_SCRIPTS / "skin_labels.py",
    "snakecase": SKIN_SCRIPTS / "skin_labels_snakecase.py",
}
LABELS_OPTIONS = list(SKIN_LABEL_FILES)
PREPROCESS_OPTIONS = ["resize_crop", "crop_only"]


rule score_skin:
    """Zero-shot patch scoring against the 16 skin labels."""
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        csv=SKIN_DATASET_ROOT / "data/tiles-v2.csv",
        labels_module=lambda wildcards: SKIN_LABEL_FILES[wildcards.labels],
    output:
        scores=SKIN_RESULTS / "{model}" / "{labels}" / "{preprocess}" / "skin_scores.csv",
    params:
        dataset_root=str(SKIN_DATASET_ROOT),
        batch_size=128,
        preprocess=lambda wildcards: wildcards.preprocess,
    wildcard_constraints:
        labels="|".join(LABELS_OPTIONS),
        preprocess="|".join(PREPROCESS_OPTIONS),
    conda:
        "cellwhisperer"
    resources:
        mem_mb=40000,
        slurm=slurm_gres("medium", num_cpus=4, time="2:00:00"),
    log:
        "logs/score_skin_{model}_{labels}_{preprocess}.log"
    script:
        "../scripts/score_kriegsmann_skin.py"


rule compute_skin_metrics:
    """Per-class one-vs-rest AUROC, macro mean, argmax confusion matrix."""
    input:
        scores=rules.score_skin.output.scores,
    output:
        per_class=SKIN_RESULTS / "{model}" / "{labels}" / "{preprocess}" / "skin_per_class.csv",
        summary=SKIN_RESULTS / "{model}" / "{labels}" / "{preprocess}" / "skin_summary.csv",
        confusion=SKIN_RESULTS / "{model}" / "{labels}" / "{preprocess}" / "skin_confusion.csv",
    wildcard_constraints:
        labels="|".join(LABELS_OPTIONS),
        preprocess="|".join(PREPROCESS_OPTIONS),
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1"
    log:
        "logs/compute_skin_metrics_{model}_{labels}_{preprocess}.log"
    script:
        "../scripts/compute_skin_metrics.py"
