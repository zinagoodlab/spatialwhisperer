# OmiCLIP (Loki) baseline: zero-shot tissue annotation via marker genes
#
# Uses the pretrained OmiCLIP foundation model (CoCa ViT-L-14) to annotate
# PathoCellBench patches by computing cosine similarity between image
# embeddings and marker-gene "sentence" embeddings.
#
# This addresses the reviewer comment about comparing with methods that
# predict gene expression from H&E (GHIST, Loki, sCellST).

OMICLIP_CHECKPOINT = PROJECT_DIR / "resources" / "omiclip" / "checkpoint.pt"
OMICLIP_LOGITS = PATHOCELL_RESULTS / "omiclip" / "omiclip_logits.csv"


rule omiclip_download_checkpoint:
    """
    Download pretrained OmiCLIP checkpoint from HuggingFace.
    """
    output:
        checkpoint=OMICLIP_CHECKPOINT,
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=2 partition=cmackall"
    shell: """
        mkdir -p $(dirname {output.checkpoint})
        huggingface-cli download \
            WangGuangyuLab/Loki \
            checkpoint.pt \
            --local-dir $(dirname {output.checkpoint}) \
            --local-dir-use-symlinks False
    """


rule omiclip_pathocell_score:
    """
    Run OmiCLIP zero-shot marker-gene annotation on all PathoCellBench patches.
    Produces a single CSV with cosine-similarity logits matching the CONCH/PLIP
    baseline schema (source_image, spot_id, <class columns>).
    """
    input:
        checkpoint=OMICLIP_CHECKPOINT,
        adatas=expand(
            PATHOCELL_DATA / "processed/{dataset}_patch.h5ad",
            dataset=DATASETS,
        ),
    output:
        logits_csv=OMICLIP_LOGITS,
    params:
        data_dir=PATHOCELL_DATA / "processed",
        prediction_level="patch",
        batch_size=64,
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=4, time="4:00:00")
    script:
        "../scripts/run_omiclip_baseline.py"


rule omiclip_split_scores:
    """
    Split the single OmiCLIP logits CSV into per-dataset score CSVs,
    reusing the same schema as the CONCH/PLIP baselines.
    """
    input:
        baseline_csv=OMICLIP_LOGITS,
    output:
        score=PATHOCELL_RESULTS / "omiclip" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall"
    script:
        "../scripts/split_baseline_logits.py"


ruleorder: omiclip_split_scores > pathocell_cell_type_prediction


# ── Pseudobulk variant: expanded gene sentences ──────────────────────────────

OMICLIP_PSEUDOBULK_GENES = PROJECT_DIR / "analysis" / "scripts" / "generate_pseudobulk_genes.py"
OMICLIP_PB_LOGITS = PATHOCELL_RESULTS / "omiclip_pseudobulk" / "omiclip_logits.csv"


rule omiclip_pseudobulk_pathocell_score:
    """
    Run OmiCLIP with pseudobulk-style gene sentences (~25-30 genes per class,
    ranked by expression) instead of short marker lists.
    """
    input:
        checkpoint=OMICLIP_CHECKPOINT,
        gene_script=OMICLIP_PSEUDOBULK_GENES,
        score_script=PROJECT_DIR / "analysis" / "scripts" / "run_omiclip_baseline.py",
        adatas=expand(
            PATHOCELL_DATA / "processed/{dataset}_patch.h5ad",
            dataset=DATASETS,
        ),
    output:
        logits_csv=OMICLIP_PB_LOGITS,
        gene_json=PATHOCELL_RESULTS / "omiclip_pseudobulk" / "gene_sentences.json",
    params:
        data_dir=PATHOCELL_DATA / "processed",
        prediction_level="patch",
        batch_size=64,
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=4, time="4:00:00")
    shell: """
        python {input.gene_script} {output.gene_json}
        python {input.score_script} \
            --checkpoint {input.checkpoint} \
            --data_dir {params.data_dir} \
            --output {output.logits_csv} \
            --prediction_level {params.prediction_level} \
            --batch_size {params.batch_size} \
            --gene_sentences_json {output.gene_json}
    """


rule omiclip_pseudobulk_split_scores:
    """
    Split pseudobulk OmiCLIP logits into per-dataset score CSVs.
    """
    input:
        baseline_csv=OMICLIP_PB_LOGITS,
    output:
        score=PATHOCELL_RESULTS / "omiclip_pseudobulk" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall"
    script:
        "../scripts/split_baseline_logits.py"


ruleorder: omiclip_pseudobulk_split_scores > omiclip_split_scores > pathocell_cell_type_prediction


# ── OmiCLIP on Lizard / PanNuke ──────────────────────────────────────────────
# Two-stage published baselines applied to the secondary cell-type benchmarks.
# Per benchmark we run two variants: short marker sentences (~8-12 genes, mirrors
# `omiclip_pathocell_score`) and pseudobulk-style extended sentences (~25-30
# genes, mirrors `omiclip_pseudobulk_pathocell_score`). Marker sentences are
# generated from `scripts/generate_marker_genes.py`.

OMICLIP_MARKER_GENES_DIR = PATHOCELL_RESULTS / "omiclip_marker_genes"


rule omiclip_generate_marker_genes:
    """
    Emit a JSON of marker-gene sentences for a (benchmark, variant) pair.
    benchmark in {lizard, pannuke}; variant in {short, pseudobulk}.
    """
    output:
        gene_json=OMICLIP_MARKER_GENES_DIR / "{benchmark}_{variant}.json",
    wildcard_constraints:
        benchmark="(lizard|pannuke)",
        variant="(short|pseudobulk)",
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall"
    shell:
        "python {workflow.basedir}/scripts/generate_marker_genes.py "
        "{wildcards.benchmark} {wildcards.variant} {output.gene_json}"


def _omiclip_secondary_score_inputs(wildcards):
    """Resolve the per-benchmark adata fan-out for the bulk-score rule."""
    if wildcards.benchmark == "lizard":
        datasets = LIZARD_DATASETS
    else:
        datasets = PANNUKE_DATASETS
    return expand(
        PATHOCELL_DATA / "processed/{bench}/{dataset}_patch.h5ad",
        bench=wildcards.benchmark,
        dataset=datasets,
    )


def _omiclip_secondary_gene_json(wildcards):
    """Map omiclip_model directory name → marker-gene variant JSON path."""
    variant = "pseudobulk" if wildcards.omiclip_model == "omiclip_pseudobulk" else "short"
    return OMICLIP_MARKER_GENES_DIR / f"{wildcards.benchmark}_{variant}.json"


rule omiclip_secondary_score:
    """
    Run OmiCLIP zero-shot annotation on all Lizard or PanNuke patches,
    using the requested marker-gene variant. The output schema matches
    omiclip_pathocell_score (single CSV with all datasets).
    """
    input:
        checkpoint=OMICLIP_CHECKPOINT,
        adatas=_omiclip_secondary_score_inputs,
        gene_json=_omiclip_secondary_gene_json,
    output:
        logits_csv=PATHOCELL_RESULTS / "{omiclip_model}" / "{benchmark}" / "omiclip_logits.csv",
    params:
        data_dir=lambda wildcards: PATHOCELL_DATA / "processed" / wildcards.benchmark,
        prediction_level="patch",
        batch_size=64,
        gene_sentences_json=lambda wildcards, input: str(input.gene_json),
    wildcard_constraints:
        omiclip_model="(omiclip|omiclip_pseudobulk)",
        benchmark="(lizard|pannuke)",
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=4, time="4:00:00")
    script:
        "../scripts/run_omiclip_baseline.py"


rule omiclip_secondary_split_scores:
    """
    Split the per-benchmark OmiCLIP logits CSV into per-dataset score CSVs,
    matching the schema expected by lizard/pannuke_metrics_from_scores.
    """
    input:
        baseline_csv=PATHOCELL_RESULTS / "{omiclip_model}" / "{benchmark}" / "omiclip_logits.csv",
    output:
        score=PATHOCELL_RESULTS / "{omiclip_model}" / "{benchmark}" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        omiclip_model="(omiclip|omiclip_pseudobulk)",
        benchmark="(lizard|pannuke)",
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall"
    script:
        "../scripts/split_baseline_logits.py"


# Resolve the variant wildcard for the bulk score rule when called via
# omiclip_secondary_split_scores. Snakemake will use input function below
# to provide gene_json based on omiclip_model.

ruleorder: omiclip_secondary_split_scores > lizard_cell_type_prediction
ruleorder: omiclip_secondary_split_scores > pannuke_cell_type_prediction
