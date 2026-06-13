# Two-stage baseline: Image → Expression → Cell Type
#
# Implements the reviewer-requested comparison pipeline:
#   Stage 1: Train UNI2 → gene expression decoder (on HEST-1K)
#   Stage 2: Use existing fine-tuned Geneformer classifier (from CellWhisperer paper)
#   Eval:    Run on PathoCellBench CRC at patch level
#
# This baseline tests whether directly predicting transcriptomics from H&E
# and then classifying cell types can match our trimodal contrastive approach.

from pathlib import Path as _Path

TWO_STAGE_DIR = PROJECT_DIR / "results" / "two_stage_baseline"
TWO_STAGE_RESOURCES = PROJECT_DIR / "resources" / "two_stage_baseline"


rule create_hest_geneformer_genelist:
    """
    Create gene list CSV: intersection of HEST-1K genes and Geneformer vocabulary.
    Defines the target gene space for the image→expression decoder.
    """
    input:
        hest_dir=PROJECT_DIR / "results" / "hest1k" / "h5ads",
    output:
        gene_list=TWO_STAGE_RESOURCES / "hest_geneformer_genes.csv",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=32000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/create_hest_geneformer_genelist.py"


rule train_two_stage_decoder:
    """
    Train image→expression decoder: frozen UNI2 (1536-dim) → MILinearBlock → gene expression.
    Uses HEST-1K data (same paired image-expression data as contrastive training).

    Uses a dedicated training script that monkey-patches the MLP processor
    to use the HEST-Geneformer gene list instead of the default cosmx6k.
    """
    input:
        gene_list=rules.create_hest_geneformer_genelist.output.gene_list,
        uni2_weights=PROJECT_DIR / "resources" / "uni2",
    output:
        decoder_ckpt=protected(TWO_STAGE_DIR / "decoder" / "decoder.ckpt"),
    params:
        max_epochs=4,
        learning_rate=1e-3,
        batch_size=256,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=250000,
        slurm=slurm_gres("large", num_cpus=8, time="16:00:00")
    script:
        "../scripts/train_two_stage_decoder.py"


rule retrain_geneformer_classifier:
    """
    Re-train the frozen Geneformer classifier (Geneformer backbone + linear head)
    on CellXGene Census cell_type labels. Produces the checkpoint needed by Stage 2.
    """
    input:
        model_weights=lambda wildcards: PROJECT_DIR / config["model_name_path_map"][wildcards.model],
        training_data=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="cellxgene_census"),
    output:
        model_weights=PROJECT_DIR / "results" / "finetuning_eval" / "{model}" / "finetuned_{training_options}.pt",
    wildcard_constraints:
        model="geneformer",
        training_options="frozen",
    params:
        label_col="cell_type",
        num_epochs=8,
        batch_size=16,
        learning_rate=1e-4,
        freeze_fm=True,
    resources:
        mem_mb=350000,
        slurm=slurm_gres("large")
    conda:
        "cellwhisperer"
    script:
        "../scripts/retrain_geneformer_classifier.py"


rule download_transfered_labels:
    """Download pre-computed GPT-4o label transfer from CellWhisperer data release."""
    output:
        transfered_labels=PROJECT_DIR / "results" / "finetuning_eval" / "tabula_sapiens" / "transfered_labels.csv",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=1000,
        slurm="cpus-per-task=1"
    shell: """
        mkdir -p $(dirname {output.transfered_labels})
        curl -sL "https://medical-epigenomics.org/papers/schaefer2025cellwhisperer/data/datasets/tabula_sapiens/transfered_labels.csv" \
            -o {output.transfered_labels}
    """


rule create_pathocell_label_mapping:
    """
    Create label mapping from CellXGene Census cell types to PathoCellBench CRC cell types.
    Uses keyword-based heuristic mapping (analogous to GPT-4o transfer in CellWhisperer paper).

    The transfered_labels input comes from the transfer_labels_download rule
    (src/figures/rules/zero_shot_finetuning.smk), which downloads from the data release.
    """
    input:
        transfered_labels=PROJECT_DIR / "results" / "finetuning_eval" / "tabula_sapiens" / "transfered_labels.csv",
    output:
        label_mapping=TWO_STAGE_RESOURCES / "pathocell_crc_label_mapping.csv",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/create_pathocell_label_mapping.py"


rule two_stage_pathocell_predict:
    """
    Run the two-stage baseline on a single PathoCellBench CRC dataset.
    Produces a score CSV matching the format of other baselines.
    """
    input:
        adata=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.tiff",
        decoder_ckpt=rules.train_two_stage_decoder.output.decoder_ckpt,
        classifier_weights=PROJECT_DIR / "results" / "finetuning_eval" / "geneformer" / "finetuned_frozen.pt",
        gene_list=rules.create_hest_geneformer_genelist.output.gene_list,
        label_mapping=rules.create_pathocell_label_mapping.output.label_mapping,
        uni2_weights=PROJECT_DIR / "resources" / "uni2",
    output:
        scores=PATHOCELL_RESULTS / "two_stage_baseline" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    script:
        "../scripts/two_stage_predict.py"


ruleorder: pathocell_split_baseline_scores > two_stage_pathocell_predict > pathocell_cell_type_prediction


# ── Two-stage on Lizard / PanNuke ────────────────────────────────────────────
# Mirror two_stage_pathocell_predict for the secondary cell-type benchmarks.
# Reuses the trained UNI2→expression decoder and the fine-tuned Geneformer
# classifier; only the per-benchmark label mapping CSV differs.

rule create_lizard_label_mapping:
    """
    Build CellXGene Census → Lizard 6-class label mapping CSV.
    """
    input:
        transfered_labels=PROJECT_DIR / "results" / "finetuning_eval" / "tabula_sapiens" / "transfered_labels.csv",
    output:
        label_mapping=TWO_STAGE_RESOURCES / "lizard_label_mapping.csv",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/create_lizard_label_mapping.py"


rule create_pannuke_label_mapping:
    """
    Build CellXGene Census → PanNuke 5-class label mapping CSV.
    """
    input:
        transfered_labels=PROJECT_DIR / "results" / "finetuning_eval" / "tabula_sapiens" / "transfered_labels.csv",
    output:
        label_mapping=TWO_STAGE_RESOURCES / "pannuke_label_mapping.csv",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1"
    script:
        "../scripts/create_pannuke_label_mapping.py"


rule two_stage_lizard_predict:
    """
    Run the two-stage baseline on a single Lizard dataset.
    Output schema matches lizard_cell_type_prediction.output.scores.
    """
    input:
        adata=PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}.tiff",
        decoder_ckpt=rules.train_two_stage_decoder.output.decoder_ckpt,
        classifier_weights=PROJECT_DIR / "results" / "finetuning_eval" / "geneformer" / "finetuned_frozen.pt",
        gene_list=rules.create_hest_geneformer_genelist.output.gene_list,
        label_mapping=rules.create_lizard_label_mapping.output.label_mapping,
        uni2_weights=PROJECT_DIR / "resources" / "uni2",
    output:
        scores=PATHOCELL_RESULTS / "two_stage_baseline" / "lizard" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    script:
        "../scripts/two_stage_predict.py"


rule two_stage_pannuke_predict:
    """
    Run the two-stage baseline on a single PanNuke dataset.
    Output schema matches pannuke_cell_type_prediction.output.scores.
    """
    input:
        adata=PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}.tiff",
        decoder_ckpt=rules.train_two_stage_decoder.output.decoder_ckpt,
        classifier_weights=PROJECT_DIR / "results" / "finetuning_eval" / "geneformer" / "finetuned_frozen.pt",
        gene_list=rules.create_hest_geneformer_genelist.output.gene_list,
        label_mapping=rules.create_pannuke_label_mapping.output.label_mapping,
        uni2_weights=PROJECT_DIR / "resources" / "uni2",
    output:
        scores=PATHOCELL_RESULTS / "two_stage_baseline" / "pannuke" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00")
    script:
        "../scripts/two_stage_predict.py"


ruleorder: two_stage_lizard_predict > lizard_cell_type_prediction
ruleorder: two_stage_pannuke_predict > pannuke_cell_type_prediction
