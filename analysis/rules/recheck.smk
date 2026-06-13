# Recheck (control) variants of the prediction rules.
#
# Outputs are redirected under <model>_recheck/<...> directories so the
# existing scores stay untouched. After a recheck run, comparing the
# recheck CSVs to the originals gives a determinism / pipeline-consistency
# check. Spatialwhisperer models (which use the {model} wildcard) are handled
# without code changes via _recheck.ckpt symlinks; this file adds clones
# only for the rules whose output paths are otherwise hard-coded
# (two-stage and OmiCLIP CRC variants).

# ── Two-stage CRC / Lizard / PanNuke ─────────────────────────────────────────

rule two_stage_pathocell_predict_recheck:
    input:
        adata=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/{dataset}_{prediction_level}.tiff",
        decoder_ckpt=rules.train_two_stage_decoder.output.decoder_ckpt,
        classifier_weights=PROJECT_DIR / "results" / "finetuning_eval" / "geneformer" / "finetuned_frozen.pt",
        gene_list=rules.create_hest_geneformer_genelist.output.gene_list,
        label_mapping=rules.create_pathocell_label_mapping.output.label_mapping,
        uni2_weights=PROJECT_DIR / "resources" / "uni2",
    output:
        scores=PATHOCELL_RESULTS / "two_stage_baseline_recheck" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00"),
    script:
        "../scripts/two_stage_predict.py"


rule two_stage_lizard_predict_recheck:
    input:
        adata=PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/lizard/{dataset}_{prediction_level}.tiff",
        decoder_ckpt=rules.train_two_stage_decoder.output.decoder_ckpt,
        classifier_weights=PROJECT_DIR / "results" / "finetuning_eval" / "geneformer" / "finetuned_frozen.pt",
        gene_list=rules.create_hest_geneformer_genelist.output.gene_list,
        label_mapping=rules.create_lizard_label_mapping.output.label_mapping,
        uni2_weights=PROJECT_DIR / "resources" / "uni2",
    output:
        scores=PATHOCELL_RESULTS / "two_stage_baseline_recheck" / "lizard" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00"),
    script:
        "../scripts/two_stage_predict.py"


rule two_stage_pannuke_predict_recheck:
    input:
        adata=PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}.h5ad",
        image=PATHOCELL_DATA / "processed/pannuke/{dataset}_{prediction_level}.tiff",
        decoder_ckpt=rules.train_two_stage_decoder.output.decoder_ckpt,
        classifier_weights=PROJECT_DIR / "results" / "finetuning_eval" / "geneformer" / "finetuned_frozen.pt",
        gene_list=rules.create_hest_geneformer_genelist.output.gene_list,
        label_mapping=rules.create_pannuke_label_mapping.output.label_mapping,
        uni2_weights=PROJECT_DIR / "resources" / "uni2",
    output:
        scores=PATHOCELL_RESULTS / "two_stage_baseline_recheck" / "pannuke" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=slurm_gres("medium", num_cpus=8, time="2:00:00"),
    script:
        "../scripts/two_stage_predict.py"


# ── OmiCLIP CRC (short and pseudobulk variants) ──────────────────────────────

rule omiclip_pathocell_score_recheck:
    """Mirror of omiclip_pathocell_score; outputs into omiclip_recheck/."""
    input:
        checkpoint=OMICLIP_CHECKPOINT,
        adatas=expand(
            PATHOCELL_DATA / "processed/{dataset}_patch.h5ad",
            dataset=DATASETS,
        ),
    output:
        logits_csv=PATHOCELL_RESULTS / "omiclip_recheck" / "omiclip_logits.csv",
    params:
        data_dir=PATHOCELL_DATA / "processed",
        prediction_level="patch",
        batch_size=64,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=4, time="4:00:00"),
    script:
        "../scripts/run_omiclip_baseline.py"


rule omiclip_pathocell_split_scores_recheck:
    input:
        baseline_csv=rules.omiclip_pathocell_score_recheck.output.logits_csv,
    output:
        score=PATHOCELL_RESULTS / "omiclip_recheck" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall",
    script:
        "../scripts/split_baseline_logits.py"


rule omiclip_pseudobulk_pathocell_score_recheck:
    """Mirror of omiclip_pseudobulk_pathocell_score; outputs into omiclip_pseudobulk_recheck/."""
    input:
        checkpoint=OMICLIP_CHECKPOINT,
        gene_script=OMICLIP_PSEUDOBULK_GENES,
        score_script=PROJECT_DIR / "src" / "spatialwhisperer_eval" / "scripts" / "run_omiclip_baseline.py",
        adatas=expand(
            PATHOCELL_DATA / "processed/{dataset}_patch.h5ad",
            dataset=DATASETS,
        ),
    output:
        logits_csv=PATHOCELL_RESULTS / "omiclip_pseudobulk_recheck" / "omiclip_logits.csv",
        gene_json=PATHOCELL_RESULTS / "omiclip_pseudobulk_recheck" / "gene_sentences.json",
    params:
        data_dir=PATHOCELL_DATA / "processed",
        prediction_level="patch",
        batch_size=64,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=4, time="4:00:00"),
    shell:
        "python {input.gene_script} {output.gene_json} && "
        "python {input.score_script} "
        "--checkpoint {input.checkpoint} --data_dir {params.data_dir} "
        "--output {output.logits_csv} --prediction_level {params.prediction_level} "
        "--batch_size {params.batch_size} --gene_sentences_json {output.gene_json}"


rule omiclip_pseudobulk_pathocell_split_scores_recheck:
    input:
        baseline_csv=rules.omiclip_pseudobulk_pathocell_score_recheck.output.logits_csv,
    output:
        score=PATHOCELL_RESULTS / "omiclip_pseudobulk_recheck" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall",
    script:
        "../scripts/split_baseline_logits.py"


# ── OmiCLIP Lizard / PanNuke (recheck variants of the secondary rules) ───────

rule omiclip_secondary_score_recheck:
    input:
        checkpoint=OMICLIP_CHECKPOINT,
        adatas=lambda wildcards: expand(
            PATHOCELL_DATA / "processed/{bench}/{dataset}_patch.h5ad",
            bench=wildcards.benchmark,
            dataset=LIZARD_DATASETS if wildcards.benchmark == "lizard" else PANNUKE_DATASETS,
        ),
        gene_json=lambda wildcards: OMICLIP_MARKER_GENES_DIR
        / f"{wildcards.benchmark}_{'pseudobulk' if wildcards.omiclip_model == 'omiclip_pseudobulk' else 'short'}.json",
    output:
        logits_csv=PATHOCELL_RESULTS / "{omiclip_model}_recheck" / "{benchmark}" / "omiclip_logits.csv",
    params:
        data_dir=lambda wildcards: PATHOCELL_DATA / "processed" / wildcards.benchmark,
        prediction_level="patch",
        batch_size=64,
        gene_sentences_json=lambda wildcards, input: str(input.gene_json),
    wildcard_constraints:
        omiclip_model="(omiclip|omiclip_pseudobulk)",
        benchmark="(lizard|pannuke)",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm=slurm_gres("medium", num_cpus=4, time="4:00:00"),
    script:
        "../scripts/run_omiclip_baseline.py"


rule omiclip_secondary_split_scores_recheck:
    input:
        baseline_csv=rules.omiclip_secondary_score_recheck.output.logits_csv,
    output:
        score=PATHOCELL_RESULTS / "{omiclip_model}_recheck" / "{benchmark}" / "{dataset}_{prediction_level}_scores_seed{seed}.csv",
    wildcard_constraints:
        omiclip_model="(omiclip|omiclip_pseudobulk)",
        benchmark="(lizard|pannuke)",
        prediction_level="patch",
        seed="0",
        dataset="[^/]+",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=1 partition=cmackall",
    script:
        "../scripts/split_baseline_logits.py"


# ── Disambiguate recheck rules from the generic *_cell_type_prediction rules.
# All four ce ll-type-prediction rules use a {model} wildcard, so without these
# orderings Snakemake would treat e.g. omiclip_recheck as a model name and try
# to load <jointemb_models>/omiclip_recheck.ckpt.

ruleorder: two_stage_pathocell_predict_recheck > pathocell_cell_type_prediction
ruleorder: two_stage_lizard_predict_recheck > lizard_cell_type_prediction
ruleorder: two_stage_pannuke_predict_recheck > pannuke_cell_type_prediction
ruleorder: omiclip_pathocell_split_scores_recheck > pathocell_cell_type_prediction
ruleorder: omiclip_pseudobulk_pathocell_split_scores_recheck > pathocell_cell_type_prediction
ruleorder: omiclip_secondary_split_scores_recheck > lizard_cell_type_prediction
ruleorder: omiclip_secondary_split_scores_recheck > pannuke_cell_type_prediction
