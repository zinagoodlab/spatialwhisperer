rule all:
    input:
        PLIP_CONTROL_OUTPUT_DIR / "results_table.md",
    default_target: True


rule plip_harmonization_control:
    output:
        metrics=PLIP_CONTROL_OUTPUT_DIR / "metrics.csv",
        sample_metadata=PLIP_CONTROL_OUTPUT_DIR / "sample_metadata.csv",
        results_table_csv=PLIP_CONTROL_OUTPUT_DIR / "results_table.csv",
        results_table_md=PLIP_CONTROL_OUTPUT_DIR / "results_table.md",
        manifest=PLIP_CONTROL_OUTPUT_DIR / "run_manifest.json",
    params:
        script=PROJECT_DIR / "src/spotwhisperer_eval/scripts/plip_quilt1m_harmonization_control.py",
        original_metadata=PLIP_CONTROL_ORIGINAL_METADATA,
        curated_metadata=PLIP_CONTROL_CURATED_METADATA,
        image_zip_root=PLIP_CONTROL_IMAGE_ZIP_ROOT,
        output_dir=PLIP_CONTROL_OUTPUT_DIR,
        sample_size=PLIP_CONTROL_SAMPLE_SIZE,
        seed=PLIP_CONTROL_SEED,
        model_name=PLIP_CONTROL_MODEL_NAME,
        bridge_original=PLIP_CONTROL_BRIDGE_ORIGINAL,
        bridge_curated=PLIP_CONTROL_BRIDGE_CURATED,
    resources:
        mem_mb=180000,
        slurm="cpus-per-task=8 gres=gpu:rtx8000:1",
    shell:
        """
        cd {PROJECT_DIR}
        uv run --no-progress python {params.script} \
            --original-metadata {params.original_metadata} \
            --curated-metadata {params.curated_metadata} \
            --image-zip-root {params.image_zip_root} \
            --output-dir {params.output_dir} \
            --sample-size {params.sample_size} \
            --seed {params.seed} \
            --model-name {params.model_name} \
            --bridge-original {params.bridge_original} \
            --bridge-curated {params.bridge_curated}
        """
