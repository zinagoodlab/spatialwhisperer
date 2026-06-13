from pathlib import Path


QUILT_TEXT_CONTROL_ROOT = Path("/dfs/user/moritzs/quilt1m_control")
SNAP_OAK_ROOT = Path.home() / "oak" / "moritzs" / "cellwhisperer"
SECTION43_CURATED_SUBSET_CSV = (
    SNAP_OAK_ROOT
    / "results/spatialwhisperer_eval/csv_logging/sweval___cellxgene_census__archs4_geo__hest1k___quilt1m_curated/test_individual_clip_scores.csv"
)


rule quilt_text_harmonization_subset_manifest:
    input:
        source_csv=SECTION43_CURATED_SUBSET_CSV
    output:
        manifest=QUILT_TEXT_CONTROL_ROOT / "subset/section43_patch_subset.csv"
    params:
        project_dir=PROJECT_DIR
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=2 partition=il-interactive time=00:20:00"
    shell: """
        export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/cellwhisperer
        export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
        export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
        export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share
        cd {params.project_dir}
        uv run --no-progress python analysis/experiments/plip_text_harmonization_control/scripts/build_subset_manifest.py \
            --source-csv {input.source_csv} \
            --output-csv {output.manifest}
    """


rule quilt_text_harmonization_model_condition:
    input:
        manifest=rules.quilt_text_harmonization_subset_manifest.output.manifest
    output:
        metrics=QUILT_TEXT_CONTROL_ROOT / "results/{model}/{caption_condition}/metrics.json",
        per_sample=QUILT_TEXT_CONTROL_ROOT / "results/{model}/{caption_condition}/per_sample_scores.csv",
        similarity=QUILT_TEXT_CONTROL_ROOT / "results/{model}/{caption_condition}/similarity_matrix.npz",
    params:
        original_h5ad_dir=SNAP_OAK_ROOT / "results/quilt1m/h5ads",
        curated_h5ad_dir=SNAP_OAK_ROOT / "results/quilt1m_curated/h5ads",
        output_dir=lambda wildcards: QUILT_TEXT_CONTROL_ROOT / f"results/{wildcards.model}/{wildcards.caption_condition}",
        project_dir=PROJECT_DIR,
    resources:
        mem_mb=64000,
        slurm="cpus-per-task=8 gres=gpu:RTX8000:1 partition=il-interactive time=04:00:00"
    shell: """
        export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/cellwhisperer
        export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
        export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
        export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share
        cd {params.project_dir}
        uv run --no-progress python analysis/experiments/plip_text_harmonization_control/scripts/run_quilt_retrieval_control.py \
            --model {wildcards.model} \
            --caption-condition {wildcards.caption_condition} \
            --subset-manifest {input.manifest} \
            --original-h5ad-dir {params.original_h5ad_dir} \
            --curated-h5ad-dir {params.curated_h5ad_dir} \
            --output-dir {params.output_dir}
    """


rule quilt_text_harmonization_summary:
    input:
        expand(
            rules.quilt_text_harmonization_model_condition.output.metrics,
            model=["plip"],
            caption_condition=["original", "curated"],
        )
    output:
        csv=QUILT_TEXT_CONTROL_ROOT / "results/control_summary.csv",
        md=QUILT_TEXT_CONTROL_ROOT / "results/control_summary.md",
        long_csv=QUILT_TEXT_CONTROL_ROOT / "results/control_summary_long.csv",
    params:
        results_root=QUILT_TEXT_CONTROL_ROOT / "results",
        project_dir=PROJECT_DIR,
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=2 partition=il-interactive time=00:20:00"
    shell: """
        export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/cellwhisperer
        export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
        export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
        export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share
        cd {params.project_dir}
        uv run --no-progress python analysis/experiments/plip_text_harmonization_control/scripts/summarize_control_results.py \
            --results-root {params.results_root} \
            --output-csv {output.csv} \
            --output-md {output.md}
    """
