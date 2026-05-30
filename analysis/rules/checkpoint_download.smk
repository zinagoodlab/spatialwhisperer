# Fetch the published SpatialWhisperer checkpoint from Hugging Face so the
# eval pipeline can reproduce paper numbers WITHOUT training anything locally.
#
# The HF release `Good-Lab/spatialwhisperer` ships one checkpoint named
# `spatialwhisperer.ckpt`. Pipeline-internal rules expect the file at
# `<results>/models/jointemb/spatialwhisperer_<dataset_combo>.ckpt`. This
# rule renames into that location for the canonical paper "Trimodal" combo
# (`cellxgene_census__archs4_geo__hest1k`), and ruleorder prefers it over
# `train_spatialwhisperer` so the public reproducer never trains.

PUBLISHED_MODEL_HF_REPO = "Good-Lab/spatialwhisperer"
PUBLISHED_MODEL_DATASET_COMBO = "cellxgene_census__archs4_geo__hest1k"
PUBLISHED_MODEL_NAME = f"spatialwhisperer_{PUBLISHED_MODEL_DATASET_COMBO}"


rule download_spatialwhisperer_checkpoint:
    """Pull the published Lightning checkpoint from Good-Lab/spatialwhisperer on HF."""
    output:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{PUBLISHED_MODEL_NAME}.ckpt",
    params:
        hf_repo=PUBLISHED_MODEL_HF_REPO,
        hf_filename="spatialwhisperer.ckpt",
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=2",
    shell: """
        mkdir -p $(dirname {output.model})
        # Drop into a temp dir so hf_hub_download doesn't pollute jointemb/
        TMPDIR=$(mktemp -d)
        huggingface-cli download \
            {params.hf_repo} \
            {params.hf_filename} \
            --local-dir $TMPDIR \
            --local-dir-use-symlinks False
        mv $TMPDIR/{params.hf_filename} {output.model}
        rm -rf $TMPDIR
    """


ruleorder: download_spatialwhisperer_checkpoint > train_spatialwhisperer
