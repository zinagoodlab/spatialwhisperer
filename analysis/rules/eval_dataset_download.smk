# Download + post-process the three eval-only datasets so the paper pipeline is
# reproducible from a fresh checkout. No upstream MUSK bundle, no manual conversion.
#
# Provenance:
#   - Kriegsmann skin (heiDATA) -> resources/kriegsmann_skin/data/
#       Direct from Kriegsmann et al. (2022), doi:10.11588/data/7QCR8S.
#       data.zip ~3.7 GB; contains tiles-v2.csv + tiles/<class>/*.tif at top level.
#   - Lizard + PanNuke (Kainmueller-Lab/PathoCell on HF) -> LMDB
#       -> convert via scripts/convert_lmdb_to_hdf.py to the per-sample / batched
#       HDFs that lizard_process_data / pannuke_process_data consume.
#       Note: Kainmueller-Lab/PathoCell is a *gated* dataset; the running HF token
#       must have access (request once at https://huggingface.co/datasets/Kainmueller-Lab/PathoCell).
#   - PathoCell-CRC (existing pathocell_download_dataset rule, unchanged) supplies
#       the CRC HDFs from the same gated repo.

# ----------------------------------------------------------------------------
# Kriegsmann skin
# ----------------------------------------------------------------------------

KRIEGSMANN_ROOT = PROJECT_DIR / "resources/kriegsmann_skin"


rule download_kriegsmann_skin:
    """Download and unzip the Kriegsmann et al. (2022) skin H&E dataset (~3.7 GB).

    heiDATA file id 7166 = data.zip (the full tiled dataset + tiles-v2.csv).
    Layout after unzip (the zip's internal top level is `data/`, so we extract
    directly into KRIEGSMANN_ROOT):
        {KRIEGSMANN_ROOT}/data/
            class_dict.json
            tiles/<class>/<tile>.jpg    (16 class subdirs, ~129k tiles total)
            tiles-v2.csv                (14 cols incl. file, class, set ∈
                                         {Train, Validation, Test}, case)
    """
    output:
        csv=KRIEGSMANN_ROOT / "data/tiles-v2.csv",
        class_dict=KRIEGSMANN_ROOT / "data/class_dict.json",
        marker=touch(KRIEGSMANN_ROOT / "download_complete.marker"),
    params:
        url="https://heidata.uni-heidelberg.de/api/access/datafile/7166",
        zip_sha256_expected="",  # filled in after first verified download
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=2",
    log:
        "logs/download_kriegsmann_skin.log",
    shell: """
        set -euo pipefail
        LOG=$(realpath {log})
        TMP=$(mktemp -d)
        trap 'rm -rf "$TMP"' EXIT

        echo "Downloading Kriegsmann skin data.zip from heiDATA ($(date -Iseconds))" >> $LOG
        curl -fsSL --retry 3 --retry-delay 10 -o "$TMP/data.zip" {params.url} 2>>$LOG
        SIZE=$(stat -c%s "$TMP/data.zip")
        echo "Downloaded $SIZE bytes" >> $LOG

        mkdir -p {KRIEGSMANN_ROOT}
        # data.zip's internal top level is `data/`, so extracting into the root
        # produces {KRIEGSMANN_ROOT}/data/{tiles-v2.csv, tiles/, class_dict.json}.
        unzip -q -o "$TMP/data.zip" -d {KRIEGSMANN_ROOT}
        echo "Unzipped to {KRIEGSMANN_ROOT}/data/" >> $LOG

        # Sanity: tiles-v2.csv must exist and have a Test split.
        TEST_N=$(awk -F, 'NR>1 && $NF=="Test" {{c++}} END {{print c+0}}' {output.csv})
        echo "Test split rows: $TEST_N" >> $LOG
        test "$TEST_N" -ge 1000  # paper uses 386 patients ≈ ~13k test tiles; abort if too few
    """


# ----------------------------------------------------------------------------
# Lizard (via Kainmueller-Lab/PathoCell LMDB)
# ----------------------------------------------------------------------------

LIZARD_RAW = PATHOCELL_DATA / "raw_lmdb/lizard"


rule download_lizard_lmdb:
    """Fetch the Lizard subset (LMDB + splits) from Kainmueller-Lab/PathoCell.

    Lizard LMDB: data.mdb + lock.mdb (~1.67 GB).
    Splits: data/lizard/splits/lizard_default_train_test_val_split.csv (~3.4 KB).
    Requires HF token with access to the gated repo.
    """
    output:
        lmdb_dir=directory(LIZARD_RAW / "lizard_lmdb"),
        splits_csv=LIZARD_RAW / "splits/lizard_default_train_test_val_split.csv",
        marker=touch(LIZARD_RAW / "download_complete.marker"),
    params:
        repo_id="Kainmueller-Lab/PathoCell",
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=2",
    log:
        "logs/download_lizard_lmdb.log",
    shell: """
        set -euo pipefail
        LOG=$(realpath {log})
        mkdir -p {LIZARD_RAW}
        # HF cli ignores --local-dir's parent prefix; download into a staging dir
        # and move only the lizard subtree to keep paths predictable.
        TMP=$(mktemp -d)
        trap 'rm -rf "$TMP"' EXIT

        echo "Downloading Lizard LMDB + splits from {params.repo_id}" >> $LOG
        huggingface-cli download {params.repo_id} \
            --repo-type dataset \
            --include "data/lizard/lizard_lmdb/*" \
            --include "data/lizard/splits/*" \
            --local-dir "$TMP" 2>>$LOG

        rsync -a "$TMP/data/lizard/lizard_lmdb/" {output.lmdb_dir}/
        mkdir -p $(dirname {output.splits_csv})
        rsync -a "$TMP/data/lizard/splits/" $(dirname {output.splits_csv})/

        echo "Lizard download complete: $(du -sh {output.lmdb_dir} | cut -f1)" >> $LOG
    """


rule convert_lizard_lmdb_to_hdf:
    """Stitch test-split Lizard tiles into per-sample HDFs (~70 samples)."""
    input:
        lmdb_dir=rules.download_lizard_lmdb.output.lmdb_dir,
        splits_csv=rules.download_lizard_lmdb.output.splits_csv,
    output:
        output_dir=directory(PATHOCELL_DATA / "converted/lizard_hdf"),
        sample_list=PATHOCELL_DATA / "converted/lizard_hdf/sample_list.txt",
    params:
        tile_size=224,
        batch_size=1,  # one HDF per sample (matches Sherlock-derived artifacts)
    conda:
        "cellwhisperer"
    resources:
        mem_mb=32000,
        slurm="cpus-per-task=2 partition=cmackall",
    log:
        "logs/convert_lizard_lmdb_to_hdf.log",
    script:
        "../scripts/convert_lmdb_to_hdf.py"


# ----------------------------------------------------------------------------
# PanNuke (via Kainmueller-Lab/PathoCell LMDB; split into 3 parts on HF)
# ----------------------------------------------------------------------------

PANNUKE_RAW = PATHOCELL_DATA / "raw_lmdb/pannuke"


rule download_pannuke_lmdb_parts:
    """Fetch the three split PanNuke LMDB parts + splits CSV.

    HF stores data.mdb as part_aa (48G) + part_ab (48G) + part_ac (8G).
    Disk: ~104 GB just for parts, +104 GB for concatenated LMDB. Use a node
    with ≥250 GB scratch (ILC ampere/blackwell nodes via /lfs/local/0).
    """
    output:
        parts_dir=directory(PANNUKE_RAW / "pannuke_lmdb_parts"),
        splits_csv=PANNUKE_RAW / "splits/train_test_val_split.csv",
        marker=touch(PANNUKE_RAW / "download_complete.marker"),
    params:
        repo_id="Kainmueller-Lab/PathoCell",
    resources:
        mem_mb=8000,
        slurm="cpus-per-task=4",
    log:
        "logs/download_pannuke_lmdb_parts.log",
    shell: """
        set -euo pipefail
        LOG=$(realpath {log})
        mkdir -p {PANNUKE_RAW}
        TMP=$(mktemp -d)
        trap 'rm -rf "$TMP"' EXIT

        echo "Downloading PanNuke LMDB parts + splits from {params.repo_id}" >> $LOG
        huggingface-cli download {params.repo_id} \
            --repo-type dataset \
            --include "data/pannuke/pannuke_lmdb/*" \
            --include "data/pannuke/splits/*" \
            --local-dir "$TMP" 2>>$LOG

        mkdir -p {output.parts_dir}
        rsync -a "$TMP/data/pannuke/pannuke_lmdb/" {output.parts_dir}/
        mkdir -p $(dirname {output.splits_csv})
        rsync -a "$TMP/data/pannuke/splits/" $(dirname {output.splits_csv})/

        echo "PanNuke parts download complete: $(du -sh {output.parts_dir} | cut -f1)" >> $LOG
    """


rule concat_pannuke_lmdb_parts:
    """Concatenate split LMDB parts (~104 GB) into a single data.mdb.

    The original LMDB was sharded by Kainmueller into 48G+48G+8G chunks to fit
    under HF's per-file limit. They concatenate bit-for-bit (no header).
    """
    input:
        parts_dir=rules.download_pannuke_lmdb_parts.output.parts_dir,
    output:
        lmdb_dir=directory(PANNUKE_RAW / "pannuke_lmdb"),
    resources:
        mem_mb=4000,
        slurm="cpus-per-task=2",
    log:
        "logs/concat_pannuke_lmdb_parts.log",
    shell: """
        set -euo pipefail
        LOG=$(realpath {log})
        mkdir -p {output.lmdb_dir}
        # Concatenate the data.mdb chunks. lock.mdb is fine to copy as-is.
        cat {input.parts_dir}/data.mdb.part_aa \
            {input.parts_dir}/data.mdb.part_ab \
            {input.parts_dir}/data.mdb.part_ac > {output.lmdb_dir}/data.mdb 2>>$LOG
        cp {input.parts_dir}/lock.mdb {output.lmdb_dir}/lock.mdb
        echo "Concat complete: $(du -sh {output.lmdb_dir}/data.mdb | cut -f1)" >> $LOG
    """


rule convert_pannuke_lmdb_to_hdf:
    """Stitch test-split PanNuke tiles into batched HDFs (~51 batches of 50 samples)."""
    input:
        lmdb_dir=rules.concat_pannuke_lmdb_parts.output.lmdb_dir,
        splits_csv=rules.download_pannuke_lmdb_parts.output.splits_csv,
    output:
        output_dir=directory(PATHOCELL_DATA / "converted/pannuke_hdf"),
        sample_list=PATHOCELL_DATA / "converted/pannuke_hdf/sample_list.txt",
    params:
        tile_size=224,
        batch_size=50,  # ~51 HDFs of ≤50 tiles each (matches Sherlock-derived artifacts)
    conda:
        "cellwhisperer"
    resources:
        mem_mb=64000,
        slurm="cpus-per-task=4 partition=cmackall",
    log:
        "logs/convert_pannuke_lmdb_to_hdf.log",
    script:
        "../scripts/convert_lmdb_to_hdf.py"
