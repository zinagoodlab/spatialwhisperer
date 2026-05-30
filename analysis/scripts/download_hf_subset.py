"""Download a subset of files from a HuggingFace dataset via snapshot_download.

Used by `download_lizard_lmdb` / `download_pannuke_lmdb_parts` because
`huggingface-cli download --include` was silently dropping the LMDB files
even though they exist in the repo (only the `splits/*` patterns took
effect). `snapshot_download` with `allow_patterns=[...]` is the reliable
multi-pattern path.

Inputs (Snakemake):
- snakemake.params.repo_id: HF dataset repo id
- snakemake.params.allow_patterns: list of fnmatch patterns to fetch
- snakemake.params.staging_dir: where to materialise the HF tree
- snakemake.params.subdirs_to_move: list of (src_relpath, dst_path) pairs
  to rsync from `staging_dir` to the rule's declared outputs
"""

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = snakemake.params.repo_id
allow_patterns = list(snakemake.params.allow_patterns)
staging_dir = Path(snakemake.params.staging_dir)
subdirs_to_move = list(snakemake.params.subdirs_to_move)

staging_dir.mkdir(parents=True, exist_ok=True)

print(f"snapshot_download {repo_id} → {staging_dir}")
print(f"  allow_patterns = {allow_patterns}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(staging_dir),
    allow_patterns=allow_patterns,
    token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
)

for src_rel, dst in subdirs_to_move:
    src = staging_dir / src_rel
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Expected {src} after snapshot_download")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)
    print(f"  moved {src} → {dst}")

print("done")
