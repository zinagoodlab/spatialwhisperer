from pathlib import Path
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def patch_id_to_image_stem(patch_id: str) -> str:
    return patch_id.rsplit("_crop_", 1)[0]


def main():
    args = parse_args()
    df = pd.read_csv(args.source_csv)
    manifest = df[["sample_idx", "orig_indices", "orig_ids"]].copy()
    manifest = manifest.rename(
        columns={
            "sample_idx": "subset_position",
            "orig_indices": "dataset_index",
            "orig_ids": "patch_id",
        }
    )
    manifest["image_stem"] = manifest["patch_id"].map(patch_id_to_image_stem)
    manifest["crop_id"] = manifest["patch_id"].str.rsplit("_crop_", n=1).str[-1]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
