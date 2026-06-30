"""
Convert paired data (lr/gt) into the HF Dataset format.
The generated dataset contains two columns, jpg_0 (GT) and jpg_1 (LQ),
matching the DPO training data format expected by train.py.
"""
import os
import argparse
from datasets import Dataset, Features, Image

def build_dataset(paired_dir, output_dir):
    lr_dir = os.path.join(paired_dir, "lr")
    gt_dir = os.path.join(paired_dir, "gt")

    filenames = sorted(os.listdir(gt_dir))
    # Keep only files that also have a matching lr counterpart
    filenames = [f for f in filenames if os.path.exists(os.path.join(lr_dir, f))]

    gt_paths = [os.path.join(gt_dir, f) for f in filenames]
    lr_paths = [os.path.join(lr_dir, f) for f in filenames]

    print(f"Found {len(filenames)} image pairs")

    ds = Dataset.from_dict(
        {"jpg_0": gt_paths, "jpg_1": lr_paths},
        features=Features({"jpg_0": Image(), "jpg_1": Image()})
    )

    ds.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    print(f"  Number of samples: {len(ds)}")
    print(f"  Columns: {ds.column_names}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_dir", type=str, default="./data/paired_train",
                        help="Paired data directory containing lr/ and gt/ subdirectories (with matching filenames)")
    parser.add_argument("--output_dir", type=str, default="./data/dataset",
                        help="Output HF Dataset directory")
    args = parser.parse_args()
    build_dataset(args.paired_dir, args.output_dir)
