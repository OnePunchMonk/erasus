"""
download_datasets.py — Download benchmark datasets for Erasus.

Downloads and prepares the following datasets:
- TOFU (Task of Fictitious Unlearning)
- WMDP (Weapons of Mass Destruction Proxy)
- I2P (Inappropriate Image Prompts)

Usage::

    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets tofu,wmdp
    python scripts/download_datasets.py --output ./data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


AVAILABLE_DATASETS = {
    "tofu": {
        "hf_name": "locuslab/TOFU",
        "description": "Task of Fictitious Unlearning — QA about fictional authors",
    },
    "wmdp": {
        "hf_name": "cais/wmdp",
        "description": "Weapons of Mass Destruction Proxy benchmark",
    },
    "i2p": {
        "hf_name": "AIML-TUDA/i2p",
        "description": "Inappropriate Image Prompts for diffusion models",
    },
}


def download_tofu(output_dir: Path):
    """Download TOFU dataset from HuggingFace."""
    print("  Downloading TOFU...")
    tofu_dir = output_dir / "tofu"
    tofu_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        for split in ["forget_01", "forget_05", "forget_10", "retain"]:
            print(f"    Split: {split}... ", end="", flush=True)
            try:
                ds = load_dataset("locuslab/TOFU", split=split)
                samples = [{"question": s["question"], "answer": s["answer"]} for s in ds]
                with open(tofu_dir / f"{split}.json", "w") as f:
                    json.dump(samples, f, indent=2)
                print(f"✓ ({len(samples)} samples)")
            except Exception as e:
                print(f"⚠ Skipped ({e})")

    except ImportError:
        print("  ⚠ 'datasets' library not installed. Run: pip install datasets")


def download_wmdp(output_dir: Path):
    """Download WMDP dataset from HuggingFace."""
    print("  Downloading WMDP...")
    wmdp_dir = output_dir / "wmdp"
    wmdp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        for subset in ["bio", "cyber"]:
            print(f"    Subset: {subset}... ", end="", flush=True)
            try:
                ds = load_dataset("cais/wmdp", f"wmdp-{subset}", split="test")
                samples = list(ds)
                with open(wmdp_dir / f"wmdp_{subset}.json", "w") as f:
                    json.dump(samples, f, indent=2, default=str)
                print(f"✓ ({len(samples)} samples)")
            except Exception as e:
                print(f"⚠ Skipped ({e})")

    except ImportError:
        print("  ⚠ 'datasets' library not installed. Run: pip install datasets")


def download_i2p(output_dir: Path):
    """Download I2P prompts from HuggingFace."""
    print("  Downloading I2P...")
    i2p_dir = output_dir / "i2p"
    i2p_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print("    Loading prompts... ", end="", flush=True)
        try:
            ds = load_dataset("AIML-TUDA/i2p", split="train")
            prompts = [{"prompt": s["prompt"], "categories": s.get("categories", [])} for s in ds]
            with open(i2p_dir / "i2p_prompts.json", "w") as f:
                json.dump(prompts, f, indent=2)
            print(f"✓ ({len(prompts)} prompts)")
        except Exception as e:
            print(f"⚠ Skipped ({e})")

    except ImportError:
        print("  ⚠ 'datasets' library not installed. Run: pip install datasets")


def main():
    parser = argparse.ArgumentParser(description="Download Erasus benchmark datasets")
    parser.add_argument(
        "--datasets",
        default=",".join(AVAILABLE_DATASETS.keys()),
        help="Comma-separated list of datasets to download",
    )
    parser.add_argument(
        "--output",
        default="./data",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = args.datasets.split(",")

    print("=" * 50)
    print("  Erasus — Dataset Downloader")
    print("=" * 50)
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Datasets: {', '.join(requested)}")

    download_fns = {
        "tofu": download_tofu,
        "wmdp": download_wmdp,
        "i2p": download_i2p,
    }

    for ds_name in requested:
        ds_name = ds_name.strip()
        if ds_name in download_fns:
            print(f"\n  [{ds_name.upper()}]")
            download_fns[ds_name](output_dir)
        else:
            print(f"\n  ⚠ Unknown dataset: {ds_name}")

    print("\n✅ Downloads complete!")


if __name__ == "__main__":
    main()
