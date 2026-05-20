"""Prepare flat 2D image folders for this repository.

The upstream training code expects:

    DATASET_ROOT/
      train/
      val/
      test/

This script creates that structure from a flat image directory without changing
the original files. It can also convert RGB MRI exports to grayscale and resize
images during preparation.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def parse_split(value: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split must contain train,val,test ratios")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("split ratios must be positive")
    return tuple(x / total for x in parts)


def collect_images(source: Path, recursive: bool) -> list[Path]:
    iterator = source.rglob("*") if recursive else source.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: p.name)


def split_files(files: list[Path], split: tuple[float, float, float], seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * split[0])
    n_val = int(n_total * split[1])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_image(src: Path, dst: Path, image_mode: str | None, image_size: int | None, image_format: str) -> None:
    if src.suffix.lower() == ".npy":
        shutil.copy2(src, dst.with_suffix(".npy"))
        return

    if image_mode is None and image_size is None and src.suffix.lower() == f".{image_format.lower()}":
        shutil.copy2(src, dst.with_suffix(src.suffix.lower()))
        return

    with Image.open(src) as image:
        if image_mode is not None:
            image = image.convert(image_mode)
        if image_size is not None:
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)

        out_path = dst.with_suffix(f".{image_format.lower()}")
        save_kwargs = {}
        if image_format.lower() in {"jpg", "jpeg"}:
            save_kwargs.update({"quality": 95})
        image.save(out_path, **save_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a flat 2D image folder for diffusion training.")
    parser.add_argument("--source", required=True, type=Path, help="Flat source image directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset root.")
    parser.add_argument("--split", type=parse_split, default=parse_split("0.8,0.1,0.1"), help="Train,val,test ratios.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--mode", choices=["L", "RGB"], default=None, help="Optional output image mode.")
    parser.add_argument("--size", type=int, default=None, help="Optional square resize size.")
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png", help="Output format when converting.")
    parser.add_argument("--recursive", action="store_true", help="Search source recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output folder.")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)

    files = collect_images(args.source, args.recursive)
    if not files:
        raise RuntimeError(f"No supported image files found in {args.source}")

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{args.output} is not empty. Pass --overwrite to add/replace files.")

    for split_name in ["train", "val", "test"]:
        (args.output / split_name).mkdir(parents=True, exist_ok=True)

    grouped = split_files(files, args.split, args.seed)
    for split_name, split_files_ in grouped.items():
        split_dir = args.output / split_name
        for src in split_files_:
            dst_stem = split_dir / src.stem
            prepare_image(src, dst_stem, args.mode, args.size, args.format)

    print(f"Prepared {len(files)} images at {args.output}")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {len(grouped[split_name])}")


if __name__ == "__main__":
    main()
