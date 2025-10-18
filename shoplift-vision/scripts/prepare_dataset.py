"""Utility to arrange datasets into train/val/test splits."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _copy_file(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _validate_splits(splits: Tuple[float, float, float]) -> None:
    if not abs(sum(splits) - 1.0) < 1e-6:
        raise ValueError("Splits must sum to 1.0")


def _apply_split(
    df: pd.DataFrame,
    output_root: Path,
    splits: Tuple[float, float, float],
    symlink: bool,
    seed: int,
    stratify: bool,
) -> None:
    _validate_splits(splits)
    train_ratio, val_ratio, test_ratio = splits
    stratify_data = df["label"] if stratify else None
    train_df, temp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=stratify_data,
    )
    if test_ratio > 0:
        val_size = val_ratio / (val_ratio + test_ratio)
        stratify_temp = temp_df["label"] if stratify else None
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_size,
            random_state=seed,
            stratify=stratify_temp,
        )
    else:
        val_df, test_df = temp_df, pd.DataFrame(columns=df.columns)

    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        for _, row in split_df.iterrows():
            src = Path(row["path"]).expanduser()
            label = row["label"]
            dst = output_root / split_name / label / src.name
            _copy_file(src, dst, symlink)

    print("Dataset prepared:")
    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        if split_dir.exists():
            count = sum(1 for _ in split_dir.rglob("*"))
            print(f"  {split}: {count} files")


def _split_from_csv(csv_path: Path, **kwargs) -> None:
    df = pd.read_csv(csv_path)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'path' and 'label' columns")
    _apply_split(df, **kwargs)


def _split_from_directory(source_dir: Path, **kwargs) -> None:
    rows = []
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            for video in class_dir.glob("*"):
                if video.is_file():
                    rows.append({"path": video, "label": class_dir.name})
    if not rows:
        raise ValueError("No videos found in source directory")
    df = pd.DataFrame(rows)
    _apply_split(df, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--csv", type=str, help="CSV file with path,label columns")
    parser.add_argument("--source-dir", type=str, help="Directory with class subfolders")
    parser.add_argument("--output-root", type=str, required=True, help="Output root directory")
    parser.add_argument("--splits", type=float, nargs=3, default=(0.7, 0.2, 0.1))
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", action="store_true", help="Stratify splits by label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_args = dict(
        output_root=output_root,
        splits=(args.splits[0], args.splits[1], args.splits[2]),
        symlink=args.symlink,
        seed=args.seed,
        stratify=args.stratify,
    )
    if args.csv:
        _split_from_csv(Path(args.csv), **split_args)
    elif args.source_dir:
        _split_from_directory(Path(args.source_dir), **split_args)
    else:
        raise ValueError("Provide either --csv or --source-dir")


if __name__ == "__main__":
    main()
