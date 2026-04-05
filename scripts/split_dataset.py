"""
split_dataset.py
================
Stratified split of the full SciCite corpus on the isKeyCitation label.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --ratio 0.7 0.15 0.15
    python scripts/split_dataset.py --ratio 0.8 0.1 0.1 --seed 42 --out_dir data/raw/scicite/scicite

Outputs (in --out_dir):
    resplit_train.jsonl
    resplit_dev.jsonl
    resplit_test.jsonl
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_DIR = "data/raw/scicite/scicite"


def parse_args():
    p = argparse.ArgumentParser(description="Stratified resplit of SciCite on isKeyCitation.")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help="Directory containing train.jsonl, dev.jsonl, test.jsonl")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (defaults to --data_dir)")
    p.add_argument("--ratio", nargs=3, type=float, default=[0.80, 0.10, 0.10],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train / val / test proportions (must sum to 1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="resplit",
                   help="Prefix for output files: <prefix>_train.jsonl etc.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    train_ratio, val_ratio, test_ratio = args.ratio
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0 (got {total:.4f})")

    SEP = "=" * 60
    print(SEP)
    print("  STRATIFIED RESPLIT — SciCite isKeyCitation")
    print(f"  Split: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}  "
          f"|  Seed: {args.seed}")
    print(SEP)

    # ── Load & pool all original splits ────────────────────────────────────────
    parts = []
    for fname in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        df = pd.read_json(path, lines=True)
        df["_source"] = fname
        parts.append(df)

    df_all = pd.concat(parts, ignore_index=True)
    df_all["isKeyCitation"] = df_all["isKeyCitation"].fillna(False).astype(bool)
    labels = df_all["isKeyCitation"].astype(int)
    n_total = len(df_all)
    n_key   = labels.sum()

    print(f"\n  Pooled corpus : {n_total:,} samples")
    print(f"  Key citations : {n_key:,}  ({n_key/n_total:.1%})")
    print(f"  Not-Key       : {n_total-n_key:,}  ({(n_total-n_key)/n_total:.1%})")
    print()
    print("  Original split key-citation ratios:")
    for src, part in zip(["train.jsonl", "dev.jsonl", "test.jsonl"], parts):
        kr = part["isKeyCitation"].fillna(False).mean()
        print(f"    {src:<14s}: {kr:.3f}  ({kr:.1%} Key)")

    # ── Stratified split ────────────────────────────────────────────────────────
    # Step 1: train vs (val+test)
    df_train, df_temp, y_train, y_temp = train_test_split(
        df_all, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=args.seed,
    )

    # Step 2: val vs test from the temp portion
    val_frac_of_temp = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_frac_of_temp),
        stratify=y_temp,
        random_state=args.seed,
    )

    splits = {"train": df_train, "val": df_val, "test": df_test}
    expected_sizes = {
        "train": int(n_total * train_ratio),
        "dev":   int(n_total * val_ratio),
        "test":  int(n_total * test_ratio),
    }

    print(f"\n  New stratified split:")
    print(f"  {'Split':<8} {'N':>6}  {'Key':>5}  {'Key%':>6}  {'File'}")
    print(f"  {'-'*55}")
    saved_paths = {}
    for name, df in splits.items():
        kr   = df["isKeyCitation"].mean()
        n_k  = df["isKeyCitation"].sum()
        fname = f"{args.prefix}_{name}.jsonl"
        path  = os.path.join(out_dir, fname)
        df.drop(columns=["_source"], errors="ignore").to_json(
            path, orient="records", lines=True
        )
        saved_paths[name] = path
        print(f"  {name:<8} {len(df):>6,}  {n_k:>5,}  {kr:>5.1%}  → {path}")

    print(f"\n  Key ratio std across splits: "
          f"{pd.Series({n: df['isKeyCitation'].mean() for n, df in splits.items()}).std():.5f}"
          f"  (ideally < 0.001)")
    print(f"\n{SEP}")
    print("  ✅ Resplit complete.")
    print(f"  Use these files with:")
    print(f"    python scripts/train_scicite_model.py \\")
    print(f"      --train_file {saved_paths['train']} \\")
    print(f"      --dev_file   {saved_paths['dev']} \\")
    print(f"      --test_file  {saved_paths['test']}")
    print(SEP)


if __name__ == "__main__":
    main()
