from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_factual import read_jsonl, write_jsonl


SCORED_PATH = Path("data/factual_conflict/factual_conflict_scored.jsonl")
OUT_PATH = Path("data/factual_conflict/factual_conflict_margin_matched.jsonl")
OUT_CSV_PATH = Path("artifacts/results/factual_conflict_margin_matched.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--require-positive-clean", action="store_true")
    parser.add_argument("--require-positive-correction", action="store_true")
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Optional margin bin width. If set, keep clean/correction pairs in same bin.",
    )
    args = parser.parse_args()

    rows = read_jsonl(SCORED_PATH)
    df = pd.DataFrame(rows)

    keep = df[df["abs_margin_gap_clean_vs_correction"] <= args.tau].copy()
    if args.require_positive_clean:
        keep = keep[keep["margin_clean"] > 0]
    if args.require_positive_correction:
        keep = keep[keep["margin_correction"] > 0]

    if args.bin_width is not None:
        clean_bin = (keep["margin_clean"] / args.bin_width).round().astype(int)
        corr_bin = (keep["margin_correction"] / args.bin_width).round().astype(int)
        keep = keep[clean_bin == corr_bin].copy()

    keep = keep.sort_values("abs_margin_gap_clean_vs_correction")
    write_jsonl(OUT_PATH, keep.to_dict("records"))
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(OUT_CSV_PATH, index=False)

    print(f"raw scored examples: {len(df)}")
    print(f"kept margin-matched examples: {len(keep)}")
    print(f"tau: {args.tau}")
    if args.bin_width is not None:
        print(f"bin_width: {args.bin_width}")
    print(f"saved JSONL to {OUT_PATH}")
    print(f"saved CSV to {OUT_CSV_PATH}")
    if len(keep) < 30:
        print("WARNING: fewer than 30 examples; smoke-test only.")
    if len(keep) < 100:
        print(
            "WARNING: fewer than 100 examples; expand FACTS or switch model before "
            "expensive patching."
        )
    print(
        keep[
            [
                "subject",
                "relation",
                "true_object",
                "false_object",
                "margin_clean",
                "margin_correction",
                "abs_margin_gap_clean_vs_correction",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
