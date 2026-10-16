"""Audit whether 2B obstruction detects conflict beyond margin, fragility, and transport-free scores.

Bootstrap is over examples, not rows. This does not add a prompt-shift / paraphrase condition;
that remains a separate data-collection experiment.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.factual_obstruction_eval import auc_scores, partial_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare factual obstruction to confound baselines.")
    parser.add_argument("--long", default="artifacts/factual_low_rank/results/low_rank_obstruction_long.csv")
    parser.add_argument("--pairs", default="artifacts/factual_low_rank/results/low_rank_obstruction_pairs.csv")
    parser.add_argument("--output", default="artifacts/factual_low_rank/robustness/obstruction_vs_confounds.csv")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--variant", default="clean_pca")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_04b first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
    return values[idx]


def filter_rows(rows: list[dict], args: argparse.Namespace, learned_only: bool = True) -> list[dict]:
    out = []
    for row in rows:
        if row.get("variant") != args.variant:
            continue
        if int(float(row.get("rank", -1))) != args.rank:
            continue
        if args.split != "all" and row.get("split") != args.split:
            continue
        if learned_only and row.get("transport_control") not in {"", "learned", None}:
            if row.get("transport_control") != "learned":
                continue
        out.append(row)
    return out


def auc_from_long(long_rows: list[dict], score_col: str) -> float:
    return auc_scores(
        [int(float(row["label_conflict"])) for row in long_rows],
        [float(row[score_col]) for row in long_rows],
    )


def main() -> None:
    args = parse_args()
    long_rows = filter_rows(read_csv(args.long), args)
    pair_rows = filter_rows(read_csv(args.pairs), args)
    if not long_rows or not pair_rows:
        raise RuntimeError("No rows after variant/rank/split filter. Check --split (try test or all).")

    by_id: dict[str, list[dict]] = {}
    for row in long_rows:
        by_id.setdefault(row["id"], []).append(row)
    ids = sorted(by_id)

    scores = {
        "obstruction_normalized": "obstruction (main)",
        "obstruction_raw": "obstruction raw",
        "abs_margin": "absolute margin (confidence)",
        "fragility": "fragility = O / |m|",
    }

    point = {}
    for col in scores:
        point[col] = auc_from_long(long_rows, col)
    point["partial_corr_O_conflict_given_margin"] = partial_corr(
        long_rows, "obstruction_normalized", "label_conflict", ["margin"]
    )

    rng = random.Random(args.seed)
    boots: dict[str, list[float]] = {k: [] for k in list(scores) + ["partial_corr_O_conflict_given_margin"]}
    for _ in range(args.n_bootstrap):
        sampled = [ids[rng.randrange(len(ids))] for _ in ids]
        sample = []
        for ex_id in sampled:
            sample.extend(by_id[ex_id])
        for col in scores:
            boots[col].append(auc_from_long(sample, col))
        boots["partial_corr_O_conflict_given_margin"].append(
            partial_corr(sample, "obstruction_normalized", "label_conflict", ["margin"])
        )

    rows = []
    for key, label in {**scores, "partial_corr_O_conflict_given_margin": "partial corr(O, conflict | margin)"}.items():
        vals = boots[key]
        rows.append(
            {
                "variant": args.variant,
                "rank": args.rank,
                "split": args.split,
                "n_examples": len(ids),
                "n_rows": len(long_rows),
                "n_pairs": len(pair_rows),
                "score": key,
                "label": label,
                "point": point[key],
                "boot_mean": sum(vals) / len(vals),
                "ci_low": percentile(vals, 0.025),
                "ci_high": percentile(vals, 0.975),
                "n_bootstrap": len(vals),
                "note": "bootstrap over examples, not rows",
            }
        )
    write_csv(args.output, rows)

    o = next(row for row in rows if row["score"] == "obstruction_normalized")
    m = next(row for row in rows if row["score"] == "abs_margin")
    print(f"examples={len(ids)} split={args.split} variant={args.variant} r{args.rank}")
    print(f"O AUC {o['point']:.4f} [{o['ci_low']:.4f}, {o['ci_high']:.4f}]")
    print(f"|margin| AUC {m['point']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]")
    print(f"partial corr(O, conflict | margin) {point['partial_corr_O_conflict_given_margin']:.4f}")
    if o["ci_low"] > m["ci_high"]:
        print("O interval is above |margin|: obstruction is not just residual confidence.")
    else:
        print("O does not cleanly dominate |margin| on this split. Do not claim monitoring beyond confidence.")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
