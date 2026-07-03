from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.factual_obstruction_eval import (
    auc_scores,
    bootstrap_metric_ci,
    evaluate_obstruction_metrics,
    partial_corr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness checks for low-rank factual obstruction.")
    parser.add_argument("--metrics", default="artifacts/factual_low_rank/results/low_rank_obstruction_metrics.csv")
    parser.add_argument("--pairs", default="artifacts/factual_low_rank/results/low_rank_obstruction_pairs.csv")
    parser.add_argument("--long", default="artifacts/factual_low_rank/results/low_rank_obstruction_long.csv")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="artifacts/factual_low_rank/robustness")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict], cols: list[str]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(col, "") for col in cols)
        groups.setdefault(key, []).append(row)
    return groups


def long_group_for_pairs(long_rows: list[dict], pair_group: list[dict]) -> list[dict]:
    ids = {row["id"] for row in pair_group}
    if not pair_group:
        return []
    first = pair_group[0]
    return [
        row for row in long_rows
        if row["id"] in ids
        and row.get("model") == first.get("model")
        and row.get("variant") == first.get("variant")
        and row.get("rank") == first.get("rank")
        and row.get("transport_control") == first.get("transport_control")
        and row.get("split") == first.get("split")
    ]


def auc_metric_from_long(rows: list[dict]) -> float:
    return auc_scores(
        [int(float(row["label_conflict"])) for row in rows],
        [float(row["obstruction_normalized"]) for row in rows],
    )


def mean_delta_metric(rows: list[dict]) -> float:
    values = [float(row["delta_O"]) for row in rows]
    return sum(values) / len(values) if values else float("nan")


def bootstrap_summary(long_rows: list[dict], pair_rows: list[dict], n_bootstrap: int, seed: int) -> list[dict]:
    out = []
    pair_groups = group_rows(pair_rows, ["model", "variant", "rank", "transport_control", "split"])
    for key, pair_group in pair_groups.items():
        model, variant, rank, control, split = key
        lg = long_group_for_pairs(long_rows, pair_group)
        auc_ci = bootstrap_metric_ci(lg, auc_metric_from_long, n_bootstrap, seed)
        pcorr_ci = bootstrap_metric_ci(
            lg,
            lambda rows: partial_corr(rows, "obstruction_normalized", "label_conflict", ["margin"]),
            n_bootstrap,
            seed + 1,
        )
        delta_ci = bootstrap_metric_ci(pair_group, mean_delta_metric, n_bootstrap, seed + 2)
        out.append(
            {
                "model": model,
                "variant": variant,
                "rank": rank,
                "transport_control": control,
                "split": split,
                "n_pairs": len(pair_group),
                "auc_mean": auc_ci["mean"],
                "auc_ci_low": auc_ci["ci_low"],
                "auc_ci_high": auc_ci["ci_high"],
                "partial_corr_margin_mean": pcorr_ci["mean"],
                "partial_corr_margin_ci_low": pcorr_ci["ci_low"],
                "partial_corr_margin_ci_high": pcorr_ci["ci_high"],
                "mean_delta_O": delta_ci["mean"],
                "mean_delta_O_ci_low": delta_ci["ci_low"],
                "mean_delta_O_ci_high": delta_ci["ci_high"],
            }
        )
    return out


def margin_subset_summary(long_rows: list[dict], pair_rows: list[dict]) -> list[dict]:
    out = []
    for threshold in [4.0, 2.0, 1.0]:
        subset_pairs = [row for row in pair_rows if float(row.get("abs_margin_gap", "nan")) <= threshold]
        subset_ids = {row["id"] for row in subset_pairs}
        subset_long = [row for row in long_rows if row["id"] in subset_ids]
        out.extend(
            {
                **row,
                "margin_gap_threshold": threshold,
            }
            for row in evaluate_obstruction_metrics(subset_long, subset_pairs)
            if int(row["n_pairs"]) >= 20
        )
    return out


def relation_breakdown(long_rows: list[dict], pair_rows: list[dict]) -> list[dict]:
    out = []
    relations = sorted({row.get("relation", "") for row in pair_rows if row.get("relation", "")})
    for relation in relations:
        subset_pairs = [row for row in pair_rows if row.get("relation") == relation]
        ids = {row["id"] for row in subset_pairs}
        subset_long = [row for row in long_rows if row["id"] in ids]
        out.extend(
            {
                **row,
                "relation": relation,
            }
            for row in evaluate_obstruction_metrics(subset_long, subset_pairs)
        )
    return out


def control_comparison(metrics_rows: list[dict]) -> list[dict]:
    groups = group_rows(metrics_rows, ["model", "variant", "rank", "split"])
    out = []
    for key, rows in groups.items():
        learned = [row for row in rows if row.get("transport_control") == "learned"]
        if not learned:
            continue
        learned_row = learned[0]
        for row in rows:
            if row.get("transport_control") == "learned":
                continue
            out.append(
                {
                    "model": key[0],
                    "variant": key[1],
                    "rank": key[2],
                    "split": key[3],
                    "control": row.get("transport_control"),
                    "learned_auc": learned_row.get("auc_conflict_gt_clean"),
                    "control_auc": row.get("auc_conflict_gt_clean"),
                    "auc_lift": float(learned_row.get("auc_conflict_gt_clean", "nan")) - float(row.get("auc_conflict_gt_clean", "nan")),
                    "learned_partial_corr_margin": learned_row.get("partial_corr_margin"),
                    "control_partial_corr_margin": row.get("partial_corr_margin"),
                    "partial_corr_lift": float(learned_row.get("partial_corr_margin", "nan")) - float(row.get("partial_corr_margin", "nan")),
                }
            )
    return out


def edge_contribution_summary(long_rows: list[dict]) -> list[dict]:
    out = []
    groups = group_rows(long_rows, ["model", "variant", "rank", "transport_control", "split", "condition"])
    for key, rows in groups.items():
        sums: list[float] = []
        counts: list[int] = []
        for row in rows:
            terms = [float(value) for value in str(row.get("edge_terms", "")).split() if value]
            if not sums:
                sums = [0.0] * len(terms)
                counts = [0] * len(terms)
            for idx, value in enumerate(terms):
                if idx >= len(sums):
                    sums.append(0.0)
                    counts.append(0)
                sums[idx] += value
                counts[idx] += 1
        for edge_idx, total in enumerate(sums):
            count = counts[edge_idx]
            out.append(
                {
                    "model": key[0],
                    "variant": key[1],
                    "rank": key[2],
                    "transport_control": key[3],
                    "split": key[4],
                    "condition": key[5],
                    "edge": edge_idx,
                    "mean_edge_term": total / count if count else "",
                }
            )
    return out


def decision_rows(metrics_rows: list[dict], control_rows: list[dict], bootstrap_rows: list[dict]) -> list[dict]:
    control_lookup = {}
    for row in control_rows:
        key = (row["model"], row["variant"], row["rank"], row["split"])
        control_lookup.setdefault(key, []).append(row)
    bootstrap_lookup = {
        (row["model"], row["variant"], row["rank"], row["transport_control"], row["split"]): row
        for row in bootstrap_rows
    }
    out = []
    for row in metrics_rows:
        if row.get("transport_control") != "learned":
            continue
        n_pairs = int(float(row.get("n_pairs", 0)))
        auc = float(row.get("auc_conflict_gt_clean", "nan"))
        pcorr = float(row.get("partial_corr_margin", "nan"))
        key = (row["model"], row["variant"], row["rank"], row["split"])
        controls = control_lookup.get(key, [])
        beats_controls = all(float(item["auc_lift"]) > 0 and float(item["partial_corr_lift"]) >= 0 for item in controls)
        boot = bootstrap_lookup.get((row["model"], row["variant"], row["rank"], "learned", row["split"]), {})
        lower = float(boot.get("partial_corr_margin_ci_low", "nan"))
        if n_pairs >= 100 and auc > 0.65 and pcorr > 0.25 and beats_controls and lower > 0.10:
            decision = "PASS_STRONG"
        elif n_pairs >= 100 and auc > 0.65 and pcorr > 0.20:
            decision = "PASS_WEAK"
        else:
            decision = "NO_GO"
        out.append({**row, "beats_controls": beats_controls, "bootstrap_pcorr_low": lower, "decision": decision})
    return out


def main() -> None:
    args = parse_args()
    metrics_rows = read_csv(args.metrics)
    pair_rows = read_csv(args.pairs)
    long_rows = read_csv(args.long)
    output_dir = Path(args.output_dir)

    boot = bootstrap_summary(long_rows, pair_rows, args.n_bootstrap, args.seed)
    margin = margin_subset_summary(long_rows, pair_rows)
    relation = relation_breakdown(long_rows, pair_rows)
    controls = control_comparison(metrics_rows)
    edges = edge_contribution_summary(long_rows)
    decisions = decision_rows(metrics_rows, controls, boot)

    write_csv(output_dir / "bootstrap_summary.csv", boot)
    write_csv(output_dir / "margin_subset_summary.csv", margin)
    write_csv(output_dir / "relation_breakdown.csv", relation)
    write_csv(output_dir / "control_comparison.csv", controls)
    write_csv(output_dir / "edge_contribution_summary.csv", edges)
    write_csv(output_dir / "decision_summary.csv", decisions)

    for row in sorted(decisions, key=lambda item: item["decision"]):
        print(
            f"{row['decision']}: {row['split']} {row['variant']} r{row['rank']} "
            f"n={row['n_pairs']} auc={float(row['auc_conflict_gt_clean']):.4f} "
            f"pcorr={float(row['partial_corr_margin']):.4f}"
        )
    print("Do not run component patching unless PASS_STRONG or a deliberately accepted PASS_WEAK is obtained.")
    print(f"saved robustness outputs to {output_dir}")


if __name__ == "__main__":
    main()
