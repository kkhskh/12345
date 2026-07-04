from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from run_06c_factual_response_baselines import metrics_for_methods, read_csv, repair_at_k, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap predictive factual response stability.")
    parser.add_argument("--joined", default="artifacts/factual_predictive_response/predictive_response_joined.csv")
    parser.add_argument("--metrics", default="artifacts/factual_predictive_response/predictive_response_metrics.csv")
    parser.add_argument("--repair-at-k", default="artifacts/factual_predictive_response/predictive_response_repair_at_k.csv")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="artifacts/factual_predictive_response/predictive_response_bootstrap.csv")
    parser.add_argument("--summary", default="artifacts/factual_predictive_response/predictive_response_decision_summary.md")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
    return values[idx]


def group_by_example(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row)
    return grouped


def bootstrap(rows: list[dict], n_bootstrap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped = group_by_example(rows)
    ids = sorted(grouped)
    values: dict[tuple[str, str], list[float]] = {}
    for _ in range(n_bootstrap):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        sample = []
        for ex_id in sampled_ids:
            sample.extend(dict(row) for row in grouped[ex_id])
        metric_rows = metrics_for_methods(sample, pareto_lambda=0.25, kl_gamma=0.0, seed=seed)
        repair_rows = repair_at_k(sample, ks=[1, 3, 5, 10], pareto_lambda=0.25, kl_gamma=0.0, seed=seed)
        for row in metric_rows:
            for metric in [
                "spearman_delta_m",
                "spearman_obstruction_reduction",
                "spearman_pareto",
                "auc_top_repair",
                "auc_top_obstruction_reduction",
            ]:
                values.setdefault((row["method"], metric), []).append(float(row[metric]))
        for row in repair_rows:
            if int(row["k"]) in {1, 3, 5, 10}:
                for metric in ["mean_actual_delta_m", "mean_actual_obstruction_reduction", "mean_pareto"]:
                    values.setdefault((f"{row['method']}@{row['k']}", metric), []).append(float(row[metric]))

    out = []
    for (method, metric), vals in values.items():
        out.append(
            {
                "method": method,
                "metric": metric,
                "mean": sum(vals) / len(vals),
                "ci_low": percentile(vals, 0.025),
                "ci_high": percentile(vals, 0.975),
                "n_bootstrap": len(vals),
            }
        )
    return out


def best_response(metrics_rows: list[dict]) -> dict | None:
    response = [row for row in metrics_rows if row["method"].startswith("response_score")]
    if not response:
        return None
    return max(response, key=lambda row: float(row["spearman_pareto"]))


def write_summary(path: str | Path, metrics_rows: list[dict], bootstrap_rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = best_response(metrics_rows)
    attribution = next((row for row in metrics_rows if row["method"] == "attribution_score"), None)
    abs_attr = next((row for row in metrics_rows if row["method"] == "abs_attribution_score"), None)
    obstruction = next((row for row in metrics_rows if row["method"] == "obstruction_only_score"), None)
    lines = ["# Experiment 2D Predictive Response Decision Summary", ""]
    if best:
        lines.extend(
            [
                "## Best Response Score",
                "",
                f"- method: `{best['method']}`",
                f"- Spearman with Pareto: {float(best['spearman_pareto']):.4f}",
                f"- AUC top repair: {float(best['auc_top_repair']):.4f}",
                f"- AUC top obstruction reduction: {float(best['auc_top_obstruction_reduction']):.4f}",
                "",
            ]
        )
    lines.append("## Baselines")
    for label, row in [
        ("attribution", attribution),
        ("absolute attribution", abs_attr),
        ("obstruction only", obstruction),
    ]:
        if row:
            lines.append(
                f"- {label}: Spearman Pareto {float(row['spearman_pareto']):.4f}, "
                f"AUC repair {float(row['auc_top_repair']):.4f}"
            )
    lines.extend(["", "## Decision", ""])
    if best and attribution:
        best_p = float(best["spearman_pareto"])
        best_auc = float(best["auc_top_repair"])
        attr_p = float(attribution["spearman_pareto"])
        if best_p > 0.35 and best_auc > 0.75 and best_p > attr_p * 1.05:
            decision = "PASS_STRONG"
        elif best_p > 0.25 and best_auc > 0.65:
            decision = "PASS_WEAK"
        else:
            decision = "NO_GO"
    else:
        decision = "NO_GO"
    lines.append(f"Decision: **{decision}**")
    lines.append("")
    lines.append("Experiment 2D computes predictive scores before patching; patch outcomes are used only for evaluation.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    joined = read_csv(args.joined)
    metrics = read_csv(args.metrics) if Path(args.metrics).exists() else metrics_for_methods(
        [dict(row) for row in joined], pareto_lambda=0.25, kl_gamma=0.0, seed=args.seed
    )
    boot = bootstrap(joined, args.n_bootstrap, args.seed)
    write_csv(args.output, boot)
    write_summary(args.summary, metrics, boot)
    print(f"saved bootstrap to {args.output}")
    print(f"saved decision summary to {args.summary}")


if __name__ == "__main__":
    main()
