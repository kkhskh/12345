"""Out-of-sample validation of the standardized obstruction-response score.

The updated theory requires obstruction to be tested as an additive signal rather than a
scale artifact. This script selects score hyperparameters on a tuning half of the examples
and reports metrics on a held-out half, then measures the paired obstruction lift: how much
the best obstruction-using score beats the best obstruction-free score on held-out examples.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.factual_obstruction_eval import auc_scores

from run_06c_factual_response_baselines import (
    add_standardized_scores,
    add_targets,
    base_rows,
    mean,
    read_csv,
    spearman_rows,
    top_labels_by_example,
    uses_obstruction,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select standardized response score on a tuning split and validate out of sample."
    )
    parser.add_argument("--joined", default="artifacts/factual_predictive_response/predictive_response_joined.csv")
    parser.add_argument("--output-tuning", default="artifacts/factual_predictive_response/standardized_tuning_metrics.csv")
    parser.add_argument("--output-test", default="artifacts/factual_predictive_response/standardized_test_metrics.csv")
    parser.add_argument("--output-lift", default="artifacts/factual_predictive_response/standardized_obstruction_lift.csv")
    parser.add_argument("--summary", default="artifacts/factual_predictive_response/standardized_decision_summary.md")
    parser.add_argument("--pareto-lambda", type=float, default=0.25)
    parser.add_argument("--kl-gamma", type=float, default=0.0)
    parser.add_argument("--tune-frac", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def split_example_ids(rows: list[dict], tune_frac: float, seed: int) -> tuple[set[str], set[str]]:
    ids = sorted({row["id"] for row in rows})
    random.Random(seed).shuffle(ids)
    n_tune = max(1, int(round(tune_frac * len(ids))))
    if n_tune >= len(ids):
        n_tune = len(ids) - 1
    return set(ids[:n_tune]), set(ids[n_tune:])


def build_methods(joined: list[dict], seed: int) -> list[tuple[str, list[dict], str, bool]]:
    """Every candidate scoring rule, tagged by whether it uses the obstruction term."""
    base = base_rows(joined)
    rng = random.Random(seed)
    for row in base:
        row["random_score"] = rng.random()
    methods: list[tuple[str, list[dict], str, bool]] = [
        ("attribution_score", base, "attribution_score", False),
        ("abs_attribution_score", base, "abs_attribution_score", False),
        ("grad_saliency_score", base, "grad_saliency_score", False),
        ("activation_norm_score", base, "activation_norm_score", False),
        ("random_score", base, "random_score", False),
        ("obstruction_only_score", base, "obstruction_only_score", True),
    ]
    for method, score_col in add_standardized_scores(base):
        methods.append((method, base, score_col, uses_obstruction(method)))
    raw_groups: dict[tuple[str, str], list[dict]] = {}
    for row in joined:
        raw_groups.setdefault((row["beta"], row["gamma"]), []).append(row)
    for (beta, gamma), rows in sorted(raw_groups.items()):
        methods.append((f"raw_response_beta{beta}_gamma{gamma}", rows, "response_score", True))
    return methods


def subset(rows: list[dict], keep_ids: set[str]) -> list[dict]:
    return [row for row in rows if row["id"] in keep_ids]


def eval_method(rows: list[dict], score_col: str, pareto_lambda: float, kl_gamma: float) -> dict:
    if not rows:
        return {
            "n_rows": 0,
            "spearman_delta_m": float("nan"),
            "spearman_pareto": float("nan"),
            "auc_top_repair": float("nan"),
            "auc_top_obstruction_reduction": float("nan"),
        }
    add_targets(rows, pareto_lambda, kl_gamma)
    repair_labels = top_labels_by_example(rows, "actual_delta_m")
    obstruction_labels = top_labels_by_example(rows, "actual_obstruction_reduction")
    scores = [float(row[score_col]) for row in rows]
    return {
        "n_rows": len(rows),
        "spearman_delta_m": spearman_rows(rows, score_col, "actual_delta_m"),
        "spearman_pareto": spearman_rows(rows, score_col, "pareto_score"),
        "auc_top_repair": auc_scores([repair_labels[(r["id"], r["site_id"])] for r in rows], scores),
        "auc_top_obstruction_reduction": auc_scores(
            [obstruction_labels[(r["id"], r["site_id"])] for r in rows], scores
        ),
    }


def evaluate_split(
    methods: list[tuple[str, list[dict], str, bool]],
    keep_ids: set[str],
    split_name: str,
    pareto_lambda: float,
    kl_gamma: float,
) -> list[dict]:
    out = []
    for method, rows, score_col, with_obstruction in methods:
        metrics = eval_method(subset(rows, keep_ids), score_col, pareto_lambda, kl_gamma)
        out.append(
            {
                "split": split_name,
                "method": method,
                "score_col": score_col,
                "uses_obstruction": with_obstruction,
                **metrics,
            }
        )
    return out


def pick_best(metric_rows: list[dict], with_obstruction: bool, criterion: str = "spearman_pareto") -> dict | None:
    candidates = [
        row
        for row in metric_rows
        if row["uses_obstruction"] is with_obstruction
        and row["method"] != "random_score"
        and float(row[criterion]) == float(row[criterion])  # drop NaN
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[criterion]))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
    return values[idx]


def paired_lift_bootstrap(
    rows_with: list[dict],
    col_with: str,
    rows_without: list[dict],
    col_without: str,
    test_ids: set[str],
    n_bootstrap: int,
    pareto_lambda: float,
    kl_gamma: float,
    seed: int,
) -> list[dict]:
    """Resample test examples once per iteration and score both methods on the same resample."""
    grouped_with: dict[str, list[dict]] = {}
    for row in subset(rows_with, test_ids):
        grouped_with.setdefault(row["id"], []).append(row)
    grouped_without: dict[str, list[dict]] = {}
    for row in subset(rows_without, test_ids):
        grouped_without.setdefault(row["id"], []).append(row)
    ids = sorted(set(grouped_with) & set(grouped_without))
    rng = random.Random(seed)
    lifts: dict[str, list[float]] = {"spearman_pareto": [], "auc_top_repair": []}
    for _ in range(n_bootstrap):
        sampled = [ids[rng.randrange(len(ids))] for _ in ids]
        sample_with: list[dict] = []
        sample_without: list[dict] = []
        for ex_id in sampled:
            sample_with.extend(grouped_with[ex_id])
            sample_without.extend(grouped_without[ex_id])
        m_with = eval_method(sample_with, col_with, pareto_lambda, kl_gamma)
        m_without = eval_method(sample_without, col_without, pareto_lambda, kl_gamma)
        for metric in lifts:
            a, b = float(m_with[metric]), float(m_without[metric])
            if a == a and b == b:
                lifts[metric].append(a - b)
    out = []
    for metric, vals in lifts.items():
        positive = mean([1.0 if v > 0 else 0.0 for v in vals])
        out.append(
            {
                "metric": metric,
                "mean_lift": mean(vals),
                "ci_low": percentile(vals, 0.025),
                "ci_high": percentile(vals, 0.975),
                "frac_positive": positive,
                "n_bootstrap": len(vals),
            }
        )
    return out


def decide(lift_rows: list[dict], best_test_with: dict | None) -> str:
    if not best_test_with or not lift_rows:
        return "NO_GO"
    by_metric = {row["metric"]: row for row in lift_rows}
    pareto = by_metric.get("spearman_pareto")
    repair = by_metric.get("auc_top_repair")
    if not pareto or not repair:
        return "NO_GO"
    auc = float(best_test_with["auc_top_repair"])
    strong = (
        float(pareto["ci_low"]) > 0.0
        and float(repair["mean_lift"]) > 0.0
        and auc > 0.70
    )
    weak = float(pareto["mean_lift"]) > 0.0 and float(pareto["frac_positive"]) > 0.90 and auc > 0.65
    if strong:
        return "PASS_STRONG"
    if weak:
        return "PASS_WEAK"
    return "NO_GO"


def write_summary(
    path: str | Path,
    tune_best_with: dict | None,
    tune_best_without: dict | None,
    test_with: dict | None,
    test_without: dict | None,
    lift_rows: list[dict],
    decision: str,
    n_tune: int,
    n_test: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 2D Standardized Response Score: Out-of-Sample Validation",
        "",
        "Standardized score: `S_z = z(pred_delta_m) - alpha * z(pred_delta_O) - gamma * z(cost)`,",
        "with z-scores taken across candidate sites within each example.",
        "",
        f"- tuning examples: {n_tune}",
        f"- held-out examples: {n_test}",
        "",
        "## Selected on Tuning Split",
        "",
    ]
    for label, row in [("with obstruction", tune_best_with), ("without obstruction", tune_best_without)]:
        if row:
            lines.append(
                f"- {label}: `{row['method']}` "
                f"(tune Spearman Pareto {float(row['spearman_pareto']):.4f})"
            )
    lines.extend(["", "## Held-Out Performance", ""])
    for label, row in [("with obstruction", test_with), ("without obstruction", test_without)]:
        if row:
            lines.append(
                f"- {label}: `{row['method']}` Spearman Pareto {float(row['spearman_pareto']):.4f}, "
                f"AUC repair {float(row['auc_top_repair']):.4f}, "
                f"Spearman delta_m {float(row['spearman_delta_m']):.4f}"
            )
    lines.extend(["", "## Paired Obstruction Lift on Held-Out Examples", ""])
    for row in lift_rows:
        lines.append(
            f"- {row['metric']}: mean lift {float(row['mean_lift']):+.4f} "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}], "
            f"positive in {float(row['frac_positive']):.1%} of resamples"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision: **{decision}**",
            "",
            "The obstruction term survives only if the lift over the best obstruction-free score is",
            "positive out of sample. A negative or zero lift means the score reduces to attribution.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    joined = read_csv(args.joined)
    tune_ids, test_ids = split_example_ids(joined, args.tune_frac, args.seed)
    methods = build_methods(joined, args.seed)

    tune_metrics = evaluate_split(methods, tune_ids, "tune", args.pareto_lambda, args.kl_gamma)
    test_metrics = evaluate_split(methods, test_ids, "test", args.pareto_lambda, args.kl_gamma)
    write_csv(args.output_tuning, tune_metrics)
    write_csv(args.output_test, test_metrics)

    tune_best_with = pick_best(tune_metrics, with_obstruction=True)
    tune_best_without = pick_best(tune_metrics, with_obstruction=False)
    test_by_method = {row["method"]: row for row in test_metrics}
    test_with = test_by_method.get(tune_best_with["method"]) if tune_best_with else None
    test_without = test_by_method.get(tune_best_without["method"]) if tune_best_without else None

    lift_rows: list[dict] = []
    if tune_best_with and tune_best_without:
        lookup = {name: (rows, col) for name, rows, col, _ in methods}
        rows_with, col_with = lookup[tune_best_with["method"]]
        rows_without, col_without = lookup[tune_best_without["method"]]
        lift_rows = paired_lift_bootstrap(
            rows_with,
            col_with,
            rows_without,
            col_without,
            test_ids,
            args.n_bootstrap,
            args.pareto_lambda,
            args.kl_gamma,
            args.seed,
        )
        for row in lift_rows:
            row["method_with_obstruction"] = tune_best_with["method"]
            row["method_without_obstruction"] = tune_best_without["method"]
        write_csv(args.output_lift, lift_rows)

    decision = decide(lift_rows, test_with)
    write_summary(
        args.summary,
        tune_best_with,
        tune_best_without,
        test_with,
        test_without,
        lift_rows,
        decision,
        len(tune_ids),
        len(test_ids),
    )

    if tune_best_with:
        print(f"selected with obstruction:    {tune_best_with['method']}")
    if tune_best_without:
        print(f"selected without obstruction: {tune_best_without['method']}")
    for row in lift_rows:
        print(
            f"held-out lift {row['metric']}: {float(row['mean_lift']):+.4f} "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"
        )
    print(f"decision: {decision}")
    print(f"saved summary to {args.summary}")


if __name__ == "__main__":
    main()
