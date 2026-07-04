from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from src.factual_obstruction_eval import auc_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predictive factual response scores to baselines.")
    parser.add_argument("--joined", default="artifacts/factual_predictive_response/predictive_response_joined.csv")
    parser.add_argument("--output-metrics", default="artifacts/factual_predictive_response/predictive_response_metrics.csv")
    parser.add_argument("--output-repair-at-k", default="artifacts/factual_predictive_response/predictive_response_repair_at_k.csv")
    parser.add_argument("--output-top-sites", default="artifacts/factual_predictive_response/predictive_response_top_sites.csv")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--pareto-lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--kl-gammas", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
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


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman_rows(rows: list[dict], score_col: str, target_col: str) -> float:
    kept = [row for row in rows if row.get(score_col, "") != "" and row.get(target_col, "") != ""]
    xs = [float(row[score_col]) for row in kept]
    ys = [float(row[target_col]) for row in kept]
    rx = {v: i for i, v in enumerate(sorted(set(xs)), start=1)}
    ry = {v: i for i, v in enumerate(sorted(set(ys)), start=1)}
    return pearson([rx[v] for v in xs], [ry[v] for v in ys])


def base_rows(joined: list[dict]) -> list[dict]:
    out = {}
    for row in joined:
        key = (row["id"], row["site_id"])
        out.setdefault(key, dict(row))
    return list(out.values())


def add_targets(rows: list[dict], pareto_lambda: float = 0.25, kl_gamma: float = 0.0) -> None:
    for row in rows:
        kl = float(row.get("off_target_kl") or 0.0)
        row["pareto_score"] = (
            float(row["actual_delta_m"])
            + pareto_lambda * float(row["actual_obstruction_reduction"])
            - kl_gamma * kl
        )


def top_labels_by_example(rows: list[dict], target_col: str, frac: float = 0.10) -> dict[tuple[str, str], int]:
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        by_id.setdefault(row["id"], []).append(row)
    labels = {}
    for ex_id, group in by_id.items():
        sorted_vals = sorted(float(row[target_col]) for row in group)
        threshold = sorted_vals[max(0, int((1.0 - frac) * len(sorted_vals)))]
        for row in group:
            labels[(row["id"], row["site_id"])] = 1 if float(row[target_col]) >= threshold else 0
    return labels


def method_rows(joined: list[dict], seed: int) -> list[tuple[str, list[dict], str]]:
    base = base_rows(joined)
    rng = random.Random(seed)
    for row in base:
        row["random_score"] = rng.random()
    methods = [
        ("obstruction_only_score", base, "obstruction_only_score"),
        ("attribution_score", base, "attribution_score"),
        ("abs_attribution_score", base, "abs_attribution_score"),
        ("grad_saliency_score", base, "grad_saliency_score"),
        ("activation_norm_score", base, "activation_norm_score"),
        ("random_score", base, "random_score"),
        ("oracle_delta_m", base, "actual_delta_m"),
    ]
    response_groups: dict[tuple[str, str], list[dict]] = {}
    for row in joined:
        response_groups.setdefault((row["beta"], row["gamma"]), []).append(row)
    for (beta, gamma), rows in response_groups.items():
        methods.append((f"response_score_beta{beta}_gamma{gamma}", rows, "response_score"))
    return methods


def metrics_for_methods(joined: list[dict], pareto_lambda: float, kl_gamma: float, seed: int) -> list[dict]:
    add_targets(joined, pareto_lambda, kl_gamma)
    base = base_rows(joined)
    repair_labels = top_labels_by_example(base, "actual_delta_m")
    obstruction_labels = top_labels_by_example(base, "actual_obstruction_reduction")
    out = []
    for method, rows, score_col in method_rows(joined, seed):
        y_repair = [repair_labels[(row["id"], row["site_id"])] for row in rows]
        y_obstruction = [obstruction_labels[(row["id"], row["site_id"])] for row in rows]
        scores = [float(row[score_col]) for row in rows]
        out.append(
            {
                "method": method,
                "score_col": score_col,
                "pareto_lambda": pareto_lambda,
                "kl_gamma": kl_gamma,
                "n_rows": len(rows),
                "spearman_delta_m": spearman_rows(rows, score_col, "actual_delta_m"),
                "spearman_obstruction_reduction": spearman_rows(rows, score_col, "actual_obstruction_reduction"),
                "spearman_pareto": spearman_rows(rows, score_col, "pareto_score"),
                "auc_top_repair": auc_scores(y_repair, scores),
                "auc_top_obstruction_reduction": auc_scores(y_obstruction, scores),
            }
        )
    return out


def repair_at_k(joined: list[dict], ks: list[int], pareto_lambda: float, kl_gamma: float, seed: int) -> list[dict]:
    add_targets(joined, pareto_lambda, kl_gamma)
    out = []
    for method, rows, score_col in method_rows(joined, seed):
        by_id: dict[str, list[dict]] = {}
        for row in rows:
            by_id.setdefault(row["id"], []).append(row)
        for k in ks:
            selected = []
            for group in by_id.values():
                selected.extend(sorted(group, key=lambda row: float(row[score_col]), reverse=True)[: min(k, len(group))])
            out.append(
                {
                    "method": method,
                    "score_col": score_col,
                    "pareto_lambda": pareto_lambda,
                    "kl_gamma": kl_gamma,
                    "k": k,
                    "mean_actual_delta_m": mean([float(row["actual_delta_m"]) for row in selected]),
                    "mean_actual_obstruction_reduction": mean([float(row["actual_obstruction_reduction"]) for row in selected]),
                    "mean_pareto": mean([float(row["pareto_score"]) for row in selected]),
                    "margin_increase_success_rate": mean([
                        1.0 if str(row["margin_increase_success"]) == "True" else 0.0
                        for row in selected
                    ]),
                    "margin_flip_success_rate": mean([
                        1.0 if str(row["margin_flip_success"]) == "True" else 0.0
                        for row in selected
                    ]),
                    "mean_kl": mean([float(row.get("off_target_kl") or 0.0) for row in selected]),
                }
            )
    return out


def top_sites(joined: list[dict]) -> list[dict]:
    base = base_rows(joined)
    return sorted(
        base,
        key=lambda row: float(row["actual_delta_m"]),
        reverse=True,
    )[:100]


def main() -> None:
    args = parse_args()
    joined = read_csv(args.joined)
    metrics = []
    repairs = []
    for lam in args.pareto_lambdas:
        for gamma in args.kl_gammas:
            rows_copy = [dict(row) for row in joined]
            metrics.extend(metrics_for_methods(rows_copy, lam, gamma, args.seed))
            rows_copy = [dict(row) for row in joined]
            repairs.extend(repair_at_k(rows_copy, args.ks, lam, gamma, args.seed))
    write_csv(args.output_metrics, metrics)
    write_csv(args.output_repair_at_k, repairs)
    write_csv(args.output_top_sites, top_sites(joined))
    ranked = sorted(metrics, key=lambda row: float(row["spearman_pareto"]), reverse=True)
    print("Top predictive methods by Spearman with observed Pareto score")
    for row in ranked[:20]:
        print(
            f"{row['method']}: rho_pareto={float(row['spearman_pareto']):.4f} "
            f"auc_repair={float(row['auc_top_repair']):.4f} "
            f"auc_obstruction={float(row['auc_top_obstruction_reduction']):.4f}"
        )
    print(f"saved metrics to {args.output_metrics}")
    print(f"saved repair@k to {args.output_repair_at_k}")


if __name__ == "__main__":
    main()
