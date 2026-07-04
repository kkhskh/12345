from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from src.factual_obstruction_eval import auc_scores


PREDICTIVE_BASELINE_COLS = {
    "mean_activation_delta_norm",
    "mean_abs_pred_delta_margin",
    "mean_gradient_saliency",
    "random",
}

ORACLE_SUMMARY_COLS = {
    "repair_success_margin_increase",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 2C factual component patching.")
    parser.add_argument("--input", default="artifacts/factual_component_patching/component_patch_oracle.csv")
    parser.add_argument("--output-dir", default="artifacts/factual_component_patching")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing component patching CSV: {path}")
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


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


def spearman(rows: list[dict], x_col: str, y_col: str) -> float:
    kept = [row for row in rows if row.get(x_col, "") != "" and row.get(y_col, "") != ""]
    x = [float(row[x_col]) for row in kept]
    y = [float(row[y_col]) for row in kept]
    rx = {value: rank for rank, value in enumerate(sorted(set(x)), start=1)}
    ry = {value: rank for rank, value in enumerate(sorted(set(y)), start=1)}
    return pearson([rx[v] for v in x], [ry[v] for v in y])


def group_by_site(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["site_id"], []).append(row)
    out = []
    for site_id, group in groups.items():
        first = group[0]
        site_row = {
            "site_id": site_id,
            "kind": first.get("kind", ""),
            "layer": first.get("layer", ""),
            "head": first.get("head", ""),
            "model": first.get("model", ""),
            "variant": first.get("variant", ""),
            "rank": first.get("rank", ""),
            "n_examples": len(group),
            "mean_delta_margin": mean([float(row["delta_margin"]) for row in group]),
            "mean_obstruction_reduction": mean([float(row["obstruction_reduction"]) for row in group]),
            "mean_delta_O": mean([float(row["delta_O"]) for row in group]),
            "repair_success_margin_increase": mean([
                1.0 if str(row["repair_success_margin_increase"]) == "True" else 0.0
                for row in group
            ]),
            "repair_success_flip": mean([
                1.0 if str(row["repair_success_flip"]) == "True" else 0.0
                for row in group
            ]),
        }
        optional_means = {
            "activation_delta_norm": "mean_activation_delta_norm",
            "pred_delta_margin": "mean_pred_delta_margin",
            "abs_pred_delta_margin": "mean_abs_pred_delta_margin",
            "gradient_saliency": "mean_gradient_saliency",
        }
        for source_col, target_col in optional_means.items():
            values = [float(row[source_col]) for row in group if row.get(source_col, "") not in {"", None}]
            if values:
                site_row[target_col] = mean(values)
        out.append(site_row)
    return sorted(out, key=lambda row: float(row["mean_delta_margin"]), reverse=True)


def repair_at_k(site_rows: list[dict], score_col: str, ks=(1, 5, 10, 20, 50)) -> list[dict]:
    ranked = sorted(site_rows, key=lambda row: float(row[score_col]), reverse=True)
    out = []
    for k in ks:
        sub = ranked[: min(k, len(ranked))]
        out.append(
            {
                "method": score_col,
                "k": k,
                "mean_delta_margin": mean([float(row["mean_delta_margin"]) for row in sub]),
                "mean_obstruction_reduction": mean([float(row["mean_obstruction_reduction"]) for row in sub]),
                "repair_success_margin_increase": mean([
                    float(row["repair_success_margin_increase"]) for row in sub
                ]),
                "repair_success_flip": mean([float(row["repair_success_flip"]) for row in sub]),
            }
        )
    return out


def metric_rows(site_rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    for row in site_rows:
        row["random"] = rng.random()
    y = [1 if float(row["mean_delta_margin"]) >= sorted(float(r["mean_delta_margin"]) for r in site_rows)[max(0, int(0.9 * len(site_rows)) - 1)] else 0 for row in site_rows]
    rows = []
    candidate_cols = [
        "mean_obstruction_reduction",
        "mean_activation_delta_norm",
        "mean_abs_pred_delta_margin",
        "mean_gradient_saliency",
        "repair_success_margin_increase",
        "random",
    ]
    available_cols = [col for col in candidate_cols if all(row.get(col, "") != "" for row in site_rows)]
    for col in available_cols:
        rows.append(
            {
                "metric": f"spearman_{col}_with_delta_margin",
                "value": spearman(site_rows, col, "mean_delta_margin"),
            }
        )
        rows.append(
            {
                "metric": f"auc_{col}_detects_top_delta_margin_sites",
                "value": auc_scores(y, [float(row[col]) for row in site_rows]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    site_rows = group_by_site(rows)
    metrics = metric_rows(site_rows, args.seed)
    repair_rows = []
    candidate_cols = [
        "mean_obstruction_reduction",
        "mean_activation_delta_norm",
        "mean_abs_pred_delta_margin",
        "mean_gradient_saliency",
        "repair_success_margin_increase",
        "random",
    ]
    for score_col in [col for col in candidate_cols if all(row.get(col, "") != "" for row in site_rows)]:
        repair_rows.extend(repair_at_k(site_rows, score_col))

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "component_patch_site_summary.csv", site_rows)
    write_csv(output_dir / "component_patch_metric_summary.csv", metrics)
    write_csv(output_dir / "component_patch_repair_at_k.csv", repair_rows)

    print("Top component sites by observed delta margin")
    for row in site_rows[:20]:
        print(
            f"{row['site_id']}: delta_margin={float(row['mean_delta_margin']):.4f} "
            f"O_reduction={float(row['mean_obstruction_reduction']):.4f} "
            f"flip={float(row['repair_success_flip']):.3f}"
        )
    print("\nMetric summary")
    for row in metrics:
        print(f"{row['metric']}: {float(row['value']):.4f}")

    obstruction_auc = next(
        float(row["value"])
        for row in metrics
        if row["metric"] == "auc_mean_obstruction_reduction_detects_top_delta_margin_sites"
    )
    obstruction_rho = next(
        float(row["value"])
        for row in metrics
        if row["metric"] == "spearman_mean_obstruction_reduction_with_delta_margin"
    )
    if obstruction_auc > 0.65 and obstruction_rho > 0.30:
        print("PASS_WEAK: obstruction reduction is associated with causal patch repair.")
    else:
        print("NO_GO: obstruction reduction does not yet predict component repair strongly.")
    baseline_metrics = [
        row for row in metrics
        if row["metric"].startswith("auc_")
        and row["metric"] != "auc_mean_obstruction_reduction_detects_top_delta_margin_sites"
    ]
    if baseline_metrics:
        predictive_baseline_metrics = [
            row for row in baseline_metrics
            if any(f"auc_{col}_" in row["metric"] for col in PREDICTIVE_BASELINE_COLS)
        ]
        oracle_summary_metrics = [
            row for row in baseline_metrics
            if any(f"auc_{col}_" in row["metric"] for col in ORACLE_SUMMARY_COLS)
        ]
        if predictive_baseline_metrics:
            best_predictive = max(float(row["value"]) for row in predictive_baseline_metrics)
            print(f"best predictive baseline AUC: {best_predictive:.4f}")
        else:
            best_predictive = float("nan")
        if oracle_summary_metrics:
            best_oracle_summary = max(float(row["value"]) for row in oracle_summary_metrics)
            print(f"best outcome-derived summary AUC: {best_oracle_summary:.4f}")
        if predictive_baseline_metrics and obstruction_auc > best_predictive:
            print("obstruction reduction beats available predictive baselines on top-site AUC.")
        else:
            print("obstruction reduction does not beat the best available predictive baseline on top-site AUC.")
    print(f"saved summaries to {output_dir}")


if __name__ == "__main__":
    main()
