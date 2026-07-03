from __future__ import annotations

import math
import random
from typing import Callable


def _values(rows, col):
    return [float(row[col]) for row in rows if row.get(col, "") not in {"", None}]


def auc_scores(y_true: list[int], scores: list[float]) -> float:
    """AUC via rank statistic; avoids a hard sklearn dependency."""

    pairs = sorted(zip(scores, y_true), key=lambda item: item[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += sum(label for _, label in pairs[idx:end]) * avg_rank
        idx = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_conflict_greater_clean(pair_rows: list[dict], obstruction_col: str = "delta_O") -> float:
    """Pairwise AUC-like probability that conflict obstruction exceeds clean."""

    values = [float(row[obstruction_col]) for row in pair_rows if row.get(obstruction_col, "") != ""]
    if not values:
        return float("nan")
    return sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in values) / len(values)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    mx, my = _mean(x), _mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def _residualize(y: list[float], controls: list[list[float]]) -> list[float]:
    if not controls:
        return [v - _mean(y) for v in y]
    try:
        import numpy as np
    except ImportError:
        # Fallback: remove only the mean if numpy is unavailable.
        return [v - _mean(y) for v in y]
    X = np.asarray([[1.0] + list(row) for row in zip(*controls)], dtype=float)
    y_arr = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    return (y_arr - X @ beta).tolist()


def partial_corr(rows: list[dict], x_col: str, y_col: str, control_cols: list[str]) -> float:
    """Partial correlation by linear residualization."""

    kept = [
        row for row in rows
        if row.get(x_col, "") != "" and row.get(y_col, "") != ""
        and all(row.get(col, "") != "" for col in control_cols)
    ]
    if len(kept) < 4:
        return float("nan")
    x = [float(row[x_col]) for row in kept]
    y = [float(row[y_col]) for row in kept]
    controls = [[float(row[col]) for row in kept] for col in control_cols]
    return _pearson(_residualize(x, controls), _residualize(y, controls))


def paired_delta_summary(pair_rows: list[dict]) -> dict:
    deltas = [float(row["delta_O"]) for row in pair_rows if row.get("delta_O", "") != ""]
    if not deltas:
        return {"paired_mean_delta_O": float("nan"), "paired_median_delta_O": float("nan")}
    sorted_deltas = sorted(deltas)
    mid = len(sorted_deltas) // 2
    median = (
        sorted_deltas[mid]
        if len(sorted_deltas) % 2
        else (sorted_deltas[mid - 1] + sorted_deltas[mid]) / 2
    )
    return {"paired_mean_delta_O": _mean(deltas), "paired_median_delta_O": median}


def spearman_corr(rows: list[dict], x_col: str, y_col: str) -> float:
    kept = [row for row in rows if row.get(x_col, "") != "" and row.get(y_col, "") != ""]
    if len(kept) < 3:
        return float("nan")
    x = [float(row[x_col]) for row in kept]
    y = [float(row[y_col]) for row in kept]
    rx = {value: rank for rank, value in enumerate(sorted(set(x)), start=1)}
    ry = {value: rank for rank, value in enumerate(sorted(set(y)), start=1)}
    return _pearson([rx[v] for v in x], [ry[v] for v in y])


def bootstrap_metric_ci(
    rows: list[dict],
    metric_fn: Callable[[list[dict]], float],
    n_bootstrap: int = 1000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap percentile CI for a row-level metric."""

    if not rows:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = metric_fn(sample)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    values.sort()
    low_idx = max(0, int((alpha / 2) * len(values)) - 1)
    high_idx = min(len(values) - 1, int((1 - alpha / 2) * len(values)) - 1)
    return {"mean": _mean(values), "ci_low": values[low_idx], "ci_high": values[high_idx]}


def evaluate_obstruction_metrics(
    long_rows: list[dict],
    pair_rows: list[dict],
    diagnostics_rows: list[dict] | None = None,
) -> list[dict]:
    """Compute metrics by model/variant/rank/control/split."""

    diagnostics_rows = diagnostics_rows or []
    diag_lookup = {}
    for row in diagnostics_rows:
        key = (row.get("model"), row.get("variant"), str(row.get("rank")))
        diag_lookup.setdefault(key, []).append(row)

    groups = {}
    for row in long_rows:
        key = (
            row.get("model", ""),
            row.get("variant", ""),
            str(row.get("rank", "")),
            row.get("transport_control", ""),
            row.get("split", ""),
        )
        groups.setdefault(key, {"long": [], "pairs": []})["long"].append(row)
    for row in pair_rows:
        key = (
            row.get("model", ""),
            row.get("variant", ""),
            str(row.get("rank", "")),
            row.get("transport_control", ""),
            row.get("split", ""),
        )
        groups.setdefault(key, {"long": [], "pairs": []})["pairs"].append(row)

    out = []
    for key, group in groups.items():
        model, variant, rank, control, split = key
        long_group = group["long"]
        pair_group = group["pairs"]
        y = [int(float(row["label_conflict"])) for row in long_group]
        scores = [float(row["obstruction_normalized"]) for row in long_group]
        diag = diag_lookup.get((model, variant, rank), [])
        val_r2 = [float(row["val_r2"]) for row in diag if row.get("val_r2", "") != ""]
        weights = [float(row["weight"]) for row in diag if row.get("weight", "") != ""]
        row = {
            "model": model,
            "variant": variant,
            "rank": rank,
            "transport_control": control,
            "split": split,
            "n_pairs": len(pair_group),
            "auc_conflict_gt_clean": auc_scores(y, scores),
            "partial_corr_margin": partial_corr(
                long_group, "obstruction_normalized", "label_conflict", ["margin"]
            ),
            "partial_corr_abs_margin": partial_corr(
                long_group, "obstruction_normalized", "label_conflict", ["abs_margin"]
            ),
            "partial_corr_margin_answer_lengths": partial_corr(
                long_group,
                "obstruction_normalized",
                "label_conflict",
                ["margin", "true_answer_token_len", "false_answer_token_len"],
            ),
            "spearman_O_margin": spearman_corr(long_group, "obstruction_normalized", "margin"),
            "learned_transport_mean_val_r2": _mean(val_r2),
            "active_edges": sum(1 for value in weights if value > 0),
        }
        row.update(paired_delta_summary(pair_group))
        out.append(row)
    return out


def make_summary_tables(long_rows: list[dict], pair_rows: list[dict], diagnostics_rows: list[dict] | None = None):
    """Compatibility wrapper for script use."""

    return evaluate_obstruction_metrics(long_rows, pair_rows, diagnostics_rows)
