from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.data_ioi import read_jsonl
from src.model import load_gpt2_small

from run_02g_component_level_obstruction import (
    SEED,
    TEST_PATH,
    build_sites,
    compute_component_scores_for_example,
    names_filter,
    patch_site_batch,
)


ALPHAS = [-1, -0.5, 0, 0.5, 1, 2]
GAMMAS = [0, 0.01, 0.1, 0.5, 1]
BETAS = [-1, -0.5, 0.5, 1, 2]
KS = [1, 3, 5]

VAL_SCORES_PATH = Path("artifacts/results/component_standardized_val_scores.csv")
VAL_ORACLE_PATH = Path("artifacts/results/component_standardized_val_oracle.csv")
TEST_SCORES_PATH = Path("artifacts/results/component_standardized_test_scores.csv")
TEST_ORACLE_PATH = Path("artifacts/results/component_standardized_test_oracle.csv")
TUNING_PATH = Path("artifacts/results/component_standardized_validation_tuning.csv")
TEST_METRICS_PATH = Path("artifacts/results/component_standardized_test_metrics.csv")
OVERLAP_PATH = Path("artifacts/results/component_standardized_overlap_metrics.csv")
TOP_SITES_PATH = Path("artifacts/results/component_standardized_top_sites.csv")
FIG_REPAIR_PATH = Path("artifacts/figures/component_standardized_repair_at_k.png")
FIG_SCATTER_PATH = Path("artifacts/figures/component_standardized_score_vs_observed.png")


def zscore_by_example(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("id")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[col] - mean) / std).fillna(0.0)


def add_standardized_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["z_attr"] = zscore_by_example(out, "pred_delta_m")
    out["z_obst"] = zscore_by_example(out, "pred_delta_O")
    out["z_cost"] = zscore_by_example(out, "cost")
    out["z_norm"] = zscore_by_example(out, "activation_delta_norm")
    return out


def compute_oracle_split(
    model,
    examples,
    site_list,
    scores_path: Path,
    oracle_path: Path,
    batch_size: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores_path.exists() and oracle_path.exists() and not force:
        return pd.read_csv(scores_path), pd.read_csv(oracle_path)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    cache_names = names_filter(model)

    score_rows = []
    for ex in tqdm(examples, desc=f"scores {scores_path.stem}"):
        score_rows.extend(compute_component_scores_for_example(model, ex, site_list))
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(scores_path, index=False)

    oracle_rows = []
    for site in tqdm(site_list, desc=f"patch oracle {oracle_path.stem}"):
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            oracle_rows.extend(patch_site_batch(model, batch, site, cache_names))
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(oracle_path, index=False)

    return score_df, oracle_df


def joined_example_site(score_df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    df = score_df.merge(
        oracle_df[
            [
                "id",
                "site_id",
                "observed_delta_margin",
                "repair_success_margin_increase",
                "repair_success_flip",
            ]
        ],
        on=["id", "site_id"],
        how="inner",
    )
    return add_standardized_columns(df)


def add_method_scores(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    out = df.copy()
    out["attribution_only"] = out["z_attr"]
    out["norm_only"] = out["z_norm"]
    out["random"] = [rng.random() for _ in range(len(out))]
    return out


def add_tuned_scores(
    df: pd.DataFrame,
    best_cost_gamma: float,
    best_norm_beta: float,
    best_alpha: float,
    best_gamma: float,
) -> pd.DataFrame:
    out = add_method_scores(df, random.Random(SEED))
    out["attribution_plus_cost"] = out["z_attr"] - best_cost_gamma * out["z_cost"]
    out["attribution_plus_norm"] = out["z_attr"] + best_norm_beta * out["z_norm"]
    out["obstruction_response"] = (
        out["z_attr"] - best_alpha * out["z_obst"] - best_gamma * out["z_cost"]
    )
    return out


def safe_spearman(x, y) -> float:
    rho, _ = spearmanr(x, y, nan_policy="omit")
    return float(rho)


def top_causal_auc(df: pd.DataFrame, score_col: str) -> float:
    threshold = df["observed_delta_margin"].quantile(0.90)
    y = (df["observed_delta_margin"] >= threshold).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, df[score_col]))


def rank_metrics_by_example(df: pd.DataFrame, score_col: str, method: str) -> list[dict]:
    rows = []
    for k in KS:
        per_example = []
        for _, sub in df.groupby("id"):
            top = sub.sort_values(score_col, ascending=False).head(k)
            per_example.append(
                {
                    "mean_delta_margin": top["observed_delta_margin"].mean(),
                    "repair_success_margin_increase": top[
                        "repair_success_margin_increase"
                    ].mean(),
                    "repair_success_flip": top["repair_success_flip"].mean(),
                }
            )
        tmp = pd.DataFrame(per_example)
        rows.append(
            {
                "method": method,
                "k": k,
                "mean_delta_margin_at_k": tmp["mean_delta_margin"].mean(),
                "repair_success_margin_increase_at_k": tmp[
                    "repair_success_margin_increase"
                ].mean(),
                "repair_success_flip_at_k": tmp["repair_success_flip"].mean(),
            }
        )
    return rows


def evaluate_methods(df: pd.DataFrame, split: str) -> pd.DataFrame:
    method_cols = {
        "attribution_only": "attribution_only",
        "attribution_plus_cost": "attribution_plus_cost",
        "attribution_plus_norm": "attribution_plus_norm",
        "obstruction_response": "obstruction_response",
        "norm_only": "norm_only",
        "random": "random",
    }
    rows = []
    for method, col in method_cols.items():
        rows.append(
            {
                "split": split,
                "method": method,
                "metric": "spearman",
                "value": safe_spearman(df[col], df["observed_delta_margin"]),
            }
        )
        rows.append(
            {
                "split": split,
                "method": method,
                "metric": "top_causal_auc",
                "value": top_causal_auc(df, col),
            }
        )
        for row in rank_metrics_by_example(df, col, method):
            for metric in [
                "mean_delta_margin_at_k",
                "repair_success_margin_increase_at_k",
                "repair_success_flip_at_k",
            ]:
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "k": row["k"],
                        "value": row[metric],
                    }
                )
    return pd.DataFrame(rows)


def tune_hyperparameters(val_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []

    for gamma in GAMMAS:
        score = val_df["z_attr"] - gamma * val_df["z_cost"]
        rows.append(
            {
                "method": "attribution_plus_cost",
                "alpha": 0.0,
                "gamma": gamma,
                "beta": np.nan,
                "spearman": safe_spearman(score, val_df["observed_delta_margin"]),
            }
        )

    for beta in BETAS:
        score = val_df["z_attr"] + beta * val_df["z_norm"]
        rows.append(
            {
                "method": "attribution_plus_norm",
                "alpha": np.nan,
                "gamma": np.nan,
                "beta": beta,
                "spearman": safe_spearman(score, val_df["observed_delta_margin"]),
            }
        )

    for alpha in ALPHAS:
        for gamma in GAMMAS:
            score = val_df["z_attr"] - alpha * val_df["z_obst"] - gamma * val_df["z_cost"]
            rows.append(
                {
                    "method": "obstruction_response",
                    "alpha": alpha,
                    "gamma": gamma,
                    "beta": np.nan,
                    "spearman": safe_spearman(score, val_df["observed_delta_margin"]),
                }
            )

    tuning = pd.DataFrame(rows)
    best_cost = tuning[tuning["method"] == "attribution_plus_cost"].sort_values(
        "spearman",
        ascending=False,
    ).iloc[0]
    best_norm = tuning[tuning["method"] == "attribution_plus_norm"].sort_values(
        "spearman",
        ascending=False,
    ).iloc[0]
    best_obst = tuning[tuning["method"] == "obstruction_response"].sort_values(
        "spearman",
        ascending=False,
    ).iloc[0]

    best = {
        "cost_gamma": float(best_cost["gamma"]),
        "norm_beta": float(best_norm["beta"]),
        "alpha": float(best_obst["alpha"]),
        "gamma": float(best_obst["gamma"]),
    }
    return tuning, best


def site_level_overlap_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    site = (
        test_df.groupby("site_id")
        .agg(
            observed_delta_margin=("observed_delta_margin", "mean"),
            attribution_only=("attribution_only", "mean"),
            attribution_plus_cost=("attribution_plus_cost", "mean"),
            attribution_plus_norm=("attribution_plus_norm", "mean"),
            obstruction_response=("obstruction_response", "mean"),
            norm_only=("norm_only", "mean"),
            random=("random", "mean"),
        )
        .reset_index()
    )
    oracle_ranked = site.sort_values("observed_delta_margin", ascending=False)
    oracle_sets = {k: set(oracle_ranked.head(k)["site_id"]) for k in [10, 25]}
    top_oracle_site = oracle_ranked.iloc[0]["site_id"]

    rows = []
    for method in [
        "attribution_only",
        "attribution_plus_cost",
        "attribution_plus_norm",
        "obstruction_response",
        "norm_only",
        "random",
    ]:
        ranked = site.sort_values(method, ascending=False).reset_index(drop=True)
        method_sets = {k: set(ranked.head(k)["site_id"]) for k in [10, 25]}
        rank_lookup = {site_id: idx + 1 for idx, site_id in enumerate(ranked["site_id"])}
        rho = safe_spearman(site[method], site["observed_delta_margin"])
        rows.append(
            {
                "method": method,
                "overlap_at_10": len(oracle_sets[10] & method_sets[10]),
                "jaccard_at_10": len(oracle_sets[10] & method_sets[10])
                / len(oracle_sets[10] | method_sets[10]),
                "overlap_at_25": len(oracle_sets[25] & method_sets[25]),
                "jaccard_at_25": len(oracle_sets[25] & method_sets[25])
                / len(oracle_sets[25] | method_sets[25]),
                "rank_of_top_oracle_site": rank_lookup[top_oracle_site],
                "spearman_with_oracle_site_effect": rho,
            }
        )

    rows.append(
        {
            "method": "obstruction_vs_attribution",
            "spearman_score_z_with_z_attr": safe_spearman(
                test_df["obstruction_response"],
                test_df["attribution_only"],
            ),
        }
    )
    return pd.DataFrame(rows)


def save_figures(test_df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    FIG_REPAIR_PATH.parent.mkdir(parents=True, exist_ok=True)

    repair = metrics[
        (metrics["metric"] == "repair_success_flip_at_k") & metrics["k"].notna()
    ]
    plt.figure(figsize=(6.5, 4.5))
    for method, sub in repair.groupby("method"):
        sub = sub.sort_values("k")
        plt.plot(sub["k"], sub["value"], marker="o", label=method)
    plt.xlabel("k")
    plt.ylabel("Repair@k flip")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_REPAIR_PATH, dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4.2))
    plt.scatter(
        test_df["obstruction_response"],
        test_df["observed_delta_margin"],
        s=5,
        alpha=0.25,
    )
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("tuned standardized obstruction-response score")
    plt.ylabel("observed patch Δmargin")
    plt.tight_layout()
    plt.savefig(FIG_SCATTER_PATH, dpi=200)
    plt.close()


def print_decision(test_metrics: pd.DataFrame, overlap: pd.DataFrame) -> None:
    def metric(method: str, name: str, k: int | None = None) -> float:
        sub = test_metrics[
            (test_metrics["method"] == method) & (test_metrics["metric"] == name)
        ]
        if k is not None:
            sub = sub[sub["k"] == k]
        return float(sub.iloc[0]["value"])

    attr_s = metric("attribution_only", "spearman")
    obst_s = metric("obstruction_response", "spearman")
    attr_r5 = metric("attribution_only", "repair_success_flip_at_k", k=5)
    obst_r5 = metric("obstruction_response", "repair_success_flip_at_k", k=5)
    cost_s = metric("attribution_plus_cost", "spearman")
    norm_s = metric("attribution_plus_norm", "spearman")
    corr = overlap[
        overlap["method"] == "obstruction_vs_attribution"
    ]["spearman_score_z_with_z_attr"].iloc[0]

    relative_repair_lift = (obst_r5 - attr_r5) / max(abs(attr_r5), 1e-8)
    spearman_lift = obst_s - attr_s
    improvement_not_reproduced = obst_s > max(cost_s, norm_s) + 0.03

    print("\nDecision")
    print(f"test Spearman attribution_only = {attr_s:.4f}")
    print(f"test Spearman obstruction_response = {obst_s:.4f}")
    print(f"test Repair@5 attribution_only = {attr_r5:.4f}")
    print(f"test Repair@5 obstruction_response = {obst_r5:.4f}")
    print(f"relative Repair@5 lift = {relative_repair_lift:.4f}")
    print(f"score_z vs attribution Spearman = {corr:.4f}")

    if (relative_repair_lift >= 0.05 or spearman_lift > 0.03) and improvement_not_reproduced:
        print(
            "DECISION: standardized obstruction-response beats attribution "
            "out-of-sample. Proceed to improved component obstruction-response."
        )
    elif corr > 0.95:
        print(
            "DECISION: standardized obstruction-response mostly rescales "
            "attribution. IOI is attribution-positive / obstruction-negative."
        )
    else:
        print(
            "DECISION: standardized obstruction-response does not beat attribution "
            "out-of-sample. Use IOI as patching infrastructure validation and move "
            "obstruction experiments to a harder task."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-examples", type=int, default=50)
    parser.add_argument("--test-examples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    examples = read_jsonl(TEST_PATH)
    val_examples = examples[: args.val_examples]
    test_examples = examples[
        args.val_examples : args.val_examples + args.test_examples
    ]

    print("Loading GPT-2 small...", flush=True)
    model = load_gpt2_small()
    sites = build_sites(model, include_resid_controls=False)
    if args.max_sites is not None:
        sites = sites[: args.max_sites]
    print(
        f"Using {len(val_examples)} validation examples, {len(test_examples)} test "
        f"examples, {len(sites)} component sites.",
        flush=True,
    )

    val_scores, val_oracle = compute_oracle_split(
        model,
        val_examples,
        sites,
        VAL_SCORES_PATH,
        VAL_ORACLE_PATH,
        args.batch_size,
        args.force,
    )
    test_scores, test_oracle = compute_oracle_split(
        model,
        test_examples,
        sites,
        TEST_SCORES_PATH,
        TEST_ORACLE_PATH,
        args.batch_size,
        args.force,
    )

    val_df = joined_example_site(val_scores, val_oracle)
    test_df = joined_example_site(test_scores, test_oracle)

    tuning, best = tune_hyperparameters(val_df)
    TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(TUNING_PATH, index=False)
    print("Best validation hyperparameters")
    print(best)

    val_eval = add_tuned_scores(
        val_df,
        best["cost_gamma"],
        best["norm_beta"],
        best["alpha"],
        best["gamma"],
    )
    test_eval = add_tuned_scores(
        test_df,
        best["cost_gamma"],
        best["norm_beta"],
        best["alpha"],
        best["gamma"],
    )

    val_metrics = evaluate_methods(val_eval, "validation")
    test_metrics = evaluate_methods(test_eval, "test")
    metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TEST_METRICS_PATH, index=False)

    overlap = site_level_overlap_metrics(test_eval)
    overlap.to_csv(OVERLAP_PATH, index=False)

    top_sites = (
        test_eval.groupby("site_id")
        .agg(
            observed_delta_margin=("observed_delta_margin", "mean"),
            attribution_only=("attribution_only", "mean"),
            obstruction_response=("obstruction_response", "mean"),
        )
        .reset_index()
    )
    rows = []
    for col in ["observed_delta_margin", "attribution_only", "obstruction_response"]:
        tmp = top_sites.sort_values(col, ascending=False).head(50).copy()
        tmp["ranking"] = col
        rows.append(tmp)
    pd.concat(rows, ignore_index=True).to_csv(TOP_SITES_PATH, index=False)

    save_figures(test_eval, test_metrics)

    print("Validation tuning summary")
    print(tuning.sort_values("spearman", ascending=False).head(20).to_string(index=False))
    print("\nTest metrics")
    print(test_metrics.to_string(index=False))
    print("\nOverlap metrics")
    print(overlap.to_string(index=False))
    print(f"\nSaved tuning to {TUNING_PATH}")
    print(f"Saved test metrics to {TEST_METRICS_PATH}")
    print(f"Saved overlap metrics to {OVERLAP_PATH}")
    print(f"Saved top sites to {TOP_SITES_PATH}")
    print(f"Saved figures to {FIG_REPAIR_PATH} and {FIG_SCATTER_PATH}")
    print_decision(test_metrics, overlap)


if __name__ == "__main__":
    main()
