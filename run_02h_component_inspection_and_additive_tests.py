from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCORES_PATH = Path("artifacts/results/component_level_scores.csv")
ORACLE_PATH = Path("artifacts/results/component_level_patching_oracle.csv")

TOP_ORACLE_PATH = Path("artifacts/results/top_component_oracle_sites.csv")
TOP_ATTR_PATH = Path("artifacts/results/top_component_attribution_sites.csv")
JOINED_PATH = Path("artifacts/results/component_site_joined_oracle_scores.csv")
CORR_PATH = Path("artifacts/results/component_site_spearman_correlation_matrix.csv")
REGRESSION_PATH = Path("artifacts/results/component_additive_regression_summary.csv")
RESIDUAL_TEST_PATH = Path("artifacts/results/component_residualized_obstruction_tests.csv")
SIGN_AUDIT_PATH = Path("artifacts/results/component_score_sign_scale_audit.csv")
REPAIR_AUDIT_PATH = Path("artifacts/results/component_repair_at_k_audit.csv")

ORACLE_ATTENTION_FIG = Path("artifacts/figures/component_oracle_attention_heatmap.png")
ORACLE_MLP_FIG = Path("artifacts/figures/component_oracle_mlp_heatmap.png")
ATTR_ATTENTION_FIG = Path("artifacts/figures/component_attribution_attention_heatmap.png")
ATTR_MLP_FIG = Path("artifacts/figures/component_attribution_mlp_heatmap.png")
CORR_FIG = Path("artifacts/figures/component_site_correlation_matrix.png")
RESIDUAL_SCATTER_FIG = Path("artifacts/figures/component_residualized_obstruction_scatter.png")
REPAIR_AUDIT_FIG = Path("artifacts/figures/component_repair_at_k_flip_vs_increase.png")

SITE_KEYS = ["site_id", "site_type", "layer", "head", "token_position_label"]
CORR_COLS = [
    "observed_delta_margin_mean",
    "pred_delta_m_mean",
    "score_mean",
    "activation_delta_norm_mean",
    "gradient_saliency_mean",
    "pred_delta_O_mean",
    "cost_mean",
]
KS = [1, 5, 10, 20, 50]


def ensure_dirs() -> None:
    for path in [
        TOP_ORACLE_PATH,
        TOP_ATTR_PATH,
        JOINED_PATH,
        CORR_PATH,
        REGRESSION_PATH,
        RESIDUAL_TEST_PATH,
        SIGN_AUDIT_PATH,
        REPAIR_AUDIT_PATH,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in [
        ORACLE_ATTENTION_FIG,
        ORACLE_MLP_FIG,
        ATTR_ATTENTION_FIG,
        ATTR_MLP_FIG,
        CORR_FIG,
        RESIDUAL_SCATTER_FIG,
        REPAIR_AUDIT_FIG,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)


def load_component_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SCORES_PATH.exists() or not ORACLE_PATH.exists():
        raise FileNotFoundError(
            "Missing component CSVs. Run "
            "`python -u run_02g_component_level_obstruction.py --max-examples 100 "
            "--batch-size 16` first."
        )

    score_df = pd.read_csv(SCORES_PATH)
    oracle_df = pd.read_csv(ORACLE_PATH)

    for df in [score_df, oracle_df]:
        df.rename(
            columns={
                "kind": "site_type",
                "position_name": "token_position_label",
            },
            inplace=True,
        )
        if "head" in df.columns:
            df["head"] = df["head"].where(df["head"].notna(), np.nan)

    return score_df, oracle_df


def aggregate_oracle(oracle_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "observed_delta_margin_mean": ("observed_delta_margin", "mean"),
        "observed_delta_margin_median": ("observed_delta_margin", "median"),
        "repair_success_flip_mean": ("repair_success_flip", "mean"),
        "repair_success_margin_increase_mean": (
            "repair_success_margin_increase",
            "mean",
        ),
    }
    if "off_target_kl" in oracle_df.columns:
        agg_spec["off_target_kl_mean"] = ("off_target_kl", "mean")

    return (
        oracle_df.groupby(SITE_KEYS, dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values("observed_delta_margin_mean", ascending=False)
    )


def aggregate_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    return (
        score_df.groupby(SITE_KEYS, dropna=False)
        .agg(
            pred_delta_m_mean=("pred_delta_m", "mean"),
            score_mean=("score", "mean"),
            activation_delta_norm_mean=("activation_delta_norm", "mean"),
            gradient_saliency_mean=("gradient_saliency", "mean"),
            pred_delta_O_mean=("pred_delta_O", "mean"),
            cost_mean=("cost", "mean"),
        )
        .reset_index()
    )


def plot_heatmap(
    table: pd.DataFrame,
    value_col: str,
    site_type: str,
    save_path: Path,
    title: str,
) -> None:
    sub = table[table["site_type"] == site_type].copy()
    if sub.empty:
        return

    if site_type == "head":
        heat = sub.pivot_table(
            index="layer",
            columns="head",
            values=value_col,
            aggfunc="mean",
        )
        xlabel = "head"
    else:
        heat = sub.pivot_table(
            index="layer",
            columns="token_position_label",
            values=value_col,
            aggfunc="mean",
        )
        xlabel = "token position"

    plt.figure(figsize=(8, 4.8))
    im = plt.imshow(heat.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label=value_col)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel(xlabel)
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr(method="spearman")


def plot_corr_matrix(corr: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Spearman rho")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def safe_spearman(x, y) -> tuple[float, float]:
    rho, p = spearmanr(x, y, nan_policy="omit")
    return float(rho), float(p)


def cross_validated_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "observed_delta_margin_mean",
    n_splits: int = 5,
) -> dict:
    x = df[feature_cols].fillna(0.0).values
    y = df[target_col].values
    cv = KFold(n_splits=min(n_splits, len(df)), shuffle=True, random_state=12345)

    fold_r2 = []
    fold_mae = []
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in cv.split(x):
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        fold_r2.append(r2_score(y[test_idx], pred))
        fold_mae.append(mean_absolute_error(y[test_idx], pred))
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(pred)

    rho, p = safe_spearman(y_pred_all, y_true_all)
    return {
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2)),
        "spearman_pred_true": rho,
        "spearman_p": p,
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
    }


def cross_val_predict_model_c(df: pd.DataFrame) -> np.ndarray:
    feature_cols = ["pred_delta_m_mean", "activation_delta_norm_mean"]
    x = df[feature_cols].fillna(0.0).values
    y = df["observed_delta_margin_mean"].values
    cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=12345)
    pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in cv.split(x):
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
    return pred


def top_residual_auc(score: pd.Series, y_resid: pd.Series) -> float:
    threshold = y_resid.quantile(0.90)
    y = (y_resid >= threshold).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - series.mean()) / std


def make_repair_at_k_audit(joined: pd.DataFrame) -> pd.DataFrame:
    method_cols = {
        "score_mean": "score",
        "pred_delta_m_mean": "attribution",
        "activation_delta_norm_mean": "activation_norm",
        "gradient_saliency_mean": "gradient_saliency",
        "cost_mean": "cost",
    }
    rows = []
    rng = np.random.default_rng(12345)
    joined = joined.copy()
    joined["random"] = rng.random(len(joined))
    method_cols["random"] = "random"

    for score_col, method in method_cols.items():
        ranked = joined.sort_values(score_col, ascending=False)
        for k in KS:
            sub = ranked.head(min(k, len(ranked)))
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "mean_observed_delta_margin": sub[
                        "observed_delta_margin_mean"
                    ].mean(),
                    "repair_success_margin_increase": sub[
                        "repair_success_margin_increase_mean"
                    ].mean(),
                    "repair_success_flip": sub["repair_success_flip_mean"].mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_repair_audit(repair_df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(7, 4.8))
    for method, sub in repair_df.groupby("method"):
        sub = sub.sort_values("k")
        plt.plot(
            sub["k"],
            sub["repair_success_margin_increase"],
            marker="o",
            linestyle="-",
            label=f"{method}: increase",
        )
        plt.plot(
            sub["k"],
            sub["repair_success_flip"],
            marker="x",
            linestyle="--",
            label=f"{method}: flip",
        )
    plt.xlabel("top-k component sites")
    plt.ylabel("mean success rate")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    score_df, oracle_df = load_component_tables()

    oracle_site = aggregate_oracle(oracle_df)
    score_site = aggregate_scores(score_df)

    top_oracle = oracle_site.head(50)
    top_attr = score_site.sort_values("pred_delta_m_mean", ascending=False).head(50)
    top_oracle.to_csv(TOP_ORACLE_PATH, index=False)
    top_attr.to_csv(TOP_ATTR_PATH, index=False)

    plot_heatmap(
        oracle_site,
        "observed_delta_margin_mean",
        "head",
        ORACLE_ATTENTION_FIG,
        "Oracle patch effect: attention heads",
    )
    plot_heatmap(
        oracle_site,
        "observed_delta_margin_mean",
        "mlp",
        ORACLE_MLP_FIG,
        "Oracle patch effect: MLPs",
    )
    plot_heatmap(
        score_site,
        "pred_delta_m_mean",
        "head",
        ATTR_ATTENTION_FIG,
        "Attribution score: attention heads",
    )
    plot_heatmap(
        score_site,
        "pred_delta_m_mean",
        "mlp",
        ATTR_MLP_FIG,
        "Attribution score: MLPs",
    )

    joined = oracle_site.merge(score_site, on=SITE_KEYS, how="inner")
    joined.to_csv(JOINED_PATH, index=False)

    corr = spearman_matrix(joined, CORR_COLS)
    corr.to_csv(CORR_PATH)
    plot_corr_matrix(corr, CORR_FIG)

    models = {
        "A_attribution": ["pred_delta_m_mean"],
        "B_activation_norm": ["activation_delta_norm_mean"],
        "C_attr_plus_norm": ["pred_delta_m_mean", "activation_delta_norm_mean"],
        "D_C_plus_pred_delta_O": [
            "pred_delta_m_mean",
            "activation_delta_norm_mean",
            "pred_delta_O_mean",
        ],
        "E_C_plus_cost": [
            "pred_delta_m_mean",
            "activation_delta_norm_mean",
            "cost_mean",
        ],
        "F_C_plus_pred_delta_O_plus_cost": [
            "pred_delta_m_mean",
            "activation_delta_norm_mean",
            "pred_delta_O_mean",
            "cost_mean",
        ],
    }

    regression_rows = []
    for name, features in models.items():
        row = {"model": name, "features": ",".join(features)}
        row.update(cross_validated_regression(joined, features))
        regression_rows.append(row)
    regression_df = pd.DataFrame(regression_rows)
    c_row = regression_df[regression_df["model"] == "C_attr_plus_norm"].iloc[0]
    regression_df["delta_r2_vs_C"] = regression_df["r2_mean"] - c_row["r2_mean"]
    regression_df["delta_spearman_vs_C"] = (
        regression_df["spearman_pred_true"] - c_row["spearman_pred_true"]
    )
    regression_df.to_csv(REGRESSION_PATH, index=False)

    joined["model_c_pred"] = cross_val_predict_model_c(joined)
    joined["y_resid"] = joined["observed_delta_margin_mean"] - joined["model_c_pred"]
    residual_rows = []
    for col in ["pred_delta_O_mean", "score_mean", "cost_mean"]:
        rho, p = safe_spearman(joined[col], joined["y_resid"])
        residual_rows.append(
            {
                "variable": col,
                "spearman_with_y_resid": rho,
                "p": p,
                "auc_detects_top_residual_positive_sites": top_residual_auc(
                    joined[col],
                    joined["y_resid"],
                ),
            }
        )
    joined["neg_cost_mean"] = -joined["cost_mean"]
    rho, p = safe_spearman(joined["neg_cost_mean"], joined["y_resid"])
    residual_rows.append(
        {
            "variable": "-cost_mean",
            "spearman_with_y_resid": rho,
            "p": p,
            "auc_detects_top_residual_positive_sites": top_residual_auc(
                joined["neg_cost_mean"],
                joined["y_resid"],
            ),
        }
    )
    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(RESIDUAL_TEST_PATH, index=False)

    plt.figure(figsize=(5, 4.2))
    plt.scatter(joined["pred_delta_O_mean"], joined["y_resid"], s=14, alpha=0.6)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("pred_delta_O_mean")
    plt.ylabel("observed Δmargin residual after attribution+norm")
    plt.tight_layout()
    plt.savefig(RESIDUAL_SCATTER_FIG, dpi=200)
    plt.close()

    sign_rows = []
    dist_cols = [
        "pred_delta_m_mean",
        "pred_delta_O_mean",
        "cost_mean",
        "score_mean",
        "observed_delta_margin_mean",
    ]
    for col in dist_cols:
        sign_rows.append(
            {
                "audit": "distribution",
                "variable": col,
                "mean": joined[col].mean(),
                "std": joined[col].std(),
                "min": joined[col].min(),
                "median": joined[col].median(),
                "max": joined[col].max(),
            }
        )

    sign_tests = {
        "-pred_delta_O": -joined["pred_delta_O_mean"],
        "pred_delta_O": joined["pred_delta_O_mean"],
        "cost": joined["cost_mean"],
        "-cost": -joined["cost_mean"],
        "score": joined["score_mean"],
    }
    for name, values in sign_tests.items():
        rho, p = safe_spearman(values, joined["observed_delta_margin_mean"])
        sign_rows.append(
            {
                "audit": "spearman_with_observed_delta_margin",
                "variable": name,
                "spearman": rho,
                "p": p,
            }
        )

    z_pred_m = zscore(joined["pred_delta_m_mean"])
    z_pred_o = zscore(joined["pred_delta_O_mean"])
    z_cost = zscore(joined["cost_mean"])
    for gamma in [0, 0.01, 0.1, 1.0]:
        score_z = z_pred_m - z_pred_o - gamma * z_cost
        rho, p = safe_spearman(score_z, joined["observed_delta_margin_mean"])
        sign_rows.append(
            {
                "audit": "standardized_score_gamma_sweep",
                "variable": "score_z",
                "gamma": gamma,
                "spearman": rho,
                "p": p,
            }
        )
    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(SIGN_AUDIT_PATH, index=False)

    repair_df = make_repair_at_k_audit(joined)
    repair_df.to_csv(REPAIR_AUDIT_PATH, index=False)
    plot_repair_audit(repair_df, REPAIR_AUDIT_FIG)

    print("Top 50 causal oracle sites")
    print(top_oracle.head(50).to_string(index=False))
    print("\nAdditive regression summary")
    print(regression_df.to_string(index=False))
    print("\nResidualized obstruction tests")
    print(residual_df.to_string(index=False))
    print("\nScore sign/scale audit")
    print(sign_df.to_string(index=False))

    d_row = regression_df[regression_df["model"] == "D_C_plus_pred_delta_O"].iloc[0]
    f_row = regression_df[
        regression_df["model"] == "F_C_plus_pred_delta_O_plus_cost"
    ].iloc[0]
    d_or_f_adds = (
        max(d_row["delta_r2_vs_C"], f_row["delta_r2_vs_C"]) > 0.03
        or max(d_row["delta_spearman_vs_C"], f_row["delta_spearman_vs_C"]) > 0.03
    )

    best_score_z = sign_df[
        sign_df["audit"] == "standardized_score_gamma_sweep"
    ]["spearman"].max()
    raw_score_rho = sign_df[
        (sign_df["audit"] == "spearman_with_observed_delta_margin")
        & (sign_df["variable"] == "score")
    ]["spearman"].iloc[0]

    print("\nDecision")
    if d_or_f_adds:
        print(
            "DECISION: Obstruction term has additive predictive value beyond "
            "attribution and norm. Proceed to improved obstruction-response patching."
        )
    elif best_score_z > raw_score_rho + 0.10 and best_score_z > 0.30:
        print(
            "DECISION: Original score was mis-scaled. Replace raw score with "
            "standardized score and rerun component-level ranking."
        )
    else:
        print(
            "DECISION: Attribution patching is the main positive result. Current "
            "obstruction term has no additive value on IOI component repair. Use "
            "this as a baseline/failure section and move to a harder task where "
            "attribution is less saturated."
        )


if __name__ == "__main__":
    main()
