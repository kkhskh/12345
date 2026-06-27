from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def repair_at_k(
    patch_df: pd.DataFrame,
    method: str,
    k: int,
    success_col="repair_success_flip",
):
    sub = patch_df[(patch_df["method"] == method) & (patch_df["k"] == k)]
    return sub[success_col].mean()


def mean_delta_margin(patch_df: pd.DataFrame, method: str, k: int):
    sub = patch_df[(patch_df["method"] == method) & (patch_df["k"] == k)]
    return sub["delta_margin"].mean()


def spearman_pred_observed(
    score_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    pred_col: str,
    obs_col: str,
):
    merged = score_df.merge(observed_df, on=["id", "layer"])
    rho, p = spearmanr(merged[pred_col], merged[obs_col])
    return {"spearman": rho, "p": p, "n": len(merged)}


def effective_site_auc(score_df: pd.DataFrame, observed_df: pd.DataFrame, pred_col: str):
    """
    Effective site = actual single-layer patch flips margin positive.
    """
    merged = score_df.merge(observed_df, on=["id", "layer"])
    y = merged["repair_success_flip"].astype(int).values
    scores = merged[pred_col].values

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, scores)


def obstruction_reduction_vs_repair_probability(patch_df: pd.DataFrame):
    """
    Return logistic-ready table:
      x = -delta_O
      y = repair_success_flip
    """
    out = patch_df.copy()
    out["obstruction_reduction"] = -out["delta_O"]
    out["repair"] = out["repair_success_flip"].astype(int)
    return out[
        [
            "id",
            "method",
            "k",
            "obstruction_reduction",
            "delta_margin",
            "repair",
        ]
    ]
