from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .cache import run_with_resid_cache
from .germs import scalar_germs
from .model import logit_margin_from_logits
from .obstruction import compute_obstruction


@torch.no_grad()
def obstruction_clean_corrupt_table(model, examples, r, w, batch_size: int = 32):
    rows = []

    for start in tqdm(range(0, len(examples), batch_size)):
        batch = examples[start : start + batch_size]
        clean_prompts = [ex["clean_prompt"] for ex in batch]
        corrupt_prompts = [ex["corrupt_prompt"] for ex in batch]

        _, clean_logits, clean_cache = run_with_resid_cache(model, clean_prompts)
        _, corrupt_logits, corrupt_cache = run_with_resid_cache(model, corrupt_prompts)

        for i, ex in enumerate(batch):
            clean_cache_i = {
                key: value[i : i + 1] for key, value in clean_cache.items()
            }
            corrupt_cache_i = {
                key: value[i : i + 1] for key, value in corrupt_cache.items()
            }

            s_clean = scalar_germs(
                model,
                clean_cache_i,
                ex["repair_answer_a_id"],
                ex["repair_answer_b_id"],
            )
            o_clean = compute_obstruction(s_clean, r, w)

            m_clean = logit_margin_from_logits(
                clean_logits[i : i + 1],
                ex["repair_answer_a_id"],
                ex["repair_answer_b_id"],
            )

            s_corrupt = scalar_germs(
                model,
                corrupt_cache_i,
                ex["repair_answer_a_id"],
                ex["repair_answer_b_id"],
            )
            o_corrupt = compute_obstruction(s_corrupt, r, w)

            m_corrupt = logit_margin_from_logits(
                corrupt_logits[i : i + 1],
                ex["repair_answer_a_id"],
                ex["repair_answer_b_id"],
            )

            rows.append(
                {
                    "id": ex["id"],
                    "split": ex["split"],
                    "O_clean": float(o_clean["normalized"][0].cpu()),
                    "O_corrupt": float(o_corrupt["normalized"][0].cpu()),
                    "margin_clean": float(m_clean[0].cpu()),
                    "margin_corrupt_clean_contrast": float(m_corrupt[0].cpu()),
                    "clean_correct": ex["clean_correct"],
                    "clean_competitor": ex["clean_competitor"],
                }
            )

    return pd.DataFrame(rows)


def plot_clean_vs_corrupt_obstruction(df, save_path=None):
    y = np.concatenate(
        [
            np.zeros(len(df)),
            np.ones(len(df)),
        ]
    )
    scores = np.concatenate(
        [
            df["O_clean"].values,
            df["O_corrupt"].values,
        ]
    )
    auc = roc_auc_score(y, scores)

    plt.figure(figsize=(5, 4))
    plt.hist(df["O_clean"], bins=40, alpha=0.55, density=True, label="clean")
    plt.hist(df["O_corrupt"], bins=40, alpha=0.55, density=True, label="corrupt")
    plt.xlabel("normalized obstruction")
    plt.ylabel("density")
    plt.title(f"Clean vs corrupt obstruction; AUC={auc:.3f}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return auc


def plot_pred_vs_observed_delta_m(score_df, observed_df, save_path=None):
    merged = score_df.merge(observed_df, on=["id", "layer"])
    rho = merged[["pred_delta_m", "delta_margin"]].corr(method="spearman").iloc[0, 1]

    plt.figure(figsize=(4.5, 4))
    plt.scatter(merged["pred_delta_m"], merged["delta_margin"], s=10, alpha=0.4)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("predicted Δm")
    plt.ylabel("observed Δm")
    plt.title(f"Predicted vs observed Δm; Spearman={rho:.3f}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return rho


def plot_pred_vs_observed_delta_o(score_df, observed_df, save_path=None):
    merged = score_df.merge(observed_df, on=["id", "layer"])
    rho = merged[["pred_delta_O", "delta_O"]].corr(method="spearman").iloc[0, 1]

    plt.figure(figsize=(4.5, 4))
    plt.scatter(merged["pred_delta_O"], merged["delta_O"], s=10, alpha=0.4)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("predicted ΔO")
    plt.ylabel("observed ΔO")
    plt.title(f"Predicted vs observed ΔO; Spearman={rho:.3f}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return rho


def plot_pred_vs_observed_delta_O(score_df, observed_df, save_path=None):
    return plot_pred_vs_observed_delta_o(score_df, observed_df, save_path=save_path)


def plot_repair_at_k(patch_df, save_path=None):
    summary = (
        patch_df.groupby(["method", "k"])["repair_success_flip"].mean().reset_index()
    )

    plt.figure(figsize=(6, 4))
    for method, sub in summary.groupby("method"):
        sub = sub.sort_values("k")
        plt.plot(sub["k"], sub["repair_success_flip"], marker="o", label=method)

    plt.xlabel("k patched residual sites")
    plt.ylabel("Repair@k: patched margin > 0")
    plt.title("Top-k repair vs baselines")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return summary


def plot_obstruction_reduction_vs_repair(patch_df, save_path=None, bins=10):
    df = patch_df.copy()
    df["obstruction_reduction"] = -df["delta_O"]

    df["bin"] = pd.qcut(
        df["obstruction_reduction"],
        q=bins,
        duplicates="drop",
    )

    summary = (
        df.groupby("bin", observed=False)
        .agg(
            obstruction_reduction=("obstruction_reduction", "mean"),
            repair_prob=("repair_success_flip", "mean"),
            n=("repair_success_flip", "size"),
        )
        .reset_index()
    )

    plt.figure(figsize=(5, 4))
    plt.plot(summary["obstruction_reduction"], summary["repair_prob"], marker="o")
    plt.xlabel("actual obstruction reduction: -ΔO")
    plt.ylabel("repair probability")
    plt.title("Obstruction reduction vs repair")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return summary
