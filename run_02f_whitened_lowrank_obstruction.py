from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.cache import resid_site_names, run_with_resid_cache
from src.data_ioi import read_jsonl

from run_02d_lowrank_residual_obstruction import (
    BATCH_SIZE,
    CONDITIONS,
    CONFLICT_GROUPS,
    COHERENT_GROUPS,
    LAMBDA_RIDGE,
    POSITION_NAMES,
    RANK,
    collect_train_site_mats,
    find_positions,
    fit_vector_transports,
    learn_pca_projections,
    make_random_projections,
)
from src.model import load_gpt2_small


TRAIN_PATH = Path("data/processed/ioi_train.jsonl")
TEST_PATH = Path("data/processed/ioi_test.jsonl")
CSV_PATH = Path("artifacts/results/whitened_lowrank_obstruction_rank16.csv")
SUMMARY_PATH = Path("artifacts/results/whitened_lowrank_auc_summary.csv")
FIGURE_PATH = Path("artifacts/figures/whitened_lowrank_obstruction_rank16.png")

PROJECTION_TYPES = ["ioi_difference", "random", "generic_pca"]
COV_LAMBDA = 1e-3


def project_site_matrix(matrix: torch.Tensor, projection: torch.Tensor):
    return matrix @ projection.T


def fit_edge_precision_mats(clean_mats, projections, transports, rank: int):
    n_pos, n_layers_plus_one = projections.shape[:2]
    precisions = torch.empty(n_pos, n_layers_plus_one - 1, rank, rank)
    cov_diagnostics = []

    eye = torch.eye(rank)
    for pos_idx, pos_name in enumerate(POSITION_NAMES):
        for layer_idx in range(n_layers_plus_one - 1):
            x = project_site_matrix(
                clean_mats[pos_idx][layer_idx],
                projections[pos_idx, layer_idx, :rank],
            )
            y = project_site_matrix(
                clean_mats[pos_idx][layer_idx + 1],
                projections[pos_idx, layer_idx + 1, :rank],
            )
            residuals = x @ transports[pos_idx, layer_idx].T - y
            centered = residuals - residuals.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
            cov = cov + COV_LAMBDA * eye
            precisions[pos_idx, layer_idx] = torch.linalg.inv(cov)
            cov_diagnostics.append(
                {
                    "position": pos_name,
                    "layer_edge": layer_idx,
                    "mean_edge_residual_norm": float(
                        residuals.pow(2).sum(dim=1).mean()
                    ),
                    "cov_trace": float(torch.trace(cov)),
                }
            )

    return precisions, cov_diagnostics


def whitened_obstruction(site_vectors, projections, transports, precisions, rank: int):
    total = torch.tensor(0.0)
    n_terms = 0
    n_pos, n_layers_plus_one = projections.shape[:2]

    for pos_idx in range(n_pos):
        germs = []
        for layer_idx in range(n_layers_plus_one):
            h = site_vectors[pos_idx][layer_idx]
            p = projections[pos_idx, layer_idx, :rank]
            germs.append(p @ h)

        for layer_idx in range(n_layers_plus_one - 1):
            edge_residual = (
                transports[pos_idx, layer_idx] @ germs[layer_idx]
                - germs[layer_idx + 1]
            )
            precision = precisions[pos_idx, layer_idx]
            total = total + edge_residual @ precision @ edge_residual
            n_terms += 1

    return total, total / n_terms


@torch.no_grad()
def score_projection(
    model,
    examples,
    projections,
    transports,
    precisions,
    projection_label: str,
):
    rows = []
    site_names = resid_site_names(model)

    for condition, prompt_field in CONDITIONS.items():
        for start in tqdm(
            range(0, len(examples), BATCH_SIZE),
            desc=f"score {projection_label} {condition}",
        ):
            batch = examples[start : start + BATCH_SIZE]
            prompts = [ex[prompt_field] for ex in batch]
            tokens, _, cache = run_with_resid_cache(model, prompts)

            for i, ex in enumerate(batch):
                prompt = ex[prompt_field]
                positions = find_positions(model, tokens[i], ex, prompt, condition)
                site_vectors = []
                for pos_name in POSITION_NAMES:
                    token_pos = positions[pos_name]
                    site_vectors.append(
                        [
                            cache[site_name][i, token_pos, :].detach().cpu()
                            for site_name in site_names
                        ]
                    )

                raw, normalized = whitened_obstruction(
                    site_vectors,
                    projections,
                    transports,
                    precisions,
                    RANK,
                )

                rows.append(
                    {
                        "id": ex["id"],
                        "split": ex["split"],
                        "projection": projection_label,
                        "rank": RANK,
                        "condition": condition,
                        "O_white": float(raw),
                        "O_white_norm": float(normalized),
                    }
                )

    return pd.DataFrame(rows)


def auc_binary(df: pd.DataFrame, positive_conditions, negative_conditions):
    sub = df[df["condition"].isin(set(positive_conditions) | set(negative_conditions))]
    y = sub["condition"].isin(positive_conditions).astype(int).values
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, sub["O_white_norm"].values)


def auc_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for projection, sub in df.groupby("projection"):
        rows.append(
            {
                "projection": projection,
                "auc_conflict_or_malformed_gt_coherent": auc_binary(
                    sub,
                    positive_conditions=CONFLICT_GROUPS,
                    negative_conditions=COHERENT_GROUPS,
                ),
                "auc_conflict_gt_clean": auc_binary(
                    sub,
                    positive_conditions={"conflict"},
                    negative_conditions={"clean_coherent"},
                ),
                "auc_malformed_gt_clean": auc_binary(
                    sub,
                    positive_conditions={"malformed"},
                    negative_conditions={"clean_coherent"},
                ),
                "auc_swapped_gt_clean": auc_binary(
                    sub,
                    positive_conditions={"swapped_coherent"},
                    negative_conditions={"clean_coherent"},
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("projection")


def plot_whitened(df: pd.DataFrame, summary: pd.DataFrame, save_path: Path):
    ioi = df[df["projection"] == "ioi_difference"]
    auc = summary.loc[
        summary["projection"] == "ioi_difference",
        "auc_conflict_or_malformed_gt_coherent",
    ].iloc[0]

    plt.figure(figsize=(7, 4.5))
    for condition in CONDITIONS:
        sub = ioi[ioi["condition"] == condition]
        plt.hist(sub["O_white_norm"], bins=40, alpha=0.45, density=True, label=condition)

    plt.xlabel("whitened low-rank obstruction")
    plt.ylabel("density")
    plt.title(f"Whitened low-rank rank 16; conflict-vs-coherent AUC={auc:.3f}")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)


def evaluate_projection(model, test_examples, clean_mats, projections, label: str):
    transports, _, _ = fit_vector_transports(clean_mats, projections, RANK)
    precisions, _ = fit_edge_precision_mats(
        clean_mats,
        projections,
        transports,
        RANK,
    )
    return score_projection(
        model,
        test_examples,
        projections,
        transports,
        precisions,
        label,
    )


def main() -> None:
    train_examples = read_jsonl(TRAIN_PATH)
    test_examples = read_jsonl(TEST_PATH)

    print(f"rank = {RANK}", flush=True)
    print(f"Loaded train/test examples: {len(train_examples)} / {len(test_examples)}")
    print("Loading GPT-2 small...", flush=True)
    model = load_gpt2_small()

    clean_mats, diff_mats = collect_train_site_mats(model, train_examples)

    print("Learning IOI-difference PCA projections...", flush=True)
    ioi_projections = learn_pca_projections(diff_mats, RANK, "ioi PCA")
    del diff_mats
    gc.collect()

    print("Learning generic residual PCA projections...", flush=True)
    generic_projections = learn_pca_projections(clean_mats, RANK, "generic PCA")
    random_projections = make_random_projections(ioi_projections)

    rows = []
    for label, projections in [
        ("ioi_difference", ioi_projections),
        ("random", random_projections),
        ("generic_pca", generic_projections),
    ]:
        print(f"Evaluating projection: {label}", flush=True)
        rows.append(evaluate_projection(model, test_examples, clean_mats, projections, label))

    df = pd.concat(rows, ignore_index=True)
    summary = auc_summary(df)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_whitened(df, summary, FIGURE_PATH)

    ioi = df[df["projection"] == "ioi_difference"]
    print("mean O_white by group")
    print(ioi.groupby("condition")["O_white_norm"].mean().sort_index())
    print("AUC summary")
    print(summary)

    ioi_summary = summary[summary["projection"] == "ioi_difference"].iloc[0]
    random_summary = summary[summary["projection"] == "random"].iloc[0]
    generic_summary = summary[summary["projection"] == "generic_pca"].iloc[0]

    ioi_auc = ioi_summary["auc_conflict_or_malformed_gt_coherent"]
    random_auc = random_summary["auc_conflict_or_malformed_gt_coherent"]
    generic_auc = generic_summary["auc_conflict_or_malformed_gt_coherent"]
    swapped_auc = ioi_summary["auc_swapped_gt_clean"]

    print(f"Saved detailed results to {CSV_PATH}")
    print(f"Saved AUC summary to {SUMMARY_PATH}")
    print(f"Saved figure to {FIGURE_PATH}")

    if ioi_auc > 0.65 and ioi_auc >= random_auc + 0.10 and ioi_auc >= generic_auc + 0.10 and swapped_auc < 0.65:
        print("GO: whitened low-rank residual obstruction passes all controls.")
    else:
        print(
            "NO-GO: whitened residual-chain obstruction fails controls. "
            "Move to component-level attention head and MLP sites."
        )


if __name__ == "__main__":
    main()
