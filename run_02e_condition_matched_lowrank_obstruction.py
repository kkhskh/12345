from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.cache import resid_site_names, run_with_resid_cache
from src.data_ioi import read_jsonl
from src.model import load_gpt2_small, logit_margin_from_logits

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
    vector_obstruction,
)


TRAIN_PATH = Path("data/processed/ioi_train.jsonl")
TEST_PATH = Path("data/processed/ioi_test.jsonl")
CSV_PATH = Path("artifacts/results/condition_matched_lowrank_obstruction.csv")
SUMMARY_PATH = Path("artifacts/results/condition_matched_lowrank_auc_summary.csv")
FIGURE_PATH = Path("artifacts/figures/condition_matched_lowrank_obstruction_residualized.png")

CONFOUND_COLS = [
    "total_residual_norm",
    "mean_site_residual_norm",
    "max_site_residual_norm",
    "evidence_norm",
    "abs_margin_clean_contrast",
    "token_length",
    "condition_text_length",
]


def token_len(model, prompt: str) -> int:
    return len(model.tokenizer.encode(prompt, add_special_tokens=False)) + 1


def fit_residualizer(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for projection, sub in df.groupby("projection"):
        sub = sub.copy()
        x = sub[CONFOUND_COLS].values
        y = sub["O_norm"].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        reg = LinearRegression().fit(x_scaled, y)
        sub["O_pred_from_confounds"] = reg.predict(x_scaled)
        sub["O_resid"] = sub["O_norm"] - sub["O_pred_from_confounds"]
        sub["confound_r2"] = reg.score(x_scaled, y)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def auc_binary(df: pd.DataFrame, positive_conditions, negative_conditions, score_col: str):
    sub = df[df["condition"].isin(set(positive_conditions) | set(negative_conditions))]
    y = sub["condition"].isin(positive_conditions).astype(int).values
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, sub[score_col].values)


def auc_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for projection, sub in df.groupby("projection"):
        row = {
            "projection": projection,
            "confound_r2": sub["confound_r2"].iloc[0],
        }
        for score_col, prefix in [("O_norm", "raw"), ("O_resid", "residualized")]:
            row[f"{prefix}_auc_conflict_or_malformed_gt_clean"] = auc_binary(
                sub,
                positive_conditions=CONFLICT_GROUPS,
                negative_conditions={"clean_coherent"},
                score_col=score_col,
            )
            row[f"{prefix}_auc_conflict_or_malformed_gt_coherent"] = auc_binary(
                sub,
                positive_conditions=CONFLICT_GROUPS,
                negative_conditions=COHERENT_GROUPS,
                score_col=score_col,
            )
            row[f"{prefix}_auc_swapped_gt_clean"] = auc_binary(
                sub,
                positive_conditions={"swapped_coherent"},
                negative_conditions={"clean_coherent"},
                score_col=score_col,
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("projection")


@torch.no_grad()
def score_projection_with_confounds(
    model,
    examples,
    projections,
    transports,
    weights,
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
            tokens, logits, cache = run_with_resid_cache(model, prompts)

            for i, ex in enumerate(batch):
                prompt = ex[prompt_field]
                positions = find_positions(model, tokens[i], ex, prompt, condition)

                site_vectors = []
                site_norms = []
                evidence_norm = torch.tensor(0.0)
                for pos_idx, pos_name in enumerate(POSITION_NAMES):
                    pos_vectors = []
                    token_pos = positions[pos_name]
                    for layer_idx, site_name in enumerate(site_names):
                        h = cache[site_name][i, token_pos, :].detach().cpu()
                        pos_vectors.append(h)
                        site_norms.append(float(h.pow(2).sum()))
                        p = projections[pos_idx, layer_idx, :RANK]
                        s = p @ h
                        evidence_norm = evidence_norm + s.pow(2).sum()
                    site_vectors.append(pos_vectors)

                raw, normalized = vector_obstruction(
                    site_vectors,
                    projections,
                    transports,
                    weights,
                    RANK,
                )
                margin = logit_margin_from_logits(
                    logits[i : i + 1],
                    ex["repair_answer_a_id"],
                    ex["repair_answer_b_id"],
                    token_pos=positions["final_token"],
                )[0]

                rows.append(
                    {
                        "id": ex["id"],
                        "split": ex["split"],
                        "projection": projection_label,
                        "rank": RANK,
                        "condition": condition,
                        "O_raw": float(raw),
                        "O_norm": float(normalized),
                        "margin_clean_contrast": float(margin.cpu()),
                        "abs_margin_clean_contrast": float(abs(margin.cpu())),
                        "token_length": token_len(model, prompt),
                        "condition_text_length": len(prompt),
                        "total_residual_norm": float(sum(site_norms)),
                        "mean_site_residual_norm": float(pd.Series(site_norms).mean()),
                        "max_site_residual_norm": float(max(site_norms)),
                        "evidence_norm": float(evidence_norm),
                    }
                )

    return pd.DataFrame(rows)


def plot_residualized(df: pd.DataFrame, summary: pd.DataFrame, save_path: Path) -> None:
    ioi = df[df["projection"] == "ioi_difference"]
    auc = summary.loc[
        summary["projection"] == "ioi_difference",
        "residualized_auc_conflict_or_malformed_gt_coherent",
    ].iloc[0]

    plt.figure(figsize=(7, 4.5))
    for condition in CONDITIONS:
        sub = ioi[ioi["condition"] == condition]
        plt.hist(sub["O_resid"], bins=40, alpha=0.45, density=True, label=condition)

    plt.xlabel("residualized low-rank obstruction")
    plt.ylabel("density")
    plt.title(f"Condition-matched low-rank obstruction; AUC={auc:.3f}")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)


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

    scored = []
    for label, projections in [
        ("ioi_difference", ioi_projections),
        ("random", random_projections),
        ("generic_pca", generic_projections),
    ]:
        print(f"Fitting transports and scoring projection: {label}", flush=True)
        transports, weights, _ = fit_vector_transports(clean_mats, projections, RANK)
        scored.append(
            score_projection_with_confounds(
                model,
                test_examples,
                projections,
                transports,
                weights,
                label,
            )
        )

    df = pd.concat(scored, ignore_index=True)
    df = fit_residualizer(df)
    summary = auc_summary(df)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_residualized(df, summary, FIGURE_PATH)

    print("raw and residualized AUC summary")
    print(summary)

    ioi = summary[summary["projection"] == "ioi_difference"].iloc[0]
    random = summary[summary["projection"] == "random"].iloc[0]
    generic = summary[summary["projection"] == "generic_pca"].iloc[0]

    ioi_conflict_auc = ioi["residualized_auc_conflict_or_malformed_gt_coherent"]
    ioi_swapped_auc = ioi["residualized_auc_swapped_gt_clean"]
    beats_controls = (
        ioi_conflict_auc
        > random["residualized_auc_conflict_or_malformed_gt_coherent"]
        and ioi_conflict_auc
        > generic["residualized_auc_conflict_or_malformed_gt_coherent"]
    )

    print(f"Saved detailed results to {CSV_PATH}")
    print(f"Saved AUC summary to {SUMMARY_PATH}")
    print(f"Saved residualized figure to {FIGURE_PATH}")

    if ioi_conflict_auc > 0.65 and ioi_swapped_auc < 0.60 and beats_controls:
        print("GO: residualized low-rank obstruction passes all controls.")
    else:
        print(
            "NO-GO: residualized low-rank obstruction does not pass all controls. "
            "Do not run patching."
        )


if __name__ == "__main__":
    main()
