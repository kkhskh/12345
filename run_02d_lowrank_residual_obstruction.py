from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.cache import resid_site_names, run_with_resid_cache
from src.data_ioi import read_jsonl
from src.model import load_gpt2_small, logit_margin_from_logits


TRAIN_PATH = Path("data/processed/ioi_train.jsonl")
TEST_PATH = Path("data/processed/ioi_test.jsonl")
TRANSPORT_PATH = Path("artifacts/transports/lowrank_residual_rank16_transports.json")
CSV_PATH = Path("artifacts/results/lowrank_residual_obstruction_rank16.csv")
FIGURE_PATH = Path("artifacts/figures/lowrank_residual_obstruction_rank16.png")
SWEEP_CSV_PATH = Path("artifacts/results/lowrank_rank_sweep.csv")
SWEEP_FIGURE_PATH = Path("artifacts/figures/lowrank_rank_sweep_auc.png")

POSITION_NAMES = ["first_name", "second_name", "giver_name", "final_token"]
CONDITIONS = {
    "clean_coherent": "clean_prompt",
    "swapped_coherent": "swapped_coherent_prompt",
    "conflict": "conflict_prompt",
    "malformed": "malformed_prompt",
}
COHERENT_GROUPS = {"clean_coherent", "swapped_coherent"}
CONFLICT_GROUPS = {"conflict", "malformed"}
RANK = 16
RANK_SWEEP = [4, 8, 16, 32]
MAX_RANK = max(RANK_SWEEP)
LAMBDA_RIDGE = 1e-3
BATCH_SIZE = 32


def token_len(model, prompt: str) -> int:
    return len(model.tokenizer.encode(prompt, add_special_tokens=False)) + 1


def find_positions(model, tokens: torch.Tensor, ex: dict, prompt: str, condition: str):
    final_pos = token_len(model, prompt) - 1
    ids = tokens[: final_pos + 1].tolist()

    name_a_id = ex["clean_competitor_id"]
    name_b_id = ex["clean_correct_id"]
    name_a_positions = [idx for idx, tok in enumerate(ids) if tok == name_a_id]
    name_b_positions = [idx for idx, tok in enumerate(ids) if tok == name_b_id]

    if not name_a_positions or not name_b_positions:
        raise ValueError(
            f"Could not find both names in prompt for {ex['id']} / {condition}: "
            f"{prompt!r}"
        )

    giver_positions = (
        name_b_positions if condition == "swapped_coherent" else name_a_positions
    )
    giver_pos = giver_positions[1] if len(giver_positions) > 1 else giver_positions[0]

    return {
        "first_name": name_a_positions[0],
        "second_name": name_b_positions[0],
        "giver_name": giver_pos,
        "final_token": final_pos,
    }


def empty_site_lists(n_layers_plus_one: int):
    return [
        [[] for _ in range(n_layers_plus_one)]
        for _ in POSITION_NAMES
    ]


def append_resid_sites(site_lists, model, cache, positions, row_idx: int):
    site_names = resid_site_names(model)
    for pos_idx, pos_name in enumerate(POSITION_NAMES):
        token_pos = positions[pos_name]
        for layer_idx, site_name in enumerate(site_names):
            site_lists[pos_idx][layer_idx].append(
                cache[site_name][row_idx, token_pos, :].detach().cpu()
            )


def stack_site_lists(site_lists):
    return [
        [torch.stack(layer_rows, dim=0).float() for layer_rows in pos_rows]
        for pos_rows in site_lists
    ]


@torch.no_grad()
def collect_train_site_mats(model, examples):
    n_layers_plus_one = model.cfg.n_layers + 1
    clean_lists = empty_site_lists(n_layers_plus_one)
    diff_lists = empty_site_lists(n_layers_plus_one)

    for start in tqdm(range(0, len(examples), BATCH_SIZE), desc="collect train sites"):
        batch = examples[start : start + BATCH_SIZE]
        clean_prompts = [ex["clean_prompt"] for ex in batch]
        swapped_prompts = [ex["swapped_coherent_prompt"] for ex in batch]

        clean_tokens, _, clean_cache = run_with_resid_cache(model, clean_prompts)
        swapped_tokens, _, swapped_cache = run_with_resid_cache(model, swapped_prompts)

        for i, ex in enumerate(batch):
            clean_positions = find_positions(
                model,
                clean_tokens[i],
                ex,
                ex["clean_prompt"],
                "clean_coherent",
            )
            swapped_positions = find_positions(
                model,
                swapped_tokens[i],
                ex,
                ex["swapped_coherent_prompt"],
                "swapped_coherent",
            )
            append_resid_sites(clean_lists, model, clean_cache, clean_positions, i)

            site_names = resid_site_names(model)
            for pos_idx, pos_name in enumerate(POSITION_NAMES):
                clean_pos = clean_positions[pos_name]
                swapped_pos = swapped_positions[pos_name]
                for layer_idx, site_name in enumerate(site_names):
                    clean_h = clean_cache[site_name][i, clean_pos, :]
                    swapped_h = swapped_cache[site_name][i, swapped_pos, :]
                    diff_lists[pos_idx][layer_idx].append(
                        (clean_h - swapped_h).detach().cpu()
                    )

    return stack_site_lists(clean_lists), stack_site_lists(diff_lists)


def top_right_singular_vectors(matrix: torch.Tensor, rank: int):
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:rank].contiguous()


def learn_pca_projections(site_mats, rank: int, desc: str):
    projections = []
    for pos_idx, pos_name in enumerate(POSITION_NAMES):
        pos_projections = []
        for layer_idx, matrix in enumerate(tqdm(site_mats[pos_idx], desc=f"{desc} {pos_name}")):
            pos_projections.append(top_right_singular_vectors(matrix, rank))
        projections.append(torch.stack(pos_projections, dim=0))
    return torch.stack(projections, dim=0)


def make_random_projections(reference: torch.Tensor, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    n_pos, n_layers_plus_one, rank, d_model = reference.shape
    out = torch.empty_like(reference)

    for pos_idx in range(n_pos):
        for layer_idx in range(n_layers_plus_one):
            raw = torch.randn(d_model, rank, generator=generator)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            out[pos_idx, layer_idx] = q.T

    return out


def project_site_matrix(matrix: torch.Tensor, projection: torch.Tensor):
    return matrix @ projection.T


def fit_vector_transports(clean_mats, projections, rank: int):
    n_pos, n_layers_plus_one = projections.shape[:2]
    r = torch.empty(n_pos, n_layers_plus_one - 1, rank, rank)
    weights = torch.empty(n_pos, n_layers_plus_one - 1)
    r2 = torch.empty(n_pos, n_layers_plus_one - 1)

    eye = torch.eye(rank)
    for pos_idx in range(n_pos):
        for layer_idx in range(n_layers_plus_one - 1):
            x = project_site_matrix(
                clean_mats[pos_idx][layer_idx],
                projections[pos_idx, layer_idx, :rank],
            )
            y = project_site_matrix(
                clean_mats[pos_idx][layer_idx + 1],
                projections[pos_idx, layer_idx + 1, :rank],
            )

            xtx = x.T @ x
            ytx = y.T @ x
            r_edge = torch.linalg.solve(
                xtx + LAMBDA_RIDGE * eye,
                ytx.T,
            ).T
            y_hat = x @ r_edge.T

            ss_res = (y_hat - y).pow(2).sum()
            ss_tot = (y - y.mean(dim=0, keepdim=True)).pow(2).sum() + 1e-8
            r2_edge = 1.0 - ss_res / ss_tot

            r[pos_idx, layer_idx] = r_edge
            r2[pos_idx, layer_idx] = r2_edge
            weights[pos_idx, layer_idx] = max(float(r2_edge), 0.0)

    return r, weights, r2


def vector_obstruction(site_vectors, projections, transports, weights, rank: int):
    raw = torch.tensor(0.0)
    denom = torch.tensor(0.0)
    n_pos, n_layers_plus_one = projections.shape[:2]

    for pos_idx in range(n_pos):
        germs = []
        for layer_idx in range(n_layers_plus_one):
            h = site_vectors[pos_idx][layer_idx]
            p = projections[pos_idx, layer_idx, :rank]
            s = p @ h
            germs.append(s)
            denom = denom + s.pow(2).sum()

        for layer_idx in range(n_layers_plus_one - 1):
            pred = transports[pos_idx, layer_idx] @ germs[layer_idx]
            residual = pred - germs[layer_idx + 1]
            raw = raw + weights[pos_idx, layer_idx] * residual.pow(2).sum()

    return raw, raw / (denom + 1e-8)


@torch.no_grad()
def score_projection(model, examples, projections, transports, weights, rank: int, label: str):
    rows = []
    site_names = resid_site_names(model)

    for condition, prompt_field in CONDITIONS.items():
        for start in tqdm(
            range(0, len(examples), BATCH_SIZE),
            desc=f"score {label} {condition}",
        ):
            batch = examples[start : start + BATCH_SIZE]
            prompts = [ex[prompt_field] for ex in batch]
            tokens, logits, cache = run_with_resid_cache(model, prompts)

            for i, ex in enumerate(batch):
                prompt = ex[prompt_field]
                positions = find_positions(model, tokens[i], ex, prompt, condition)
                site_vectors = []
                for pos_name in POSITION_NAMES:
                    pos_vectors = []
                    token_pos = positions[pos_name]
                    for site_name in site_names:
                        pos_vectors.append(cache[site_name][i, token_pos, :].detach().cpu())
                    site_vectors.append(pos_vectors)

                raw, normalized = vector_obstruction(
                    site_vectors,
                    projections,
                    transports,
                    weights,
                    rank,
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
                        "projection": label,
                        "rank": rank,
                        "condition": condition,
                        "O_raw": float(raw),
                        "O_norm": float(normalized),
                        "margin_clean_contrast": float(margin.cpu()),
                    }
                )

    return pd.DataFrame(rows)


def conflict_vs_coherent_auc(df: pd.DataFrame) -> float:
    sub = df[df["condition"].isin(COHERENT_GROUPS | CONFLICT_GROUPS)].copy()
    y = sub["condition"].isin(CONFLICT_GROUPS).astype(int).values
    return roc_auc_score(y, sub["O_norm"].values)


def swapped_vs_clean_auc(df: pd.DataFrame) -> float:
    sub = df[df["condition"].isin({"clean_coherent", "swapped_coherent"})].copy()
    y = (sub["condition"] == "swapped_coherent").astype(int).values
    return roc_auc_score(y, sub["O_norm"].values)


def plot_rank16(df: pd.DataFrame, auc: float, save_path: Path):
    plt.figure(figsize=(7, 4.5))
    for condition in CONDITIONS:
        sub = df[(df["projection"] == "ioi_difference") & (df["condition"] == condition)]
        plt.hist(sub["O_norm"], bins=40, alpha=0.45, density=True, label=condition)

    plt.xlabel("low-rank residual normalized obstruction")
    plt.ylabel("density")
    plt.title(f"Low-rank residual rank 16; conflict-vs-coherent AUC={auc:.3f}")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)


def plot_rank_sweep(sweep_df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(5, 4))
    plt.plot(
        sweep_df["rank"],
        sweep_df["conflict_vs_coherent_auc"],
        marker="o",
        label="conflict/malformed > coherent",
    )
    plt.axhline(0.65, linewidth=1, linestyle="--", color="black", label="go/no-go")
    plt.xlabel("rank")
    plt.ylabel("AUC")
    plt.title("Low-rank residual obstruction rank sweep")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)


def save_rank16_transports(projections, transports, weights, r2):
    TRANSPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRANSPORT_PATH.open("w") as f:
        json.dump(
            {
                "rank": RANK,
                "lambda_ridge": LAMBDA_RIDGE,
                "position_names": POSITION_NAMES,
                "projection": "ioi_difference_pca",
                "P": projections[:, :, :RANK].tolist(),
                "R": transports.tolist(),
                "w": weights.tolist(),
                "diagnostics": {"train_r2": r2.tolist()},
            },
            f,
        )


def run_one_rank(model, test_examples, clean_mats, projections, rank: int, label: str):
    transports, weights, r2 = fit_vector_transports(clean_mats, projections, rank)
    df = score_projection(model, test_examples, projections, transports, weights, rank, label)
    return df, transports, weights, r2


def main() -> None:
    train_examples = read_jsonl(TRAIN_PATH)
    test_examples = read_jsonl(TEST_PATH)

    print(f"rank = {RANK}", flush=True)
    print(f"Loaded train/test examples: {len(train_examples)} / {len(test_examples)}")
    print("Loading GPT-2 small...", flush=True)
    model = load_gpt2_small()

    clean_mats, diff_mats = collect_train_site_mats(model, train_examples)
    print("Learning IOI-difference PCA projections...", flush=True)
    ioi_projections = learn_pca_projections(diff_mats, MAX_RANK, "ioi PCA")
    del diff_mats
    gc.collect()

    print("Learning generic residual PCA projections...", flush=True)
    generic_projections = learn_pca_projections(clean_mats, MAX_RANK, "generic PCA")
    random_projections = make_random_projections(ioi_projections)

    rank16_df, rank16_r, rank16_w, rank16_r2 = run_one_rank(
        model,
        test_examples,
        clean_mats,
        ioi_projections,
        RANK,
        "ioi_difference",
    )
    random_df, _, _, _ = run_one_rank(
        model,
        test_examples,
        clean_mats,
        random_projections,
        RANK,
        "random",
    )
    generic_df, _, _, _ = run_one_rank(
        model,
        test_examples,
        clean_mats,
        generic_projections,
        RANK,
        "generic_pca",
    )

    all_rank16 = pd.concat([rank16_df, random_df, generic_df], ignore_index=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rank16.to_csv(CSV_PATH, index=False)
    save_rank16_transports(ioi_projections, rank16_r, rank16_w, rank16_r2)

    rank16_auc = conflict_vs_coherent_auc(rank16_df)
    swapped_auc = swapped_vs_clean_auc(rank16_df)
    random_auc = conflict_vs_coherent_auc(random_df)
    generic_auc = conflict_vs_coherent_auc(generic_df)
    plot_rank16(rank16_df, rank16_auc, FIGURE_PATH)

    print("mean O_norm by group")
    print(rank16_df.groupby("condition")["O_norm"].mean().sort_index())
    print(f"AUC(conflict_or_malformed > coherent) = {rank16_auc:.4f}")
    print(f"AUC(swapped_coherent > clean_coherent) = {swapped_auc:.4f}")
    print(f"random projection AUC = {random_auc:.4f}")
    print(f"generic PCA projection AUC = {generic_auc:.4f}")
    print(f"Saved transports to {TRANSPORT_PATH}")
    print(f"Saved rank 16 results to {CSV_PATH}")
    print(f"Saved rank 16 figure to {FIGURE_PATH}")

    controls_pass = rank16_auc > random_auc and rank16_auc > generic_auc
    swapped_control_pass = swapped_auc <= 0.65
    ready_for_patching = rank16_auc > 0.65 and controls_pass and swapped_control_pass

    if rank16_auc <= 0.65:
        print("Rank 16 failed; running rank sweep r = 4, 8, 16, 32.")
        sweep_rows = []
        for rank in RANK_SWEEP:
            if rank == RANK:
                df_rank = rank16_df
            else:
                df_rank, _, _, _ = run_one_rank(
                    model,
                    test_examples,
                    clean_mats,
                    ioi_projections,
                    rank,
                    "ioi_difference",
                )
            sweep_rows.append(
                {
                    "rank": rank,
                    "conflict_vs_coherent_auc": conflict_vs_coherent_auc(df_rank),
                    "swapped_vs_clean_auc": swapped_vs_clean_auc(df_rank),
                    "mean_clean_coherent": df_rank[
                        df_rank["condition"] == "clean_coherent"
                    ]["O_norm"].mean(),
                    "mean_swapped_coherent": df_rank[
                        df_rank["condition"] == "swapped_coherent"
                    ]["O_norm"].mean(),
                    "mean_conflict": df_rank[df_rank["condition"] == "conflict"][
                        "O_norm"
                    ].mean(),
                    "mean_malformed": df_rank[df_rank["condition"] == "malformed"][
                        "O_norm"
                    ].mean(),
                }
            )

        sweep_df = pd.DataFrame(sweep_rows)
        SWEEP_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(SWEEP_CSV_PATH, index=False)
        plot_rank_sweep(sweep_df, SWEEP_FIGURE_PATH)
        print(sweep_df)
        print(f"Saved rank sweep to {SWEEP_CSV_PATH}")
        print(f"Saved rank sweep figure to {SWEEP_FIGURE_PATH}")
        print(
            "NO-GO if all ranks remain below 0.65: move to attention head and MLP "
            "component sites."
        )
    elif not controls_pass:
        print(
            "CONTROL FAIL: primary AUC passes, but IOI-difference PCA does not beat "
            "random and generic PCA controls. Do not run patching yet."
        )
    elif not swapped_control_pass:
        print(
            "CONTROL FAIL: swapped coherent separates from clean coherent, suggesting "
            "the score detects answer/condition shift rather than obstruction. "
            "Do not run patching yet."
        )
    elif ready_for_patching:
        print("GO: low-rank residual obstruction passes; patching can be implemented next.")


if __name__ == "__main__":
    main()
