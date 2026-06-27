from __future__ import annotations

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
from src.germs import unembed_contrast_vector
from src.model import load_gpt2_small, logit_margin_from_logits
from src.transports import fit_ridge_scalar_r


TRAIN_PATH = Path("data/processed/ioi_train.jsonl")
TEST_PATH = Path("data/processed/ioi_test.jsonl")
TRANSPORT_PATH = Path("artifacts/transports/multitoken_scalar_transports.json")
CSV_PATH = Path("artifacts/results/multitoken_scalar_obstruction.csv")
FIGURE_PATH = Path("artifacts/figures/multitoken_scalar_obstruction.png")

POSITION_NAMES = ["first_name", "second_name", "giver_name", "final_token"]
CONDITIONS = {
    "clean_coherent": "clean_prompt",
    "swapped_coherent": "swapped_coherent_prompt",
    "conflict": "conflict_prompt",
    "malformed": "malformed_prompt",
}
COHERENT_GROUPS = {"clean_coherent", "swapped_coherent"}
CONFLICT_GROUPS = {"conflict", "malformed"}


def token_len(model, prompt: str) -> int:
    # Add one for TransformerLens' BOS token.
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


def scalar_chain_for_positions(model, cache, ex: dict, positions: dict[str, int]):
    du = unembed_contrast_vector(
        model,
        ex["repair_answer_a_id"],
        ex["repair_answer_b_id"],
    )

    chains = []
    for position_name in POSITION_NAMES:
        pos = positions[position_name]
        vals = []
        for site_name in resid_site_names(model):
            vals.append(cache[site_name][:, pos, :] @ du)
        chains.append(torch.stack(vals, dim=1).squeeze(0))

    return torch.stack(chains, dim=0)


@torch.no_grad()
def collect_clean_multitoken_germs(model, examples, batch_size: int = 32):
    all_s = []

    for start in tqdm(range(0, len(examples), batch_size), desc="fit clean germs"):
        batch = examples[start : start + batch_size]
        prompts = [ex["clean_prompt"] for ex in batch]
        tokens, _, cache = run_with_resid_cache(model, prompts)

        for i, ex in enumerate(batch):
            cache_i = {key: value[i : i + 1] for key, value in cache.items()}
            positions = find_positions(
                model,
                tokens[i],
                ex,
                ex["clean_prompt"],
                "clean_coherent",
            )
            all_s.append(scalar_chain_for_positions(model, cache_i, ex, positions).cpu())

    return torch.stack(all_s, dim=0)


def fit_multitoken_transports(s: torch.Tensor, lambda_ridge: float = 1e-3):
    r_rows = []
    r2_rows = []
    w_rows = []

    for pos_idx in range(s.shape[1]):
        r, r2, w = fit_ridge_scalar_r(s[:, pos_idx, :], lambda_ridge=lambda_ridge)
        r_rows.append(r)
        r2_rows.append(r2)
        w_rows.append(w)

    return torch.stack(r_rows), torch.stack(r2_rows), torch.stack(w_rows)


def compute_multitoken_obstruction(s: torch.Tensor, r, w, eps: float = 1e-8):
    r = torch.as_tensor(r, device=s.device, dtype=s.dtype)
    w = torch.as_tensor(w, device=s.device, dtype=s.dtype)

    pred_next = s[:, :-1] * r
    actual_next = s[:, 1:]
    weighted_terms = w * (pred_next - actual_next).pow(2)

    raw = weighted_terms.sum()
    normalized = raw / (s.pow(2).sum() + eps)
    return raw, normalized


@torch.no_grad()
def score_condition(model, examples, condition: str, prompt_field: str, r, w, batch_size=32):
    rows = []

    for start in tqdm(range(0, len(examples), batch_size), desc=condition):
        batch = examples[start : start + batch_size]
        prompts = [ex[prompt_field] for ex in batch]
        tokens, logits, cache = run_with_resid_cache(model, prompts)

        for i, ex in enumerate(batch):
            prompt = ex[prompt_field]
            cache_i = {key: value[i : i + 1] for key, value in cache.items()}
            positions = find_positions(model, tokens[i], ex, prompt, condition)
            s = scalar_chain_for_positions(model, cache_i, ex, positions)
            raw, normalized = compute_multitoken_obstruction(s, r, w)

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
                    "condition": condition,
                    "prompt": prompt,
                    "O_raw": float(raw.cpu()),
                    "O_norm": float(normalized.cpu()),
                    "margin_clean_contrast": float(margin.cpu()),
                    **{f"pos_{name}": positions[name] for name in POSITION_NAMES},
                }
            )

    return rows


def condition_auc(df: pd.DataFrame) -> float:
    sub = df[df["condition"].isin(COHERENT_GROUPS | CONFLICT_GROUPS)].copy()
    y = sub["condition"].isin(CONFLICT_GROUPS).astype(int).values
    return roc_auc_score(y, sub["O_norm"].values)


def plot_obstruction_by_condition(df: pd.DataFrame, auc: float, save_path: Path):
    plt.figure(figsize=(7, 4.5))
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        plt.hist(
            sub["O_norm"],
            bins=40,
            alpha=0.45,
            density=True,
            label=condition,
        )

    plt.xlabel("multi-token normalized obstruction")
    plt.ylabel("density")
    plt.title(f"Multi-token scalar obstruction; conflict-vs-coherent AUC={auc:.3f}")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)


def main() -> None:
    train_examples = read_jsonl(TRAIN_PATH)
    test_examples = read_jsonl(TEST_PATH)

    print(
        f"Loaded train/test examples: {len(train_examples)} / {len(test_examples)}",
        flush=True,
    )
    print("Loading GPT-2 small...", flush=True)
    model = load_gpt2_small()

    s_train = collect_clean_multitoken_germs(model, train_examples)
    r, train_r2, w = fit_multitoken_transports(s_train)

    TRANSPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRANSPORT_PATH.open("w") as f:
        json.dump(
            {
                "position_names": POSITION_NAMES,
                "r": r.tolist(),
                "w": w.tolist(),
                "diagnostics": {"train_r2": train_r2.tolist()},
            },
            f,
            indent=2,
        )

    print("Mean train R^2 by position:")
    for pos_name, r2_row in zip(POSITION_NAMES, train_r2):
        print(f"  {pos_name}: {float(r2_row.mean()):.4f}")

    rows = []
    for condition, prompt_field in CONDITIONS.items():
        rows.extend(
            score_condition(
                model,
                test_examples,
                condition,
                prompt_field,
                r,
                w,
            )
        )

    df = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    auc = condition_auc(df)
    plot_obstruction_by_condition(df, auc, FIGURE_PATH)

    print("Mean clean-target margin by condition:")
    print(df.groupby("condition")["margin_clean_contrast"].mean().sort_index())
    print("Mean multi-token normalized obstruction by condition:")
    print(df.groupby("condition")["O_norm"].mean().sort_index())
    print(f"AUC(conflict_or_malformed > coherent): {auc:.4f}")
    print(f"Saved transports to {TRANSPORT_PATH}")
    print(f"Saved table to {CSV_PATH}")
    print(f"Saved figure to {FIGURE_PATH}")

    if auc > 0.65:
        print("GO: multi-token scalar obstruction separates conflict from coherent.")
    else:
        print(
            "NO-GO: multi-token scalar obstruction does not separate. "
            "Next step is low-rank residual vector germs."
        )


if __name__ == "__main__":
    main()
