from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.cache import run_with_resid_cache
from src.data_factual import read_jsonl
from src.germs import scalar_germs
from src.model import load_pretrained_model, logit_margin_from_logits
from src.obstruction import compute_obstruction
from src.transports import fit_ridge_scalar_r


MATCHED_PATH = Path("data/factual_conflict/factual_conflict_margin_matched.jsonl")
SCORED_PATH = Path("data/factual_conflict/factual_conflict_scored.jsonl")
CSV_PATH = Path("artifacts/results/factual_margin_matched_obstruction.csv")
FIGURE_PATH = Path("artifacts/figures/factual_margin_matched_obstruction.png")
FACT_KEY = ["subject", "relation", "true_object", "false_object"]


def final_token_pos(model, prompt: str) -> int:
    return len(model.tokenizer.encode(prompt, add_special_tokens=False))


@torch.no_grad()
def collect_clean_scalar_germs(model, examples: list[dict], batch_size: int = 32):
    all_s = []

    for start in tqdm(range(0, len(examples), batch_size), desc="fit clean germs"):
        batch = examples[start : start + batch_size]
        prompts = [ex["clean_prompt"] for ex in batch]
        _, _, cache = run_with_resid_cache(model, prompts)

        for i, ex in enumerate(batch):
            cache_i = {key: value[i : i + 1] for key, value in cache.items()}
            token_pos = final_token_pos(model, ex["clean_prompt"])
            s_i = scalar_germs(
                model,
                cache_i,
                ex["answer_a_id"],
                ex["answer_b_id"],
                token_pos=token_pos,
            )[0]
            all_s.append(s_i.detach().cpu())

    return torch.stack(all_s, dim=0)


@torch.no_grad()
def score_condition(model, examples, prompt_field: str, condition: str, r, w, batch_size=32):
    rows = []

    for start in tqdm(range(0, len(examples), batch_size), desc=condition):
        batch = examples[start : start + batch_size]
        prompts = [ex[prompt_field] for ex in batch]
        _, logits, cache = run_with_resid_cache(model, prompts)

        for i, ex in enumerate(batch):
            cache_i = {key: value[i : i + 1] for key, value in cache.items()}
            token_pos = final_token_pos(model, ex[prompt_field])
            s = scalar_germs(
                model,
                cache_i,
                ex["answer_a_id"],
                ex["answer_b_id"],
                token_pos=token_pos,
            )
            obstruction = compute_obstruction(s, r, w)
            margin = logit_margin_from_logits(
                logits[i : i + 1],
                ex["answer_a_id"],
                ex["answer_b_id"],
                token_pos=token_pos,
            )[0]

            rows.append(
                {
                    "id": ex["id"],
                    "subject": ex["subject"],
                    "relation": ex["relation"],
                    "true_object": ex["true_object"],
                    "false_object": ex["false_object"],
                    "template_id": ex.get("template_id", "unknown"),
                    "condition": condition,
                    "prompt": ex[prompt_field],
                    "margin": float(margin.cpu()),
                    "O_raw": float(obstruction["raw"][0].cpu()),
                    "O_norm": float(obstruction["normalized"][0].cpu()),
                    "answer_a": ex["answer_a"],
                    "answer_b": ex["answer_b"],
                }
            )

    return rows


def obstruction_auc(df: pd.DataFrame) -> float:
    y = (df["condition"] == "conflict_correction").astype(int).values
    return roc_auc_score(y, df["O_norm"].values)


def partial_corr_obstruction_condition_controlling_margin(df: pd.DataFrame) -> float:
    x_margin = df[["margin"]].values
    o = df["O_norm"].values
    y = (df["condition"] == "conflict_correction").astype(float).values

    o_resid = o - LinearRegression().fit(x_margin, o).predict(x_margin)
    y_resid = y - LinearRegression().fit(x_margin, y).predict(x_margin)

    if np.std(o_resid) == 0 or np.std(y_resid) == 0:
        return float("nan")
    return float(np.corrcoef(o_resid, y_resid)[0, 1])


def plot_obstruction(df: pd.DataFrame, auc: float, partial_corr: float, save_path: Path):
    plt.figure(figsize=(6, 4.5))
    for condition in ["clean", "conflict_correction"]:
        sub = df[df["condition"] == condition]
        plt.hist(sub["O_norm"], bins=30, alpha=0.55, density=True, label=condition)
    plt.xlabel("normalized scalar residual-chain obstruction")
    plt.ylabel("density")
    plt.title(f"Factual margin-matched obstruction; AUC={auc:.3f}, pcorr={partial_corr:.3f}")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-dedup-fit-facts",
        action="store_true",
        help="Use all scored clean template rows for transport fitting.",
    )
    parser.add_argument("--model-name", default="gpt2-small")
    args = parser.parse_args()

    matched = read_jsonl(MATCHED_PATH)
    if not matched:
        raise RuntimeError(
            f"No matched examples at {MATCHED_PATH}. Run run_03c first."
        )

    if SCORED_PATH.exists():
        fit_examples = read_jsonl(SCORED_PATH)
    else:
        fit_examples = matched

    if not args.no_dedup_fit_facts:
        fit_examples = (
            pd.DataFrame(fit_examples)
            .sort_values("abs_margin_gap_clean_vs_correction")
            .drop_duplicates(FACT_KEY)
            .to_dict("records")
        )

    print(f"matched rows: {len(matched)}")
    print(f"matched unique facts: {pd.DataFrame(matched)[FACT_KEY].drop_duplicates().shape[0]}")
    print(f"clean fit examples: {len(fit_examples)}")
    print(f"Loading {args.model_name}...", flush=True)
    model = load_pretrained_model(args.model_name)

    s_train = collect_clean_scalar_germs(model, fit_examples, batch_size=args.batch_size)
    r, train_r2, w = fit_ridge_scalar_r(s_train)

    rows = []
    rows.extend(
        score_condition(
            model,
            matched,
            "clean_prompt",
            "clean",
            r,
            w,
            batch_size=args.batch_size,
        )
    )
    rows.extend(
        score_condition(
            model,
            matched,
            "correction_prompt",
            "conflict_correction",
            r,
            w,
            batch_size=args.batch_size,
        )
    )

    df = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    auc = obstruction_auc(df)
    partial_corr = partial_corr_obstruction_condition_controlling_margin(df)
    plot_obstruction(df, auc, partial_corr, FIGURE_PATH)

    print("Transport train R2 by edge:")
    print([round(float(x), 4) for x in train_r2])
    print("Mean obstruction by condition:")
    print(df.groupby("condition")["O_norm"].mean())
    print("Mean margin by condition:")
    print(df.groupby("condition")["margin"].mean())
    print(f"AUC(O_conflict > O_clean | margin-matched): {auc:.4f}")
    print(f"partial corr(O, conflict_label | margin): {partial_corr:.4f}")
    print(f"saved table to {CSV_PATH}")
    print(f"saved figure to {FIGURE_PATH}")

    if auc > 0.65 and partial_corr > 0.20:
        print("GO: factual obstruction smoke diagnostic is promising; expand dataset.")
    else:
        print("NO-GO: factual scalar residual obstruction is not promising yet.")


if __name__ == "__main__":
    main()
