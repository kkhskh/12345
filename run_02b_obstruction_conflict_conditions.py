from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.cache import run_with_resid_cache
from src.data_ioi import read_jsonl
from src.germs import scalar_germs
from src.model import load_gpt2_small, logit_margin_from_logits
from src.obstruction import compute_obstruction


TEST_PATH = Path("data/processed/ioi_test.jsonl")
TRANSPORT_PATH = Path("artifacts/transports/scalar_resid_chain_transports.json")
CSV_PATH = Path("artifacts/results/obstruction_conflict_conditions.csv")
FIGURE_PATH = Path("artifacts/figures/obstruction_conflict_conditions.png")

CONDITIONS = {
    "clean_coherent": "clean_prompt",
    "swapped_coherent": "swapped_coherent_prompt",
    "conflict": "conflict_prompt",
    "malformed": "malformed_prompt",
}
COHERENT_GROUPS = {"clean_coherent", "swapped_coherent"}
CONFLICT_GROUPS = {"conflict", "malformed"}


@torch.no_grad()
def score_condition(model, examples, condition: str, prompt_field: str, r, w, batch_size=32):
    rows = []

    for start in tqdm(range(0, len(examples), batch_size), desc=condition):
        batch = examples[start : start + batch_size]
        prompts = [ex[prompt_field] for ex in batch]
        _, logits, cache = run_with_resid_cache(model, prompts)

        for i, ex in enumerate(batch):
            cache_i = {key: value[i : i + 1] for key, value in cache.items()}
            answer_a_id = ex["repair_answer_a_id"]
            answer_b_id = ex["repair_answer_b_id"]

            s = scalar_germs(
                model,
                cache_i,
                answer_a_id,
                answer_b_id,
            )
            obstruction = compute_obstruction(s, r, w)
            margin = logit_margin_from_logits(
                logits[i : i + 1],
                answer_a_id,
                answer_b_id,
            )[0]

            rows.append(
                {
                    "id": ex["id"],
                    "split": ex["split"],
                    "condition": condition,
                    "prompt": ex[prompt_field],
                    "O_raw": float(obstruction["raw"][0].cpu()),
                    "O_norm": float(obstruction["normalized"][0].cpu()),
                    "margin_clean_contrast": float(margin.cpu()),
                    "clean_correct": ex["clean_correct"],
                    "clean_competitor": ex["clean_competitor"],
                }
            )

    return rows


def conflict_vs_coherent_auc(df: pd.DataFrame) -> float:
    sub = df[df["condition"].isin(COHERENT_GROUPS | CONFLICT_GROUPS)].copy()
    y = sub["condition"].isin(CONFLICT_GROUPS).astype(int).values
    scores = sub["O_norm"].values
    return roc_auc_score(y, scores)


def plot_condition_obstructions(df: pd.DataFrame, auc: float, save_path: Path) -> None:
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

    plt.xlabel("normalized obstruction")
    plt.ylabel("density")
    plt.title(f"IOI obstruction by condition; conflict-vs-coherent AUC={auc:.3f}")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)


def main() -> None:
    examples = read_jsonl(TEST_PATH)
    missing_fields = {
        field
        for field in CONDITIONS.values()
        for ex in examples[:5]
        if field not in ex
    }
    if missing_fields:
        raise RuntimeError(
            "Dataset is missing condition fields "
            f"{sorted(missing_fields)}. Rerun run_00_generate_ioi_data.py."
        )

    with TRANSPORT_PATH.open() as f:
        transports = json.load(f)

    print(f"Loaded test examples: {len(examples)}", flush=True)
    print("Loading GPT-2 small...", flush=True)
    model = load_gpt2_small()

    rows = []
    for condition, prompt_field in CONDITIONS.items():
        rows.extend(
            score_condition(
                model=model,
                examples=examples,
                condition=condition,
                prompt_field=prompt_field,
                r=transports["r"],
                w=transports["w"],
            )
        )

    df = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    auc = conflict_vs_coherent_auc(df)
    plot_condition_obstructions(df, auc, FIGURE_PATH)

    print("Mean normalized obstruction by condition:")
    print(df.groupby("condition")["O_norm"].mean().sort_index())
    print(f"AUC(conflict_or_malformed > coherent): {auc:.4f}")
    print(f"Saved table to {CSV_PATH}")
    print(f"Saved figure to {FIGURE_PATH}")

    if auc > 0.65:
        print("GO: conflict/malformed obstruction separates from coherent controls.")
    else:
        print(
            "NO-GO: conflict/malformed obstruction does not separate from coherent "
            "controls. Next revision should upgrade the representation."
        )


if __name__ == "__main__":
    main()
