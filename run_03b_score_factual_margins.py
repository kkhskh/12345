from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data_factual import read_jsonl, write_jsonl
from src.model import load_pretrained_model


IN_PATH = Path("data/factual_conflict/factual_conflict_raw.jsonl")
SCORED_PATH = Path("data/factual_conflict/factual_conflict_scored.jsonl")
SCORED_CSV_PATH = Path("artifacts/results/factual_conflict_scored_margins.csv")


@torch.no_grad()
def margin(model, prompt: str, answer_a_id: int, answer_b_id: int) -> float:
    toks = model.to_tokens([prompt], prepend_bos=True)
    logits = model(toks)
    return float((logits[0, -1, answer_a_id] - logits[0, -1, answer_b_id]).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2-small")
    args = parser.parse_args()

    rows = read_jsonl(IN_PATH)

    print(f"loaded raw factual examples: {len(rows)}", flush=True)
    print(f"Loading {args.model_name}...", flush=True)
    model = load_pretrained_model(args.model_name)

    scored = []
    for ex in rows:
        a = ex["answer_a_id"]
        b = ex["answer_b_id"]
        m_clean = margin(model, ex["clean_prompt"], a, b)
        m_false = margin(model, ex["false_context_prompt"], a, b)
        m_corr = margin(model, ex["correction_prompt"], a, b)

        ex = dict(ex)
        ex["margin_clean"] = m_clean
        ex["margin_false_context"] = m_false
        ex["margin_correction"] = m_corr
        ex["abs_margin_gap_clean_vs_false"] = abs(m_clean - m_false)
        ex["abs_margin_gap_clean_vs_correction"] = abs(m_clean - m_corr)
        ex["clean_supports_true"] = m_clean > 0
        ex["false_context_supports_true"] = m_false > 0
        ex["correction_supports_true"] = m_corr > 0
        scored.append(ex)

    write_jsonl(SCORED_PATH, scored)
    SCORED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scored)
    df.to_csv(SCORED_CSV_PATH, index=False)

    print(f"saved scored JSONL to {SCORED_PATH}")
    print(f"saved scored CSV to {SCORED_CSV_PATH}")
    print(
        df[
            [
                "subject",
                "relation",
                "true_object",
                "false_object",
                "margin_clean",
                "margin_correction",
                "abs_margin_gap_clean_vs_correction",
            ]
        ]
        .sort_values("abs_margin_gap_clean_vs_correction")
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
