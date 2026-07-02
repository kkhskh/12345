from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data_factual import read_jsonl, write_jsonl
from src.model import continuation_logprob_margin, load_pretrained_model


IN_PATH = Path("data/factual_conflict/factual_conflict_raw.jsonl")
SCORED_PATH = Path("data/factual_conflict/factual_conflict_scored.jsonl")
SCORED_CSV_PATH = Path("artifacts/results/factual_conflict_scored_margins.csv")


@torch.no_grad()
def margin(model, prompt: str, answer_a_id: int, answer_b_id: int) -> float:
    toks = model.to_tokens([prompt], prepend_bos=True)
    logits = model(toks)
    return float((logits[0, -1, answer_a_id] - logits[0, -1, answer_b_id]).cpu())


def answer_ids(ex: dict, key: str, fallback_key: str) -> list[int]:
    ids = ex.get(key)
    if ids is not None:
        return [int(token_id) for token_id in ids]
    return [int(ex[fallback_key])]


def score_margin(model, prompt: str, ex: dict, score_type: str) -> float:
    if score_type == "first_token":
        return margin(model, prompt, int(ex["answer_a_id"]), int(ex["answer_b_id"]))
    if score_type == "sequence_logprob":
        return continuation_logprob_margin(
            model,
            prompt,
            answer_ids(ex, "answer_a_ids", "answer_a_id"),
            answer_ids(ex, "answer_b_ids", "answer_b_id"),
        )
    raise ValueError(f"Unsupported score_type: {score_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2-small")
    parser.add_argument(
        "--score-type",
        choices=["sequence_logprob", "first_token"],
        default="sequence_logprob",
        help="Margin score used for factual matching.",
    )
    args = parser.parse_args()

    rows = read_jsonl(IN_PATH)

    print(f"loaded raw factual examples: {len(rows)}", flush=True)
    print(f"Loading {args.model_name}...", flush=True)
    model = load_pretrained_model(args.model_name)

    scored = []
    for ex in rows:
        m_clean = score_margin(model, ex["clean_prompt"], ex, args.score_type)
        m_false = score_margin(model, ex["false_context_prompt"], ex, args.score_type)
        m_corr = score_margin(model, ex["correction_prompt"], ex, args.score_type)

        ex = dict(ex)
        ex["margin_score_type"] = args.score_type
        ex.setdefault("answer_a_token_count", len(answer_ids(ex, "answer_a_ids", "answer_a_id")))
        ex.setdefault("answer_b_token_count", len(answer_ids(ex, "answer_b_ids", "answer_b_id")))
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
    print(f"margin_score_type: {args.score_type}")
    print(
        df[
            [
                "subject",
                "relation",
                "true_object",
                "false_object",
                "answer_a_token_count",
                "answer_b_token_count",
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
