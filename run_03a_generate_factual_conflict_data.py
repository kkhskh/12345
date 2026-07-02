from __future__ import annotations

import argparse
from pathlib import Path

from src.data_factual import FACTS, make_fact_examples, write_jsonl
from src.model import load_pretrained_model


OUT_DIR = Path("data/factual_conflict")
RAW_PATH = OUT_DIR / "factual_conflict_raw.jsonl"


def is_single_token(model, answer: str) -> bool:
    return len(model.tokenizer.encode(answer, add_special_tokens=False)) == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2-small")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_name} tokenizer through TransformerLens...", flush=True)
    model = load_pretrained_model(args.model_name, device="cpu")

    rows = []
    dropped = []
    for fact in FACTS:
        examples = make_fact_examples(*fact)
        if not is_single_token(model, examples[0]["answer_a"]):
            dropped.append((examples[0]["id"], "answer_a", examples[0]["answer_a"]))
            continue
        if not is_single_token(model, examples[0]["answer_b"]):
            dropped.append((examples[0]["id"], "answer_b", examples[0]["answer_b"]))
            continue

        answer_a_id = model.tokenizer.encode(
            examples[0]["answer_a"],
            add_special_tokens=False,
        )[0]
        answer_b_id = model.tokenizer.encode(
            examples[0]["answer_b"],
            add_special_tokens=False,
        )[0]

        for ex in examples:
            ex["answer_a_id"] = answer_a_id
            ex["answer_b_id"] = answer_b_id
            rows.append(ex)

    write_jsonl(RAW_PATH, rows)

    print(f"input facts: {len(FACTS)}")
    print(f"saved single-token template rows: {len(rows)}")
    print(f"unique single-token facts: {len({row['subject'] for row in rows})}")
    print(f"dropped multi-token facts: {len(dropped)}")
    for item in dropped[:20]:
        print("drop", item)
    print(f"saved to {RAW_PATH}")


if __name__ == "__main__":
    main()
