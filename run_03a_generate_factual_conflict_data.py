from __future__ import annotations

import argparse
from pathlib import Path

from src.data_factual import FACTS, make_fact_examples, write_jsonl
from src.model import answer_token_ids, load_pretrained_model


OUT_DIR = Path("data/factual_conflict")
RAW_PATH = OUT_DIR / "factual_conflict_raw.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2-small")
    parser.add_argument(
        "--single-token-only",
        action="store_true",
        help="Legacy mode: drop facts whose answers are not single-token continuations.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_name} tokenizer through TransformerLens...", flush=True)
    model = load_pretrained_model(args.model_name, device="cpu")

    rows = []
    dropped = []
    for fact in FACTS:
        examples = make_fact_examples(*fact)
        answer_a_ids = answer_token_ids(model, examples[0]["answer_a"])
        answer_b_ids = answer_token_ids(model, examples[0]["answer_b"])

        if args.single_token_only and len(answer_a_ids) != 1:
            dropped.append((examples[0]["id"], "answer_a", examples[0]["answer_a"], answer_a_ids))
            continue
        if args.single_token_only and len(answer_b_ids) != 1:
            dropped.append((examples[0]["id"], "answer_b", examples[0]["answer_b"], answer_b_ids))
            continue

        for ex in examples:
            ex["answer_a_ids"] = answer_a_ids
            ex["answer_b_ids"] = answer_b_ids
            ex["answer_a_token_count"] = len(answer_a_ids)
            ex["answer_b_token_count"] = len(answer_b_ids)
            # Backward-compatible first-token contrast for current scalar germs.
            ex["answer_a_id"] = answer_a_ids[0]
            ex["answer_b_id"] = answer_b_ids[0]
            ex["germ_answer_token_index"] = 0
            rows.append(ex)

    write_jsonl(RAW_PATH, rows)

    print(f"input facts: {len(FACTS)}")
    print(f"saved template rows: {len(rows)}")
    print(
        "unique saved facts: "
        f"{len({(row['subject'], row['relation'], row['true_object'], row['false_object']) for row in rows})}"
    )
    print(f"single_token_only: {args.single_token_only}")
    print(f"dropped facts: {len(dropped)}")
    for item in dropped[:20]:
        print("drop", item)
    print(f"saved to {RAW_PATH}")


if __name__ == "__main__":
    main()
