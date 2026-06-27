from __future__ import annotations

from pathlib import Path

from src.data_factual import FACTS, make_fact_example, write_jsonl
from src.model import load_gpt2_small


OUT_DIR = Path("data/factual_conflict")
RAW_PATH = OUT_DIR / "factual_conflict_raw.jsonl"


def is_single_token(model, answer: str) -> bool:
    return len(model.tokenizer.encode(answer, add_special_tokens=False)) == 1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 small tokenizer through TransformerLens...", flush=True)
    model = load_gpt2_small(device="cpu")

    rows = []
    dropped = []
    for fact in FACTS:
        ex = make_fact_example(*fact)
        if not is_single_token(model, ex["answer_a"]):
            dropped.append((ex["id"], "answer_a", ex["answer_a"]))
            continue
        if not is_single_token(model, ex["answer_b"]):
            dropped.append((ex["id"], "answer_b", ex["answer_b"]))
            continue

        ex["answer_a_id"] = model.tokenizer.encode(
            ex["answer_a"],
            add_special_tokens=False,
        )[0]
        ex["answer_b_id"] = model.tokenizer.encode(
            ex["answer_b"],
            add_special_tokens=False,
        )[0]
        rows.append(ex)

    write_jsonl(RAW_PATH, rows)

    print(f"input facts: {len(FACTS)}")
    print(f"saved single-token examples: {len(rows)}")
    print(f"dropped multi-token examples: {len(dropped)}")
    for item in dropped[:20]:
        print("drop", item)
    print(f"saved to {RAW_PATH}")


if __name__ == "__main__":
    main()
