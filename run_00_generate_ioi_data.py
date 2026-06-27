from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.data_ioi import make_ioi_example, validate_answer_tokens, write_jsonl
from src.model import load_gpt2_small


SEED = 12345
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

NAMES = [
    "John",
    "Mary",
    "Tom",
    "Sarah",
    "James",
    "Emily",
    "Robert",
    "Linda",
    "Michael",
    "Jessica",
    "David",
    "Susan",
    "William",
    "Karen",
    "Richard",
    "Nancy",
    "Joseph",
    "Lisa",
    "Thomas",
    "Betty",
    "Charles",
    "Sandra",
    "Daniel",
    "Ashley",
]

OBJECTS = [
    "a bottle of milk",
    "a book",
    "a laptop",
    "a football",
    "a cup of tea",
    "a bag of apples",
    "a bouquet of flowers",
    "a box of chocolates",
    "a set of keys",
    "a sandwich",
]


def is_single_gpt2_answer_token(model, name: str) -> bool:
    toks = model.tokenizer.encode(" " + name, add_special_tokens=False)
    return len(toks) == 1


def token_len(model, text: str) -> int:
    return len(model.tokenizer.encode(text, add_special_tokens=False))


def add_pair_disjoint_splits(examples: list[dict], rng: random.Random) -> list[dict]:
    pairs = sorted(
        {
            (ex["metadata"]["giver_clean"], ex["metadata"]["io_clean"])
            for ex in examples
        }
    )
    rng.shuffle(pairs)

    n = len(pairs)
    train_pairs = set(pairs[: int(0.70 * n)])
    val_pairs = set(pairs[int(0.70 * n) : int(0.85 * n)])

    for ex in examples:
        pair = (ex["metadata"]["giver_clean"], ex["metadata"]["io_clean"])
        if pair in train_pairs:
            ex["split"] = "train"
        elif pair in val_pairs:
            ex["split"] = "val"
        else:
            ex["split"] = "test"

    return examples


def main() -> None:
    rng = random.Random(SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 small tokenizer through TransformerLens...", flush=True)
    model = load_gpt2_small(device="cpu")

    valid_names = [name for name in NAMES if is_single_gpt2_answer_token(model, name)]
    invalid_names = [name for name in NAMES if name not in valid_names]
    print(f"Valid single-token names ({len(valid_names)}): {valid_names}", flush=True)
    print(f"Invalid names ({len(invalid_names)}): {invalid_names}", flush=True)
    if len(valid_names) < 16:
        raise RuntimeError("Need at least 16 single-token GPT-2 names.")

    examples = []
    for name_a in valid_names:
        for name_b in valid_names:
            if name_a == name_b:
                continue
            for obj in OBJECTS:
                examples.append(make_ioi_example(name_a, name_b, obj))
    rng.shuffle(examples)

    bad_lengths = []
    for ex in examples:
        clean_len = token_len(model, ex["clean_prompt"])
        corrupt_len = token_len(model, ex["corrupt_prompt"])
        if clean_len != corrupt_len:
            bad_lengths.append((ex["id"], clean_len, corrupt_len))
    if bad_lengths:
        raise RuntimeError(
            f"Found clean/corrupt token-length mismatches: {bad_lengths[:5]}"
        )

    validate_answer_tokens(model, examples)
    add_pair_disjoint_splits(examples, rng)

    train = [ex for ex in examples if ex["split"] == "train"]
    val = [ex for ex in examples if ex["split"] == "val"]
    test = [ex for ex in examples if ex["split"] == "test"]

    write_jsonl(RAW_DIR / "ioi_examples.jsonl", examples)
    write_jsonl(PROC_DIR / "ioi_train.jsonl", train)
    write_jsonl(PROC_DIR / "ioi_val.jsonl", val)
    write_jsonl(PROC_DIR / "ioi_test.jsonl", test)
    pd.DataFrame(examples).to_parquet(PROC_DIR / "ioi_examples.parquet", index=False)

    print(f"Total examples: {len(examples)}", flush=True)
    print(f"Train/val/test: {len(train)} / {len(val)} / {len(test)}", flush=True)
    print(
        "Answer-token sanity: all checked fields are exactly one GPT-2 token.",
        flush=True,
    )
    print(
        "Prompt-length sanity: every clean/corrupt pair has equal token length.",
        flush=True,
    )
    if len(examples) < 1000:
        raise RuntimeError(f"Expected at least 1,000 examples, got {len(examples)}.")


if __name__ == "__main__":
    main()
