from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _stable_id(*parts: str) -> str:
    raw = "||".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def make_ioi_example(name_a: str, name_b: str, obj: str) -> dict:
    """
    Construct one clean/corrupt IOI repair example.

    Convention:
      Clean prompt:
        name_a and name_b appear in context; name_a is giver;
        clean correct answer is name_b.

      Corrupt prompt:
        same context; name_b is giver;
        corrupt grammatical correct answer is name_a.

      Repair contrast:
        always clean-target repair:
          repair_answer_a = clean_correct = " " + name_b
          repair_answer_b = clean_competitor = " " + name_a

      Therefore repair margin on corrupt is:
          z[" " + name_b] - z[" " + name_a]

      Patching clean activations into corrupt should increase this margin.
    """
    if name_a == name_b:
        raise ValueError("name_a and name_b must differ")

    clean_prompt = (
        f"When {name_a} and {name_b} went to the store, "
        f"{name_a} gave {obj} to"
    )
    corrupt_prompt = (
        f"When {name_a} and {name_b} went to the store, "
        f"{name_b} gave {obj} to"
    )
    conflict_prompt = (
        f"When {name_a} and {name_b} went to the store, "
        f"{name_a} gave {obj} to {name_b}. Actually, {name_a} gave it to"
    )
    malformed_prompt = (
        f"When {name_a} and {name_b} went to the store, "
        f"{name_a} gave {obj} to {name_a}, and then {name_a} gave {obj} to"
    )

    clean_correct = f" {name_b}"
    clean_competitor = f" {name_a}"

    corrupt_correct = f" {name_a}"
    corrupt_competitor = f" {name_b}"

    ex_id = _stable_id(name_a, name_b, obj)

    return {
        "id": ex_id,
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "swapped_coherent_prompt": corrupt_prompt,
        "conflict_prompt": conflict_prompt,
        "malformed_prompt": malformed_prompt,
        "clean_correct": clean_correct,
        "clean_competitor": clean_competitor,
        "corrupt_correct": corrupt_correct,
        "corrupt_competitor": corrupt_competitor,
        # The actual candidate contrast used for repair scoring.
        "repair_answer_a": clean_correct,
        "repair_answer_b": clean_competitor,
        "repair_direction": "corrupt_to_clean_target",
        "candidate_contrast": {
            "a": clean_correct,
            "b": clean_competitor,
            "margin": "z_a_minus_z_b",
            "semantics": "clean_correct_minus_clean_competitor",
        },
        "metadata": {
            "name_a": name_a,
            "name_b": name_b,
            "object": obj,
            "giver_clean": name_a,
            "io_clean": name_b,
            "giver_corrupt": name_b,
            "io_corrupt": name_a,
            "template": "When A and B went to the store, GIVER gave OBJ to",
            "conflict_template": (
                "When A and B went to the store, A gave OBJ to B. "
                "Actually, A gave it to"
            ),
            "malformed_template": (
                "When A and B went to the store, A gave OBJ to A, "
                "and then A gave OBJ to"
            ),
        },
    }


def validate_answer_tokens(model, examples: list[dict]) -> list[dict]:
    """
    Add token IDs and check all candidate answers are single GPT-2 tokens.
    """
    fields = [
        "clean_correct",
        "clean_competitor",
        "corrupt_correct",
        "corrupt_competitor",
        "repair_answer_a",
        "repair_answer_b",
    ]

    for ex in examples:
        for field in fields:
            toks = model.tokenizer.encode(ex[field], add_special_tokens=False)
            if len(toks) != 1:
                raise ValueError(
                    f"{field}={ex[field]!r} is not single-token: {toks}"
                )
            ex[f"{field}_id"] = toks[0]

    return examples


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
