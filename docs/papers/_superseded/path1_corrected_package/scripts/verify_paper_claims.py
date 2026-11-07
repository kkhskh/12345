#!/usr/bin/env python3
"""Claim-language and integrity checks on the Path 1 manuscript. Not a scientific replication."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = (ROOT / "paper_master.md").read_text()
NUM = json.loads((ROOT / "reported_numbers.json").read_text())

FORBIDDEN_PHRASES = [
    "hidden causal substrate",
    "paves the way",
    "sheds light",
    "groundbreaking",
    "we are the first",
    "conclusively proves",
    "obstruction energy",
    "compatibility map explains",
    "Causal Obstruction",
    "fundamentally reshapes",
    "novel paradigm",
]

REQUIRED_LETTERS = [
    "08R2_VALIDATION_COMPLETE — NEGATIVE",
    "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
    "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED",
    "TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED",
]

ALLOWED_CAUSAL = re.compile(
    r"causal (effect|intervention|information|role|summary|description|account|mass|edit|tracing|abstraction|variable)",
    re.I,
)


def main() -> int:
    errors: list[str] = []
    title_line = next(line for line in PAPER.splitlines() if line.startswith("# "))
    if "COR" in title_line or "Obstruction" in title_line:
        errors.append("title contains COR/Obstruction")

    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in PAPER.lower():
            errors.append(f"forbidden phrase: {phrase}")

    for letter in REQUIRED_LETTERS:
        if letter not in PAPER:
            errors.append(f"missing official letter: {letter}")

    if "Y_learned" in PAPER and "-0.58" not in PAPER and "-0.582" not in PAPER:
        errors.append("absolute negative Y not mentioned")

    if NUM["fresh_symmetric"]["both_positive_fraction"] >= 0.60:
        errors.append("numbers file would incorrectly pass switching gate")

    if NUM["vnf"]["i_t"]["value"] >= 0.005:
        errors.append("VNF I_T would incorrectly pass floor")

    abstract = PAPER.split("## 1. Introduction")[0]
    if "Transformer decisions" in abstract and "Othello" not in abstract:
        errors.append("abstract generalizes beyond Othello")

    print("title:", title_line)
    print("forbidden-phrase scan done; required letters:", len(REQUIRED_LETTERS))
    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS (manuscript language/integrity checks)")
    print("NOTE: this does not recompute logits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
