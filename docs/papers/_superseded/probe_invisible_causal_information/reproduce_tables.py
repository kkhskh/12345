#!/usr/bin/env python3
"""Regenerate manuscript tables from reported_numbers.json. Read-only on original artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUM = json.loads((ROOT / "reported_numbers.json").read_text())
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)


def write_csv(name: str, rows: list[dict], fieldnames: list[str]) -> None:
    path = TAB / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)


def main() -> None:
    write_csv(
        "table1_primary_contrasts.csv",
        [
            {
                "contrast": "SPEC correct-minus-wrong (equal L2)",
                "n": 48,
                "unit": "recipient-game pair",
                "mean": NUM["spec_l2"]["delta_specific_mean"]["value"],
                "ci95_low": NUM["spec_l2"]["delta_specific_mean"]["ci95"][0],
                "ci95_high": NUM["spec_l2"]["delta_specific_mean"]["ci95"][1],
                "verdict": NUM["spec_l2"]["verdict"],
                "provenance": NUM["spec_l2"]["delta_specific_mean"]["provenance"],
            },
            {
                "contrast": "Fixed-JS correct-minus-wrong (JS=0.01)",
                "n": 48,
                "unit": "recipient-game pair",
                "mean": NUM["fixed_js"]["delta_wrong_mean"]["value"],
                "ci95_low": NUM["fixed_js"]["delta_wrong_mean"]["ci95"][0],
                "ci95_high": NUM["fixed_js"]["delta_wrong_mean"]["ci95"][1],
                "verdict": NUM["fixed_js"]["verdict"],
                "provenance": NUM["fixed_js"]["delta_wrong_mean"]["provenance"],
            },
            {
                "contrast": "Fixed-JS correct-minus-random (JS=0.01)",
                "n": 48,
                "unit": "recipient-game pair",
                "mean": NUM["fixed_js"]["delta_random_mean"]["value"],
                "ci95_low": NUM["fixed_js"]["delta_random_mean"]["ci95"][0],
                "ci95_high": NUM["fixed_js"]["delta_random_mean"]["ci95"][1],
                "verdict": NUM["fixed_js"]["verdict"],
                "provenance": NUM["fixed_js"]["delta_random_mean"]["provenance"],
            },
            {
                "contrast": "NST learned-minus-wrong (JS=0.01)",
                "n": 48,
                "unit": "recipient-game pair",
                "mean": NUM["nst"]["y_learned_minus_wrong"]["mean"],
                "ci95_low": NUM["nst"]["y_learned_minus_wrong"]["ci95"][0],
                "ci95_high": NUM["nst"]["y_learned_minus_wrong"]["ci95"][1],
                "verdict": NUM["nst"]["verdict"],
                "provenance": NUM["nst"]["y_learned_minus_wrong"]["provenance"],
            },
            {
                "contrast": "FRESH A_T natural-anchor FULL",
                "n": 32,
                "unit": "triplet",
                "mean": NUM["fresh_symmetric"]["a_t_mean"]["value"],
                "ci95_low": NUM["fresh_symmetric"]["a_t_mean"]["ci95"][0],
                "ci95_high": NUM["fresh_symmetric"]["a_t_mean"]["ci95"][1],
                "verdict": NUM["fresh_symmetric"]["verdict"],
                "provenance": NUM["fresh_symmetric"]["a_t_mean"]["provenance"],
            },
            {
                "contrast": "FRESH A_W natural-anchor FULL",
                "n": 32,
                "unit": "triplet",
                "mean": NUM["fresh_symmetric"]["a_w_mean"]["value"],
                "ci95_low": NUM["fresh_symmetric"]["a_w_mean"]["ci95"][0],
                "ci95_high": NUM["fresh_symmetric"]["a_w_mean"]["ci95"][1],
                "verdict": NUM["fresh_symmetric"]["verdict"],
                "provenance": NUM["fresh_symmetric"]["a_w_mean"]["provenance"],
            },
            {
                "contrast": "VNF I_T",
                "n": 32,
                "unit": "triplet",
                "mean": NUM["vnf"]["i_t"]["value"],
                "ci95_low": NUM["vnf"]["i_t"]["ci95"][0],
                "ci95_high": NUM["vnf"]["i_t"]["ci95"][1],
                "verdict": NUM["vnf"]["verdict"],
                "provenance": NUM["vnf"]["i_t"]["provenance"],
            },
            {
                "contrast": "VNF I_W",
                "n": 32,
                "unit": "triplet",
                "mean": NUM["vnf"]["i_w"]["value"],
                "ci95_low": NUM["vnf"]["i_w"]["ci95"][0],
                "ci95_high": NUM["vnf"]["i_w"]["ci95"][1],
                "verdict": NUM["vnf"]["verdict"],
                "provenance": NUM["vnf"]["i_w"]["provenance"],
            },
        ],
        ["contrast", "n", "unit", "mean", "ci95_low", "ci95_high", "verdict", "provenance"],
    )

    write_csv(
        "table2_absolute_y_and_stage_share.csv",
        [
            {
                "quantity": "NST mean Y_learned",
                "value": NUM["nst"]["mean_Y_learned"]["value"],
                "note": "negative absolute score",
            },
            {
                "quantity": "NST mean Y_EditorA",
                "value": NUM["nst"]["mean_Y_editor"]["value"],
                "note": "negative absolute score",
            },
            {
                "quantity": "FULL - EditorA",
                "value": NUM["sem"]["mean_level_correction"]["full_minus_editor_a"]["value"],
                "note": NUM["sem"]["mean_level_correction"]["full_minus_editor_a"]["provenance"],
            },
            {
                "quantity": "STAGE_ONLY - EditorA",
                "value": NUM["sem"]["mean_level_correction"]["stage_only_minus_editor_a"]["value"],
                "note": "mean-level identity",
            },
            {
                "quantity": "stage share of FULL-EditorA",
                "value": NUM["sem"]["mean_level_correction"]["stage_share_of_full_minus_editor_a"]["value"],
                "note": "ratio of means, not a causal decomposition",
            },
            {
                "quantity": "FRESH both-positive fraction",
                "value": NUM["fresh_symmetric"]["both_positive_fraction"],
                "note": "13/32; gate 0.60",
            },
        ],
        ["quantity", "value", "note"],
    )

    write_csv(
        "table3_official_letters.csv",
        [{"letter": k, "value": str(v)} for k, v in NUM["frozen_letters"].items()],
        ["letter", "value"],
    )


if __name__ == "__main__":
    main()
