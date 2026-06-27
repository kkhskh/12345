from __future__ import annotations

import json
from pathlib import Path

from src.data_ioi import read_jsonl
from src.model import load_gpt2_small
from src.plotting import (
    obstruction_clean_corrupt_table,
    plot_clean_vs_corrupt_obstruction,
)


TEST_PATH = Path("data/processed/ioi_test.jsonl")
TRANSPORT_PATH = Path("artifacts/transports/scalar_resid_chain_transports.json")
CSV_PATH = Path("artifacts/results/experiment_0_coherent_swapped_control.csv")
FIGURE_PATH = Path("artifacts/figures/experiment_0_coherent_swapped_control.png")


def main() -> None:
    examples = read_jsonl(TEST_PATH)
    with TRANSPORT_PATH.open() as f:
        transports = json.load(f)

    print(f"Loaded test examples: {len(examples)}")
    print("Loading GPT-2 small...")
    model = load_gpt2_small()

    df = obstruction_clean_corrupt_table(
        model=model,
        examples=examples,
        r=transports["r"],
        w=transports["w"],
    )
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    auc = plot_clean_vs_corrupt_obstruction(df, save_path=FIGURE_PATH)
    print(
        "Experiment 0: coherent-clean vs coherent-swapped IOI control",
        flush=True,
    )
    print(f"AUC(swapped coherent > clean coherent): {auc:.4f}")
    print(f"Saved table to {CSV_PATH}")
    print(f"Saved figure to {FIGURE_PATH}")

    if auc <= 0.65:
        print(
            "Control conclusion: scalar residual-chain obstruction does not "
            "distinguish two coherent but opposite IOI computations."
        )
    else:
        print("Unexpected: coherent opposite decisions separate under obstruction.")


if __name__ == "__main__":
    main()
