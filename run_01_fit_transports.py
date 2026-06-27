from __future__ import annotations

from pathlib import Path

from src.data_ioi import read_jsonl
from src.model import load_gpt2_small
from src.transports import fit_scalar_transports


TRAIN_PATH = Path("data/processed/ioi_train.jsonl")
VAL_PATH = Path("data/processed/ioi_val.jsonl")
SAVE_PATH = Path("artifacts/transports/scalar_resid_chain_transports.json")


def main() -> None:
    train_examples = read_jsonl(TRAIN_PATH)
    val_examples = read_jsonl(VAL_PATH)

    print(f"Loaded train/val examples: {len(train_examples)} / {len(val_examples)}")
    print("Loading GPT-2 small...")
    model = load_gpt2_small()

    out = fit_scalar_transports(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        lambda_ridge=1e-3,
        batch_size=32,
        save_path=SAVE_PATH,
    )

    train_r2 = out["diagnostics"]["train_r2"]
    val_r2 = out["diagnostics"].get("val_r2", [])
    weights = out["w"]

    print("layer\ttrain_R2\tval_R2\tweight")
    for layer, (tr, wt) in enumerate(zip(train_r2, weights)):
        vr = val_r2[layer] if val_r2 else float("nan")
        print(f"{layer:02d}\t{tr:.4f}\t{vr:.4f}\t{wt:.4f}")

    nonpositive = sum(value <= 0 for value in val_r2)
    if val_r2 and nonpositive > len(val_r2) / 2:
        print(
            "WARNING: most validation R^2 values are non-positive; "
            "scalar transport is likely weak and may need low-rank germs."
        )

    print(f"Saved transports to {SAVE_PATH}")


if __name__ == "__main__":
    main()
