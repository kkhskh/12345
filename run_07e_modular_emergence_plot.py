"""Plot Experiment 0B emergence: A_cyc, O_y, accuracy, probe decode, loss over checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot modular-addition emergence curves.")
    parser.add_argument("--input", default="artifacts/toy_modular/obstruction_over_checkpoints.csv")
    parser.add_argument("--output", default="artifacts/toy_modular/emergence_over_training.png")
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt

    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_07c first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Empty checkpoint CSV.")

    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(steps, [float(row["abstraction_strength"]) for row in rows], label="A_cyc")
    if "decode_acc_test_resid_post" in rows[0]:
        axes[0].plot(steps, [float(row["decode_acc_test_resid_post"]) for row in rows], label="probe decode acc")
    axes[0].plot(steps, [float(row["test_acc"]) for row in rows], label="test acc")
    axes[0].plot(steps, [float(row["train_acc"]) for row in rows], label="train acc", linestyle="--")
    axes[0].set_ylabel("accuracy / abstraction")
    axes[0].legend()
    axes[0].set_title("Representation vs behaviour")

    axes[1].plot(steps, [float(row["O_test_learned"]) for row in rows], label="O learned")
    axes[1].plot(steps, [float(row["O_test_identity"]) for row in rows], label="O identity")
    axes[1].plot(steps, [float(row["O_test_random"]) for row in rows], label="O random")
    axes[1].set_ylabel("normalized obstruction")
    axes[1].legend()
    axes[1].set_title("Compatibility / obstruction")

    if "test_loss" in rows[0]:
        axes[2].plot(steps, [float(row["test_loss"]) for row in rows], label="test loss")
        axes[2].plot(steps, [float(row["train_loss"]) for row in rows], label="train loss", linestyle="--")
    axes[2].plot(steps, [float(row.get("mean_test_margin", "nan")) for row in rows], label="mean test margin")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("loss / margin")
    axes[2].legend()
    axes[2].set_title("Ordinary behavioural scores")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
