"""Experiment 0B step 1: train a small Transformer on (a + b) mod p and save checkpoints.

Checkpoints are the substrate for the emergence analysis: we later track abstraction strength
A_cyc(t), latent obstruction O_y(t), and test accuracy over the same steps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.toy_modular import (
    ModularConfig,
    abstraction_strength,
    build_model,
    checkpoint_steps,
    final_logits,
    key_frequencies,
    loss_and_accuracy,
    make_dataset,
    save_checkpoint,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modular addition Transformer with checkpoints.")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--n-checkpoints", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default="artifacts/toy_modular/checkpoints")
    parser.add_argument("--output", default="artifacts/toy_modular/training_curve.csv")
    return parser.parse_args()


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import torch
    from tqdm.auto import tqdm

    args = parse_args()
    cfg = ModularConfig(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        train_frac=args.train_frac,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps=args.steps,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    tokens, labels = make_dataset(cfg)
    (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
    train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)
    print(f"p={cfg.p} train={train_tokens.shape[0]} test={test_tokens.shape[0]} device={device}")

    model = build_model(cfg, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )
    save_steps = set(checkpoint_steps(cfg.steps, args.n_checkpoints))
    rows: list[dict] = []

    for step in tqdm(range(cfg.steps), desc="steps"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_acc = loss_and_accuracy(final_logits(model, train_tokens), train_labels)
        train_loss.backward()
        optimizer.step()

        if step in save_steps:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = loss_and_accuracy(
                    final_logits(model, test_tokens), test_labels
                )
                a_cyc = abstraction_strength(model, cfg.p)
                freqs = key_frequencies(model, cfg.p)
            path = Path(args.checkpoint_dir) / f"step_{step:07d}.pt"
            save_checkpoint(model, path, step, cfg)
            rows.append(
                {
                    "step": step,
                    "path": str(path),
                    "train_loss": float(train_loss.detach()),
                    "test_loss": float(test_loss),
                    "train_acc": float(train_acc.detach()),
                    "test_acc": float(test_acc),
                    "abstraction_strength": a_cyc,
                    "key_frequencies": " ".join(str(k) for k in freqs),
                }
            )
            write_csv(args.output, rows)

    write_csv(args.output, rows)
    final = rows[-1] if rows else {}
    print(
        f"final train_acc={final.get('train_acc'):.4f} test_acc={final.get('test_acc'):.4f} "
        f"A_cyc={final.get('abstraction_strength'):.4f}"
        if final
        else "no checkpoints written"
    )
    print(f"saved {len(rows)} checkpoints to {args.checkpoint_dir}")
    print(f"saved training curve to {args.output}")


if __name__ == "__main__":
    main()
