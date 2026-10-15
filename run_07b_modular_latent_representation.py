"""Experiment 0B step 2: does the model internally represent the known cyclic latent state?

This is the representation-detection stage the updated plan requires before any steering.
Germs P^y_v are fit on training examples only and scored on held-out examples against the
ground-truth latent state from the task simulator, alongside shuffled-target and
random-subspace controls so that a high probe score cannot come from probe capacity alone.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.toy_modular import (
    abstraction_strength,
    apply_ridge,
    cache_site_activations,
    decode_sum_from_latent,
    fit_ridge,
    key_frequencies,
    latent_targets,
    load_checkpoint,
    make_dataset,
    make_toy_sites,
    r_squared,
    random_subspace,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect cyclic latent-state representations per site.")
    parser.add_argument("--checkpoint-dir", default="artifacts/toy_modular/checkpoints")
    parser.add_argument("--curve", default="artifacts/toy_modular/training_curve.csv")
    parser.add_argument("--output", default="artifacts/toy_modular/latent_probe_metrics.csv")
    parser.add_argument("--n-freqs", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 means use all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
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


def select_checkpoints(checkpoint_dir: str, max_checkpoints: int) -> list[Path]:
    paths = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}. Run run_07a first.")
    if max_checkpoints and len(paths) > max_checkpoints:
        stride = len(paths) / max_checkpoints
        chosen = {paths[min(len(paths) - 1, int(i * stride))] for i in range(max_checkpoints)}
        chosen.add(paths[-1])
        return sorted(chosen)
    return paths


def main() -> None:
    import torch
    from tqdm.auto import tqdm

    args = parse_args()
    rows: list[dict] = []
    paths = select_checkpoints(args.checkpoint_dir, args.max_checkpoints)

    for path in tqdm(paths, desc="checkpoints"):
        model, cfg, step = load_checkpoint(path, device=args.device)
        tokens, labels = make_dataset(cfg)
        (train_tokens, _), (test_tokens, _) = split_dataset(cfg, tokens, labels)
        device = next(model.parameters()).device
        train_tokens = train_tokens.to(device)
        test_tokens = test_tokens.to(device)

        freqs = key_frequencies(model, cfg.p, top_k=args.n_freqs)
        a_cyc = abstraction_strength(model, cfg.p, top_k=args.n_freqs)
        train_latent = latent_targets(train_tokens, cfg.p, freqs).cpu()
        test_latent = latent_targets(test_tokens, cfg.p, freqs).cpu()
        true_test_sum = ((test_tokens[:, 0] + test_tokens[:, 1]) % cfg.p).cpu()

        sites = make_toy_sites(cfg)
        train_acts = cache_site_activations(model, train_tokens, sites)
        test_acts = cache_site_activations(model, test_tokens, sites)

        shuffle_idx = torch.randperm(train_latent.shape[0], generator=torch.Generator().manual_seed(args.seed))
        shuffled_latent = train_latent[shuffle_idx]

        for site in sites:
            X_train, X_test = train_acts[site.site_id], test_acts[site.site_id]
            germ = fit_ridge(X_train, train_latent, args.ridge)
            pred_test = apply_ridge(germ, X_test)
            decoded = decode_sum_from_latent(pred_test, cfg.p, freqs)

            germ_shuffled = fit_ridge(X_train, shuffled_latent, args.ridge)
            pred_shuffled = apply_ridge(germ_shuffled, X_test)
            decoded_shuffled = decode_sum_from_latent(pred_shuffled, cfg.p, freqs)

            basis = random_subspace(X_train.shape[1], train_latent.shape[1], args.seed)
            germ_random = fit_ridge(X_train.float() @ basis, train_latent, args.ridge)
            pred_random = apply_ridge(germ_random, X_test.float() @ basis)
            decoded_random = decode_sum_from_latent(pred_random, cfg.p, freqs)

            rows.append(
                {
                    "step": step,
                    "abstraction_strength": a_cyc,
                    "key_frequencies": " ".join(str(k) for k in freqs),
                    **site.to_row(),
                    "d_site": int(X_train.shape[1]),
                    "r2_test": r_squared(pred_test, test_latent),
                    "r2_train": r_squared(apply_ridge(germ, X_train), train_latent),
                    "decode_acc_test": float((decoded == true_test_sum).float().mean()),
                    "r2_test_shuffled_control": r_squared(pred_shuffled, test_latent),
                    "decode_acc_shuffled_control": float((decoded_shuffled == true_test_sum).float().mean()),
                    "r2_test_random_subspace": r_squared(pred_random, test_latent),
                    "decode_acc_random_subspace": float((decoded_random == true_test_sum).float().mean()),
                    "chance_decode_acc": 1.0 / cfg.p,
                }
            )
        write_csv(args.output, rows)

    write_csv(args.output, rows)
    final_step = max(row["step"] for row in rows)
    final_rows = sorted(
        (row for row in rows if row["step"] == final_step),
        key=lambda row: row["decode_acc_test"],
        reverse=True,
    )
    print(f"\nFinal checkpoint (step {final_step}) latent decoding by site:")
    for row in final_rows[:10]:
        print(
            f"  {row['site_id']:>18}: R2_test={row['r2_test']:+.4f} "
            f"decode_acc={row['decode_acc_test']:.4f} "
            f"(shuffled {row['decode_acc_shuffled_control']:.4f}, "
            f"random-subspace {row['decode_acc_random_subspace']:.4f}, "
            f"chance {row['chance_decode_acc']:.4f})"
        )
    print(f"saved probe metrics to {args.output}")


if __name__ == "__main__":
    main()
