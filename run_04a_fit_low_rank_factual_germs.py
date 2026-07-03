from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.low_rank_germs import (
    autodetect_factual_input,
    build_answer_augmented_clean_pca_germs,
    build_clean_pca_germs,
    build_delta_pca_diagnostic_germs,
    load_factual_examples,
    project_activations,
    run_resid_final_token_cache,
    safe_model_name,
    split_examples,
)
from src.vector_transports import fit_ridge_matrix_transport, make_edge_weights, transport_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit low-rank factual germs and clean transports.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--germ-variants",
        nargs="+",
        default=["clean_pca", "answer_augmented_clean_pca", "delta_pca_diagnostic"],
        choices=["clean_pca", "answer_augmented_clean_pca", "delta_pca_diagnostic"],
    )
    parser.add_argument("--lambda-ridge", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="artifacts/factual_low_rank")
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def examples_for_split(examples: list[dict], split_by_id: dict[str, str], split: str) -> list[dict]:
    return [ex for ex in examples if split_by_id[ex["id"]] == split]


def build_variant(model, variant: str, rank: int, train_examples, H_clean_train, H_conflict_train):
    if variant == "clean_pca":
        return build_clean_pca_germs(H_clean_train, rank)
    if variant == "answer_augmented_clean_pca":
        return build_answer_augmented_clean_pca_germs(model, train_examples, H_clean_train, rank)
    if variant == "delta_pca_diagnostic":
        return build_delta_pca_diagnostic_germs(H_clean_train, H_conflict_train, rank)
    raise ValueError(f"Unsupported germ variant: {variant}")


def main() -> None:
    args = parse_args()
    input_path = args.input or autodetect_factual_input()
    examples = load_factual_examples(input_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if len(examples) < 4:
        raise RuntimeError("Need at least 4 examples for train/val/test splitting.")

    split_by_id = split_examples(examples, args.train_frac, args.val_frac, args.seed)
    for ex in examples:
        ex["split"] = split_by_id[ex["id"]]

    train_examples = examples_for_split(examples, split_by_id, "train")
    val_examples = examples_for_split(examples, split_by_id, "val")
    if not train_examples or not val_examples:
        raise RuntimeError("Train and validation splits must both be non-empty.")

    print(f"input: {input_path}")
    print(f"examples: {len(examples)} train={len(train_examples)} val={len(val_examples)}")
    print(f"Loading {args.model}...")
    from src.model import load_pretrained_model

    model = load_pretrained_model(args.model, device=args.device)
    model_slug = safe_model_name(args.model)

    H_clean_train = run_resid_final_token_cache(
        model, [ex["clean_prompt"] for ex in train_examples], args.batch_size, args.device
    )
    H_conflict_train = run_resid_final_token_cache(
        model, [ex["conflict_prompt"] for ex in train_examples], args.batch_size, args.device
    )
    H_clean_val = run_resid_final_token_cache(
        model, [ex["clean_prompt"] for ex in val_examples], args.batch_size, args.device
    )

    output_dir = Path(args.output_dir)
    split_rows = [{"id": ex["id"], "split": ex["split"]} for ex in examples]
    write_csv(output_dir / "configs" / f"{model_slug}_split_assignments.csv", split_rows)

    for variant in args.germ_variants:
        for rank in args.ranks:
            print(f"Fitting {variant} rank={rank}", flush=True)
            mu, P, explained = build_variant(
                model, variant, rank, train_examples, H_clean_train, H_conflict_train
            )
            S_train = project_activations(H_clean_train, mu, P)
            S_val = project_activations(H_clean_val, mu, P)
            R = fit_ridge_matrix_transport(S_train, lambda_ridge=args.lambda_ridge)
            train_r2 = transport_r2(S_train, R)
            val_r2 = transport_r2(S_val, R)
            weights = make_edge_weights(val_r2)

            projection_path = output_dir / "projections" / f"{model_slug}_{variant}_r{rank}.pt"
            transport_path = output_dir / "transports" / f"{model_slug}_{variant}_r{rank}.pt"
            diagnostics_path = output_dir / "diagnostics" / f"{model_slug}_{variant}_r{rank}_transport_diagnostics.csv"
            config_path = output_dir / "configs" / f"{model_slug}_{variant}_r{rank}_config.json"
            projection_path.parent.mkdir(parents=True, exist_ok=True)
            transport_path.parent.mkdir(parents=True, exist_ok=True)

            import torch

            torch.save(
                {
                    "model": args.model,
                    "variant": variant,
                    "rank": rank,
                    "mu_by_layer": mu,
                    "P_by_layer": P,
                    "explained_variance": explained,
                },
                projection_path,
            )
            torch.save(
                {
                    "model": args.model,
                    "variant": variant,
                    "rank": rank,
                    "R": R,
                    "weights": weights,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                },
                transport_path,
            )

            diagnostics = []
            for edge_idx in range(R.shape[0]):
                diagnostics.append(
                    {
                        "model": args.model,
                        "variant": variant,
                        "rank": rank,
                        "edge": edge_idx,
                        "train_r2": float(train_r2[edge_idx]),
                        "val_r2": float(val_r2[edge_idx]),
                        "weight": float(weights[edge_idx]),
                    }
                )
            write_csv(diagnostics_path, diagnostics)
            write_json(
                config_path,
                {
                    "input": input_path,
                    "model": args.model,
                    "variant": variant,
                    "rank": rank,
                    "lambda_ridge": args.lambda_ridge,
                    "train_frac": args.train_frac,
                    "val_frac": args.val_frac,
                    "seed": args.seed,
                    "n_examples": len(examples),
                    "n_train": len(train_examples),
                    "n_val": len(val_examples),
                    "projection_path": str(projection_path),
                    "transport_path": str(transport_path),
                    "diagnostics_path": str(diagnostics_path),
                    "variant_note": (
                        "diagnostic: representation chosen from clean-conflict deltas"
                        if variant == "delta_pca_diagnostic"
                        else "main non-leaky/candidate representation"
                    ),
                },
            )
            print(
                f"saved {variant} r{rank}: mean val R2={float(val_r2.mean()):.4f}, "
                f"active_edges={(weights > 0).sum().item()}"
            )


if __name__ == "__main__":
    main()
