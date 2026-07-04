from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.low_rank_germs import autodetect_factual_input, load_factual_examples
from src.predictive_response import (
    cache_component_activations,
    compute_margin_gradient_wrt_component,
    compute_obstruction_gradient_wrt_component,
    load_low_rank_artifacts,
    make_component_sites,
    make_patch_direction,
    score_site_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute predictive factual obstruction-response scores.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifact-dir", default="artifacts/factual_low_rank")
    parser.add_argument("--variant", default="clean_pca")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--max-sites", type=int, default=144)
    parser.add_argument("--include-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-resid", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--margin-gradient-mode", choices=["first_token", "sequence"], default="first_token")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="artifacts/factual_predictive_response/predicted_response_scores.csv")
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


def write_config(output: str | Path, payload: dict) -> None:
    path = Path(output).with_suffix(".config.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    input_path = args.input or autodetect_factual_input()
    examples = load_factual_examples(input_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    print(f"input: {input_path}")
    print(f"examples: {len(examples)}")
    print(f"Loading {args.model}...")
    from src.model import load_pretrained_model
    from tqdm.auto import tqdm

    model = load_pretrained_model(args.model, device=args.device)
    artifacts = load_low_rank_artifacts(args.artifact_dir, args.model, args.variant, args.rank)
    sites = make_component_sites(
        model,
        include_attn=args.include_attn,
        include_mlp=args.include_mlp,
        include_resid=args.include_resid,
        max_sites=args.max_sites,
    )
    print(f"scoring {len(sites)} sites x {len(examples)} examples before patching")

    rows: list[dict] = []
    for ex in tqdm(examples, desc="examples"):
        clean_cache = cache_component_activations(model, ex["clean_prompt"], sites)
        conflict_cache = cache_component_activations(model, ex["conflict_prompt"], sites)
        for site in tqdm(sites, desc="sites", leave=False):
            delta = make_patch_direction(clean_cache, conflict_cache, site)
            grad_O = compute_obstruction_gradient_wrt_component(
                model,
                ex["conflict_prompt"],
                site,
                artifacts,
                ex["true_answer"],
                ex["false_answer"],
            )
            grad_m = compute_margin_gradient_wrt_component(
                model,
                ex["conflict_prompt"],
                site,
                ex["true_answer"],
                ex["false_answer"],
                margin_gradient_mode=args.margin_gradient_mode,
            )
            for beta in args.betas:
                for gamma in args.gammas:
                    rows.append(
                        score_site_response(
                            ex,
                            site,
                            delta,
                            grad_O,
                            grad_m,
                            beta,
                            gamma,
                            args.model,
                            args.variant,
                            args.rank,
                            args.margin_gradient_mode,
                        )
                    )
    write_csv(args.output, rows)
    write_config(
        args.output,
        {
            "input": input_path,
            "model": args.model,
            "variant": args.variant,
            "rank": args.rank,
            "max_examples": args.max_examples,
            "max_sites": args.max_sites,
            "margin_gradient_mode": args.margin_gradient_mode,
            "betas": args.betas,
            "gammas": args.gammas,
            "note": "Scores are computed before patching; patch outcomes are not read here.",
        },
    )
    print(f"saved predictive response scores to {args.output}")


if __name__ == "__main__":
    main()
