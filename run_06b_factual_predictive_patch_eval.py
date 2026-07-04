from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.low_rank_germs import autodetect_factual_input, load_factual_examples
from src.predictive_response import (
    ComponentSite,
    cache_component_activations,
    load_low_rank_artifacts,
    observed_patch_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed predictive response rankings by patching.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--predictions", default="artifacts/factual_predictive_response/predicted_response_scores.csv")
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifact-dir", default="artifacts/factual_low_rank")
    parser.add_argument("--variant", default="clean_pca")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--output-patch", default="artifacts/factual_predictive_response/predictive_patch_outcomes.csv")
    parser.add_argument("--output-joined", default="artifacts/factual_predictive_response/predictive_response_joined.csv")
    parser.add_argument("--compute-kl", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def site_from_row(row: dict) -> ComponentSite:
    head = row.get("head", "")
    return ComponentSite(
        site_id=row["site_id"],
        site_type=row["site_type"],
        layer=int(row["layer"]),
        head=None if head in {"", "None", "none"} else int(float(head)),
        hook_name=row["hook_name"],
    )


def main() -> None:
    args = parse_args()
    input_path = args.input or autodetect_factual_input()
    examples = load_factual_examples(input_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    by_id = {ex["id"]: ex for ex in examples}
    predictions = read_csv(args.predictions)
    predictions = [
        row for row in predictions
        if row.get("id") in by_id
        and row.get("variant") == args.variant
        and int(float(row.get("rank", args.rank))) == args.rank
    ]
    if not predictions:
        raise RuntimeError("No prediction rows matched input examples/variant/rank.")

    unique_pairs = {}
    for row in predictions:
        key = (row["id"], row["site_id"])
        if args.max_sites is not None:
            seen_sites = {k[1] for k in unique_pairs}
            if row["site_id"] not in seen_sites and len(seen_sites) >= args.max_sites:
                continue
        unique_pairs.setdefault(key, row)

    print(f"input: {input_path}")
    print(f"prediction rows: {len(predictions)}")
    print(f"unique example-site pairs: {len(unique_pairs)}")
    print(f"Loading {args.model}...")
    from src.model import load_pretrained_model
    from tqdm.auto import tqdm

    model = load_pretrained_model(args.model, device=args.device)
    artifacts = load_low_rank_artifacts(args.artifact_dir, args.model, args.variant, args.rank)

    outcome_rows = []
    for (ex_id, _), pred_row in tqdm(unique_pairs.items(), desc="patch eval"):
        ex = by_id[ex_id]
        site = site_from_row(pred_row)
        clean_cache = cache_component_activations(model, ex["clean_prompt"], [site])
        outcome = observed_patch_eval(
            model,
            ex,
            site,
            clean_cache[site.site_id],
            artifacts,
            compute_kl=args.compute_kl,
        )
        outcome["model"] = args.model
        outcome["variant"] = args.variant
        outcome["rank"] = args.rank
        outcome_rows.append(outcome)

    outcome_by_key = {(row["id"], row["site_id"]): row for row in outcome_rows}
    joined = []
    for pred in predictions:
        outcome = outcome_by_key.get((pred["id"], pred["site_id"]))
        if outcome:
            joined.append({**pred, **{f"actual_{k}" if k in pred else k: v for k, v in outcome.items()}})

    write_csv(args.output_patch, outcome_rows)
    write_csv(args.output_joined, joined)
    print(f"saved patch outcomes to {args.output_patch}")
    print(f"saved joined predictions/outcomes to {args.output_joined}")


if __name__ == "__main__":
    main()
