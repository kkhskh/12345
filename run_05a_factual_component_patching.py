from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.factual_component_patching import build_component_sites, evaluate_component_patch
from src.low_rank_germs import autodetect_factual_input, load_factual_examples, safe_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2C factual component patching oracle.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifact-dir", default="artifacts/factual_low_rank")
    parser.add_argument("--variant", default="clean_pca")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--site-types", nargs="+", default=["head", "mlp"], choices=["head", "mlp", "resid_pre"])
    parser.add_argument("--include-resid-controls", action="store_true")
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--max-sites", type=int, default=64)
    parser.add_argument(
        "--compute-gradient-baselines",
        action="store_true",
        help="Compute multi-token gradient attribution/saliency baselines; expensive.",
    )
    parser.add_argument("--output", default="artifacts/factual_component_patching/component_patch_oracle.csv")
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


def load_low_rank_artifacts(artifact_dir: Path, model_name: str, variant: str, rank: int):
    import torch

    slug = safe_model_name(model_name)
    projection_path = artifact_dir / "projections" / f"{slug}_{variant}_r{rank}.pt"
    transport_path = artifact_dir / "transports" / f"{slug}_{variant}_r{rank}.pt"
    if not projection_path.exists() or not transport_path.exists():
        raise FileNotFoundError(
            "Missing low-rank artifacts. Run run_04a first.\n"
            f"projection: {projection_path}\ntransport: {transport_path}"
        )
    return (
        torch.load(projection_path, map_location="cpu"),
        torch.load(transport_path, map_location="cpu"),
    )


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
    projection, transport = load_low_rank_artifacts(
        Path(args.artifact_dir),
        args.model,
        args.variant,
        args.rank,
    )
    sites = build_component_sites(
        model,
        site_types=args.site_types,
        include_resid_controls=args.include_resid_controls,
    )
    if args.max_sites is not None:
        sites = sites[: args.max_sites]

    print(
        f"patching {len(sites)} sites x {len(examples)} examples "
        f"using {args.variant} r{args.rank}",
        flush=True,
    )
    rows = []
    for site in tqdm(sites, desc="component sites"):
        for ex in examples:
            row = evaluate_component_patch(
                model,
                ex,
                site,
                projection,
                transport,
                compute_gradient_baselines=args.compute_gradient_baselines,
            )
            row["model"] = args.model
            row["variant"] = args.variant
            row["rank"] = args.rank
            rows.append(row)
    write_csv(args.output, rows)
    print(f"saved component patching oracle to {args.output}")
    print("No patching conclusion is made here; run run_05b to summarize controls.")


if __name__ == "__main__":
    main()
