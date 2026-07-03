from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.factual_obstruction_eval import evaluate_obstruction_metrics
from src.low_rank_germs import (
    autodetect_factual_input,
    load_factual_examples,
    multi_token_margin,
    project_activations,
    run_resid_final_token_cache,
    safe_model_name,
)
from src.vector_transports import (
    compute_vector_obstruction,
    identity_transports,
    random_transports,
    shuffled_transports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-rank factual obstruction.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--artifact-dir", default="artifacts/factual_low_rank")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["clean_pca", "answer_augmented_clean_pca", "delta_pca_diagnostic"],
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["learned", "identity", "random", "shuffled"],
        choices=["learned", "identity", "random", "random_orthogonal", "shuffled", "learned_uniform"],
    )
    parser.add_argument("--output-dir", default="artifacts/factual_low_rank/results")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_split_assignments(artifact_dir: Path, model_slug: str) -> dict[str, str]:
    path = artifact_dir / "configs" / f"{model_slug}_split_assignments.csv"
    return {row["id"]: row["split"] for row in read_csv(path)}


def control_transport(control: str, R, weights, seed: int):
    import torch

    L, r = R.shape[0], R.shape[1]
    if control == "learned":
        return R, weights
    if control == "identity":
        return identity_transports(L, r), weights
    if control == "random":
        return random_transports(L, r, seed=seed, orthogonal=False), weights
    if control == "random_orthogonal":
        return random_transports(L, r, seed=seed, orthogonal=True), weights
    if control == "shuffled":
        return shuffled_transports(R, seed=seed), weights
    if control == "learned_uniform":
        return R, torch.ones_like(weights)
    raise ValueError(f"Unknown control: {control}")


def add_rows_for_condition(
    rows: list[dict],
    examples: list[dict],
    condition: str,
    model_name: str,
    variant: str,
    rank: int,
    control: str,
    split_by_id: dict[str, str],
    margins: list[float],
    obstruction,
):
    true_lens = [len(ex.get("answer_a_ids", [])) or "" for ex in examples]
    false_lens = [len(ex.get("answer_b_ids", [])) or "" for ex in examples]
    for idx, ex in enumerate(examples):
        rows.append(
            {
                "id": ex["id"],
                "split": split_by_id.get(ex["id"], ex.get("split", "unknown")),
                "model": model_name,
                "variant": variant,
                "rank": rank,
                "transport_control": control,
                "condition": condition,
                "label_conflict": 1 if condition == "conflict" else 0,
                "margin": margins[idx],
                "abs_margin": abs(margins[idx]),
                "obstruction_raw": float(obstruction["raw"][idx]),
                "obstruction_normalized": float(obstruction["normalized"][idx]),
                "fragility": float(obstruction["normalized"][idx] / (abs(margins[idx]) + 1e-8)),
                "true_answer": ex["true_answer"],
                "false_answer": ex["false_answer"],
                "true_answer_token_len": true_lens[idx],
                "false_answer_token_len": false_lens[idx],
                "subject": ex.get("subject", ""),
                "relation": ex.get("relation", ""),
                "edge_terms": " ".join(f"{float(v):.8g}" for v in obstruction["edge_terms"][idx]),
            }
        )


def make_pairs(long_rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in long_rows:
        key = (
            row["id"],
            row["model"],
            row["variant"],
            row["rank"],
            row["transport_control"],
            row["split"],
        )
        by_key.setdefault(key, {})[row["condition"]] = row
    pairs = []
    for key, conds in by_key.items():
        if "clean" not in conds or "conflict" not in conds:
            continue
        clean = conds["clean"]
        conflict = conds["conflict"]
        O_clean = float(clean["obstruction_normalized"])
        O_conflict = float(conflict["obstruction_normalized"])
        margin_clean = float(clean["margin"])
        margin_conflict = float(conflict["margin"])
        pairs.append(
            {
                "id": key[0],
                "split": key[5],
                "model": key[1],
                "variant": key[2],
                "rank": key[3],
                "transport_control": key[4],
                "O_clean": O_clean,
                "O_conflict": O_conflict,
                "delta_O": O_conflict - O_clean,
                "margin_clean": margin_clean,
                "margin_conflict": margin_conflict,
                "abs_margin_gap": abs(margin_clean - margin_conflict),
                "fragility_clean": clean["fragility"],
                "fragility_conflict": conflict["fragility"],
                "relation": clean.get("relation", ""),
                "subject": clean.get("subject", ""),
            }
        )
    return pairs


def main() -> None:
    args = parse_args()
    input_path = args.input or autodetect_factual_input()
    examples = load_factual_examples(input_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    print(f"input: {input_path}")
    print(f"examples: {len(examples)}")
    print(f"Loading {args.model}...")
    from src.model import answer_token_ids, load_pretrained_model
    import torch

    model = load_pretrained_model(args.model, device=args.device)
    for ex in examples:
        ex["answer_a_ids"] = answer_token_ids(model, ex["true_answer"])
        ex["answer_b_ids"] = answer_token_ids(model, ex["false_answer"])

    model_slug = safe_model_name(args.model)
    artifact_dir = Path(args.artifact_dir)
    split_by_id = load_split_assignments(artifact_dir, model_slug)

    H_clean = run_resid_final_token_cache(
        model, [ex["clean_prompt"] for ex in examples], args.batch_size, args.device
    )
    H_conflict = run_resid_final_token_cache(
        model, [ex["conflict_prompt"] for ex in examples], args.batch_size, args.device
    )
    margin_clean = [
        multi_token_margin(model, ex["clean_prompt"], ex["true_answer"], ex["false_answer"])
        for ex in examples
    ]
    margin_conflict = [
        multi_token_margin(model, ex["conflict_prompt"], ex["true_answer"], ex["false_answer"])
        for ex in examples
    ]

    long_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    for variant in args.variants:
        for rank in args.ranks:
            projection_path = artifact_dir / "projections" / f"{model_slug}_{variant}_r{rank}.pt"
            transport_path = artifact_dir / "transports" / f"{model_slug}_{variant}_r{rank}.pt"
            diagnostics_path = artifact_dir / "diagnostics" / f"{model_slug}_{variant}_r{rank}_transport_diagnostics.csv"
            if not projection_path.exists() or not transport_path.exists():
                print(f"skip missing artifact: {variant} r{rank}")
                continue
            projection = torch.load(projection_path, map_location="cpu")
            transport = torch.load(transport_path, map_location="cpu")
            diagnostics_rows.extend(read_csv(diagnostics_path))
            S_clean = project_activations(H_clean, projection["mu_by_layer"], projection["P_by_layer"])
            S_conflict = project_activations(H_conflict, projection["mu_by_layer"], projection["P_by_layer"])
            for control in args.controls:
                R_control, w_control = control_transport(
                    control, transport["R"], transport["weights"], seed=args.seed
                )
                obs_clean = compute_vector_obstruction(S_clean, R_control, w_control)
                obs_conflict = compute_vector_obstruction(S_conflict, R_control, w_control)
                add_rows_for_condition(
                    long_rows,
                    examples,
                    "clean",
                    args.model,
                    variant,
                    rank,
                    control,
                    split_by_id,
                    margin_clean,
                    obs_clean,
                )
                add_rows_for_condition(
                    long_rows,
                    examples,
                    "conflict",
                    args.model,
                    variant,
                    rank,
                    control,
                    split_by_id,
                    margin_conflict,
                    obs_conflict,
                )

    pair_rows = make_pairs(long_rows)
    metrics_rows = evaluate_obstruction_metrics(long_rows, pair_rows, diagnostics_rows)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "low_rank_obstruction_long.csv", long_rows)
    write_csv(output_dir / "low_rank_obstruction_pairs.csv", pair_rows)
    write_csv(output_dir / "low_rank_obstruction_metrics.csv", metrics_rows)

    print("Ranked summary by split/test partial corr and AUC:")
    ranked = sorted(
        metrics_rows,
        key=lambda row: (
            row.get("split") == "test",
            float(row.get("partial_corr_margin") or 0),
            float(row.get("auc_conflict_gt_clean") or 0),
        ),
        reverse=True,
    )
    for row in ranked[:20]:
        print(
            f"{row['split']:>5} {row['variant']} r{row['rank']} {row['transport_control']}: "
            f"n={row['n_pairs']} auc={float(row['auc_conflict_gt_clean']):.4f} "
            f"pcorr={float(row['partial_corr_margin']):.4f} "
            f"valR2={float(row['learned_transport_mean_val_r2']):.4f}"
        )
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
