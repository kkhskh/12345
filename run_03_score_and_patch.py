from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.baselines import (
    activation_patching_oracle_ranking,
    rank_activation_delta_norm,
    rank_attribution_patching,
    rank_gradient_saliency,
    rank_obstruction_response,
    rank_random,
)
from src.data_ioi import read_jsonl
from src.metrics import effective_site_auc, spearman_pred_observed
from src.model import load_gpt2_small
from src.patching import run_single_layer_sweep, run_topk_sweep
from src.plotting import (
    plot_obstruction_reduction_vs_repair,
    plot_pred_vs_observed_delta_O,
    plot_pred_vs_observed_delta_m,
    plot_repair_at_k,
)
from src.response import score_response_layers


TEST_PATH = Path("data/processed/ioi_test.jsonl")
TRANSPORT_PATH = Path("artifacts/transports/scalar_resid_chain_transports.json")
SCORES_PATH = Path("artifacts/results/response_scores.csv")
PATCHING_PATH = Path("artifacts/results/patching_results.csv")
FIGURE_DIR = Path("artifacts/figures")


def add_method(rows: list[dict], method: str) -> list[dict]:
    out = []
    for row in rows:
        row = dict(row)
        row["method"] = method
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for a quicker patching smoke run.",
    )
    args = parser.parse_args()

    examples = read_jsonl(TEST_PATH)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    with TRANSPORT_PATH.open() as f:
        transports = json.load(f)

    print(f"Loaded test examples: {len(examples)}")
    print("Loading GPT-2 small...")
    model = load_gpt2_small()

    score_rows = []
    patch_rows = []

    for idx, example in enumerate(examples, start=1):
        print(f"[{idx}/{len(examples)}] {example['id']}")

        scores = score_response_layers(
            model=model,
            example=example,
            r=transports["r"],
            w=transports["w"],
        )
        score_rows.extend(scores)

        single_layer_rows = run_single_layer_sweep(
            model,
            example,
            transports["r"],
            transports["w"],
        )
        for row in single_layer_rows:
            row = dict(row)
            row["layer"] = row["layers"][0]
            row["method"] = "single_layer_oracle_observed"
            patch_rows.append(row)

        rankings = {}
        rankings["obstruction_response"], _ = rank_obstruction_response(scores)
        rankings["random"] = rank_random(model, seed=idx)
        rankings["activation_delta_norm"], _ = rank_activation_delta_norm(model, example)
        rankings["gradient_saliency"], _ = rank_gradient_saliency(model, example)
        rankings["attribution_patching"], _ = rank_attribution_patching(model, example)
        rankings["activation_patching_oracle"], _ = activation_patching_oracle_ranking(
            single_layer_rows
        )

        for method, ranking in rankings.items():
            topk_rows = run_topk_sweep(
                model,
                example,
                transports["r"],
                transports["w"],
                ranked_layers=ranking,
                ks=(1, 2, 3, 5),
            )
            patch_rows.extend(add_method(topk_rows, method))

    score_df = pd.DataFrame(score_rows)
    patch_df = pd.DataFrame(patch_rows)

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATCHING_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    score_df.to_csv(SCORES_PATH, index=False)
    patch_df.to_csv(PATCHING_PATH, index=False)

    observed_df = patch_df[patch_df["method"] == "single_layer_oracle_observed"].copy()
    print("Main metrics:")
    print(
        "Spearman(pred_delta_m, observed_delta_m):",
        spearman_pred_observed(score_df, observed_df, "pred_delta_m", "delta_margin"),
    )
    print(
        "Spearman(pred_delta_O, observed_delta_O):",
        spearman_pred_observed(score_df, observed_df, "pred_delta_O", "delta_O"),
    )
    print(
        "Effective-site AUC(score):",
        effective_site_auc(score_df, observed_df, "score"),
    )
    print(
        patch_df[patch_df["method"] != "single_layer_oracle_observed"]
        .groupby(["method", "k"])
        .agg(
            repair_at_k=("repair_success_flip", "mean"),
            mean_delta_margin=("delta_margin", "mean"),
            mean_off_target_kl=("off_target_kl_corrupt_to_patched", "mean"),
        )
    )

    plot_pred_vs_observed_delta_m(
        score_df,
        observed_df,
        save_path=FIGURE_DIR / "predicted_vs_observed_delta_margin.png",
    )
    plot_pred_vs_observed_delta_O(
        score_df,
        observed_df,
        save_path=FIGURE_DIR / "predicted_vs_observed_delta_obstruction.png",
    )
    plot_repair_at_k(
        patch_df[patch_df["method"] != "single_layer_oracle_observed"],
        save_path=FIGURE_DIR / "topk_repair_vs_baselines.png",
    )
    plot_obstruction_reduction_vs_repair(
        patch_df[patch_df["method"] != "single_layer_oracle_observed"],
        save_path=FIGURE_DIR / "obstruction_reduction_vs_repair.png",
    )

    print(f"Saved scores to {SCORES_PATH}")
    print(f"Saved patching results to {PATCHING_PATH}")


if __name__ == "__main__":
    main()
