from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


MATCHED_PATH = Path("data/factual_conflict/factual_conflict_margin_matched.jsonl")
MATCHED_CSV_PATH = Path("artifacts/results/factual_conflict_margin_matched.csv")
OBSTRUCTION_CSV_PATH = Path("artifacts/results/factual_margin_matched_obstruction.csv")
SUMMARY_PATH = Path("artifacts/results/factual_model_sweep_summary.csv")


def slug(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def partial_corr_obstruction_condition_controlling_margin(df: pd.DataFrame) -> float:
    x_margin = df[["margin"]].values
    o = df["O_norm"].values
    y = (df["condition"] == "conflict_correction").astype(float).values

    o_resid = o - LinearRegression().fit(x_margin, o).predict(x_margin)
    y_resid = y - LinearRegression().fit(x_margin, y).predict(x_margin)

    if np.std(o_resid) == 0 or np.std(y_resid) == 0:
        return float("nan")
    return float(np.corrcoef(o_resid, y_resid)[0, 1])


def summarize_model(model_name: str, tau: float) -> dict:
    matched = pd.read_csv(MATCHED_CSV_PATH)
    obstruction = pd.read_csv(OBSTRUCTION_CSV_PATH)

    y = (obstruction["condition"] == "conflict_correction").astype(int).values
    auc = roc_auc_score(y, obstruction["O_norm"].values)
    partial_corr = partial_corr_obstruction_condition_controlling_margin(obstruction)

    return {
        "model_name": model_name,
        "tau": tau,
        "matched_rows": len(matched),
        "matched_unique_facts": matched[
            ["subject", "relation", "true_object", "false_object"]
        ].drop_duplicates().shape[0],
        "mean_clean_margin": obstruction[obstruction["condition"] == "clean"][
            "margin"
        ].mean(),
        "mean_conflict_margin": obstruction[
            obstruction["condition"] == "conflict_correction"
        ]["margin"].mean(),
        "mean_clean_obstruction": obstruction[obstruction["condition"] == "clean"][
            "O_norm"
        ].mean(),
        "mean_conflict_obstruction": obstruction[
            obstruction["condition"] == "conflict_correction"
        ]["O_norm"].mean(),
        "auc_conflict_gt_clean": auc,
        "partial_corr_obstruction_conflict_given_margin": partial_corr,
    }


def copy_artifacts(model_name: str) -> None:
    model_slug = slug(model_name)
    copies = {
        Path("data/factual_conflict/factual_conflict_raw.jsonl"): Path(
            f"data/factual_conflict/factual_conflict_raw__{model_slug}.jsonl"
        ),
        Path("data/factual_conflict/factual_conflict_scored.jsonl"): Path(
            f"data/factual_conflict/factual_conflict_scored__{model_slug}.jsonl"
        ),
        MATCHED_PATH: Path(
            f"data/factual_conflict/factual_conflict_margin_matched__{model_slug}.jsonl"
        ),
        Path("artifacts/results/factual_conflict_scored_margins.csv"): Path(
            f"artifacts/results/factual_conflict_scored_margins__{model_slug}.csv"
        ),
        MATCHED_CSV_PATH: Path(
            f"artifacts/results/factual_conflict_margin_matched__{model_slug}.csv"
        ),
        OBSTRUCTION_CSV_PATH: Path(
            f"artifacts/results/factual_margin_matched_obstruction__{model_slug}.csv"
        ),
        Path("artifacts/figures/factual_margin_matched_obstruction.png"): Path(
            f"artifacts/figures/factual_margin_matched_obstruction__{model_slug}.png"
        ),
    }
    for src, dst in copies.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2-small", "gpt2-medium"],
        help=(
            "Model names accepted by TransformerLens. Try: gpt2-small gpt2-medium "
            "EleutherAI/pythia-410m"
        ),
    )
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    rows = []
    for model_name in args.models:
        print(f"\n=== factual sweep model: {model_name} ===", flush=True)
        try:
            run(
                [
                    args.python,
                    "-u",
                    "run_03a_generate_factual_conflict_data.py",
                    "--model-name",
                    model_name,
                ]
            )
            run(
                [
                    args.python,
                    "-u",
                    "run_03b_score_factual_margins.py",
                    "--model-name",
                    model_name,
                ]
            )
            run(
                [
                    args.python,
                    "-u",
                    "run_03c_select_margin_matched_factual_pairs.py",
                    "--tau",
                    str(args.tau),
                    "--require-positive-clean",
                    "--require-positive-correction",
                    "--one-per-fact",
                ]
            )
            run(
                [
                    args.python,
                    "-u",
                    "run_03d_factual_margin_matched_obstruction.py",
                    "--model-name",
                    model_name,
                ]
            )
            row = summarize_model(model_name, args.tau)
            row["status"] = "ok"
            copy_artifacts(model_name)
        except subprocess.CalledProcessError as exc:
            row = {
                "model_name": model_name,
                "tau": args.tau,
                "status": f"failed_exit_{exc.returncode}",
            }
        rows.append(row)

        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)

    print(f"\nSaved sweep summary to {SUMMARY_PATH}")
    print("Gate: proceed only if matched_unique_facts >= 100, AUC > 0.65, pcorr > 0.20.")


if __name__ == "__main__":
    main()
