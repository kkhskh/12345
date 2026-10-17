"""08h: candidate-specific legality germs from the decoded state. Not obstruction."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_runutil import add_labels, induced_legal, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, load_othello_gpt, relative_to_absolute, run_model_logits, square_to_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08h candidate germ validation.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney
    ranks = 0.0
    for p in pos:
        ranks += float((neg < p).sum() + 0.5 * (neg == p).sum())
    return ranks / (len(pos) * len(neg))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    results_dir = ensure_dir(cfg["results_dir"])
    probe_path = Path(cfg["checkpoints_dir"]) / "08c_selected_probe.pkl"
    if not probe_path.exists():
        write_json(results_dir / "08h_status.json", {"status": "BLOCKED", "reason": "08c missing"})
        print("08h blocked.")
        return
    with probe_path.open("rb") as handle:
        pack = pickle.load(handle)
    test = add_labels(load_split(Path(cfg["data_dir"]), "test"))
    try:
        model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=args.checkpoint)
    except Exception as exc:
        write_json(results_dir / "08h_status.json", {"status": "BLOCKED", "reason": str(exc)})
        print(f"BLOCKED: {exc}")
        return
    X = cache_site(model, tokens_of(test), pack["hook"], cfg["cache_batch"]).numpy()
    logits = run_model_logits(model, tokens_of(test), cfg["batch_size"]).numpy()
    germ = []
    y = []
    margins = []
    for i, r in enumerate(test):
        pred = np.stack([int(m.predict(X[i : i + 1])[0]) for m in pack["models"]])
        lab = relative_to_absolute(pred, int(r["player"])) if pack["target"] == "rel" else pred
        dec_legal = set(induced_legal(lab, int(r["player"])))
        true = set(r["legal"])
        for sq in range(64):
            if sq in (27, 28, 35, 36):
                continue
            tok = square_to_token(sq)
            s_m = 1.0 if sq in dec_legal else 0.0
            germ.append(s_m)
            y.append(1.0 if sq in true else 0.0)
            others = [logits[i, k] for k in range(61) if k != tok]
            margins.append(float(logits[i, tok] - max(others)))
    germ = np.asarray(germ)
    y = np.asarray(y)
    margins = np.asarray(margins)
    rows = [
        {"metric": "germ_auroc_vs_true_legality", "value": auroc(germ, y)},
        {"metric": "margin_auroc_vs_true_legality", "value": auroc(margins, y)},
        {"metric": "germ_accuracy", "value": float(((germ >= 0.5) == y).mean())},
        {"metric": "n_candidate_pairs", "value": float(len(y))},
        {"metric": "note", "value": 0.0, "text": "s_m = 1[m in L(decoded board)]. Not an obstruction edge."},
    ]
    write_csv(results_dir / "08h_candidate_germ_validation.csv", rows)
    write_json(results_dir / "08h_status.json", {"status": "OK", "auroc": rows[0]["value"]})
    print(f"08h germ AUROC={rows[0]['value']:.3f} margin AUROC={rows[1]['value']:.3f}")


if __name__ == "__main__":
    main()
