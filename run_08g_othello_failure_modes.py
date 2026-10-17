"""08g: state-estimation vs decision-policy errors. No manufactured errors."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_runutil import add_labels, induced_legal, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, load_othello_gpt, relative_to_absolute, run_model_logits, softmax_np, square_to_token, token_to_square


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08g state vs decision failures.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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
        write_json(results_dir / "08g_status.json", {"status": "BLOCKED", "reason": "08c missing"})
        print("08g blocked.")
        return
    with probe_path.open("rb") as handle:
        pack = pickle.load(handle)
    test = add_labels(load_split(Path(cfg["data_dir"]), "test"))
    try:
        model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=args.checkpoint)
    except Exception as exc:
        write_json(results_dir / "08g_status.json", {"status": "BLOCKED", "reason": str(exc)})
        print(f"BLOCKED: {exc}")
        return
    X = cache_site(model, tokens_of(test), pack["hook"], cfg["cache_batch"]).numpy()
    logits = run_model_logits(model, tokens_of(test), cfg["batch_size"])
    probs = softmax_np(logits)
    rows = []
    for i, r in enumerate(test):
        pred = np.stack([int(m.predict(X[i : i + 1])[0]) for m in pack["models"]])
        if pack["target"] == "rel":
            pred_abs = relative_to_absolute(pred, int(r["player"]))
        else:
            pred_abs = pred
        dec_legal = set(induced_legal(pred_abs, int(r["player"])))
        true_legal = set(r["legal"])
        pref = int(probs[i].argmax())
        pref_sq = token_to_square(pref) if pref in {k for k in range(61)} else -1
        pref_legal_true = pref_sq in true_legal or (pref == 0 and r["must_pass"])
        pref_legal_decoded = pref_sq in dec_legal
        decode_ok = pred_abs.tolist() == r["abs_lab"].tolist()
        if pref_legal_true:
            label = "correct_legal_argmax"
        elif (not decode_ok) and pref_legal_decoded:
            label = "consistent_with_state_error"
        elif decode_ok and not pref_legal_true:
            label = "consistent_with_downstream_decision_error"
        else:
            label = "ambiguous"
        rows.append(
            {
                "game_id": r["game_id"],
                "t": r["t"],
                "decode_square_acc": float((pred_abs == r["abs_lab"]).mean()),
                "decode_exact": int(decode_ok),
                "pref_token": pref,
                "pref_legal_true": int(pref_legal_true),
                "pref_legal_decoded": int(pref_legal_decoded),
                "label": label,
                "legal_mass": float((probs[i] * r["legal_mask"]).sum()),
            }
        )
    write_csv(results_dir / "08g_state_vs_decision_failures.csv", rows)
    counts = {k: int(sum(r["label"] == k for r in rows)) for k in {r["label"] for r in rows}}
    write_json(
        results_dir / "08g_status.json",
        {
            "status": "OK",
            "counts": counts,
            "n_illegal_argmax": int(sum(1 - r["pref_legal_true"] for r in rows)),
            "note": "Labels are consistency tags, not causally proven failure types unless 08f repair distinguishes them.",
        },
    )
    print("08g counts", counts)


if __name__ == "__main__":
    main()
