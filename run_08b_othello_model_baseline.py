"""08b: frozen Othello-GPT legality baseline. Gate 0."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv, write_text
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import (
    MODEL_SPEC,
    load_config,
    load_othello_gpt,
    legal_metrics,
    run_model_logits,
    softmax_np,
    square_to_token,
    transform_board,
    transform_square,
    verify_nanda_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08b Othello-GPT legality baseline.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _rows_metrics(name: str, logits, rows: list[dict]) -> dict:
    masks = np.stack([r["legal_mask"] for r in rows], axis=0)
    met = legal_metrics(logits, masks)
    met["split"] = name
    met["n_games"] = len({r["game_id"] for r in rows})
    return met


def _stratify(logits, rows: list[dict], key) -> list[dict]:
    import torch

    buckets: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[str(key(r))].append(i)
    out = []
    for name, idx in sorted(buckets.items()):
        sub_l = logits[idx]
        sub_r = [rows[i] for i in idx]
        row = _rows_metrics(name, sub_l, sub_r)
        row["stratum"] = name
        out.append(row)
    _ = torch
    return out


def symmetry_eval(model, rows: list[dict], batch_size: int, name: str = "rot90") -> dict:
    from src.toy_othello import legal_mask_from_board, square_to_token as s2t

    tok_lists = []
    masks = []
    for r in rows[: min(120, len(rows))]:
        sq = [transform_square(s, name) for s in r["squares"]]
        try:
            toks = [s2t(s) for s in sq]
        except KeyError:
            continue
        tb = transform_board(r["board_np"], name)
        tok_lists.append(toks)
        masks.append(legal_mask_from_board(tb, int(r["player"])))
    if len(tok_lists) < 16:
        return {"split": f"symmetry_{name}", "n": len(tok_lists), "status": "too_few"}
    logits = run_model_logits(model, tok_lists, batch_size)
    met = legal_metrics(logits, np.stack(masks, axis=0))
    met["split"] = f"symmetry_{name}"
    return met


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    results_dir = ensure_dir(cfg["results_dir"])
    data_dir = Path(cfg["data_dir"])
    if not (data_dir / "prefixes_val.jsonl").exists():
        raise FileNotFoundError("Missing 08a prefixes. Run run_08a_prepare_othello_data.py first.")

    print("=== 08b load frozen Othello-GPT ===")
    print(json.dumps({k: MODEL_SPEC[k] for k in ("hf_repo", "checkpoint_file", "n_layers", "d_model")}, indent=2))
    try:
        model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=args.checkpoint)
    except Exception as exc:
        write_json(
            results_dir / "08b_baseline_summary.json",
            {"status": "BLOCKED", "reason": f"Could not load checkpoint: {exc}", "gate0": "D"},
        )
        print(f"BLOCKED: {exc}")
        return

    fixture = verify_nanda_checkpoint(model)
    print(f"  Nanda argmax fixture match={fixture['nanda_argmax_match']} mismatches={fixture['n_mismatch']}")
    if not fixture["nanda_argmax_match"]:
        write_json(results_dir / "08b_baseline_summary.json", {"status": "FAIL_FIXTURE", "fixture": fixture})
        raise RuntimeError("Loaded weights do not match the Nanda synthetic fixture.")

    val = add_labels(load_split(data_dir, "val"))
    test = add_labels(load_split(data_dir, "test"))
    print(f"  val={len(val)} test={len(test)}")

    vlog = run_model_logits(model, tokens_of(val), cfg["batch_size"])
    tlog = run_model_logits(model, tokens_of(test), cfg["batch_size"])
    val_met = _rows_metrics("val", vlog, val)
    test_met = _rows_metrics("test", tlog, test)
    print(f"  val legal_mass={val_met['mean_legal_mass']:.3f} top1={val_met['top1_legal']:.3f}")
    print(f"  test legal_mass={test_met['mean_legal_mass']:.3f} top1={test_met['top1_legal']:.3f}")

    gate_min = cfg.get("legality_gate_min")
    if gate_min is None:
        gate_min = 0.70
        cfg["legality_gate_min"] = gate_min
        print(f"  froze val legality gate: mean_legal_mass >= {gate_min}")
    gate0 = "PASS" if val_met["mean_legal_mass"] >= float(gate_min) else "FAIL"
    print(f"  Gate 0 {gate0}")

    rows = [
        {**val_met, "stratum": "all"},
        {**test_met, "stratum": "all"},
    ]
    rows.extend(_stratify(vlog, val, lambda r: f"val_stage_{r['stage']}"))
    rows.extend(_stratify(tlog, test, lambda r: f"test_stage_{r['stage']}"))
    rows.extend(_stratify(tlog, test, lambda r: f"test_player_{r['player']}"))
    rows.extend(_stratify(tlog, test, lambda r: f"test_nlegal_{min(len(r['legal']), 8)}"))
    rows.append(symmetry_eval(model, test, cfg["batch_size"], "rot90"))
    write_csv(results_dir / "08b_model_behavior.csv", rows)
    write_json(
        results_dir / "08b_baseline_summary.json",
        {
            "status": "OK",
            "gate0": gate0,
            "val": val_met,
            "test": test_met,
            "legality_gate_min": gate_min,
            "fixture": {k: fixture[k] for k in ("nanda_argmax_match", "n_mismatch")},
            "model_spec": MODEL_SPEC,
            "note": "Single-move accuracy is not the target; legal-mass is. Training policy is random legal, not uniform over games.",
        },
    )
    write_text(
        results_dir / "08b_gate0.md",
        f"Gate 0 {gate0}. val legal_mass={val_met['mean_legal_mass']:.3f} threshold={gate_min}.\n",
    )


if __name__ == "__main__":
    main()
