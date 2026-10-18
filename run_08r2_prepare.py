"""08R2 Phase 0/1 prepare: dependency audit and validation pairs.

Does not load Othello-GPT. Does not evaluate the sealed confirmation split.
Does not write fabricated real-model metrics.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

from src.othello_r1_eval import resolve_probe_path, sha256_file
from src.othello_r2 import (
    CONFIRMATION_SPLIT,
    OFFICIAL_0A_LETTER,
    confirmation_guard_text,
    isolate_r2_results_dir,
    load_holdout_game_ids,
    pair_census_public,
    r2_pair_categories,
    require_no_confirmation,
)
from src.othello_runutil import add_labels, load_split, write_json
from src.toy_othello import load_config


SOURCE_FILES = (
    "src/othello_r2.py",
    "src/othello_r1.py",
    "src/othello_r1_eval.py",
    "src/othello_runutil.py",
    "src/toy_othello.py",
    "run_08r2_prepare.py",
    "run_08r2_train_and_validate.py",
    "run_08r2_confirmation.py",
    "run_08r2_check_readiness.py",
    "src/othello_r2_eval.py",
    "configs/08r2_preregistration.md",
    "configs/08r2_protocol_option1_full_abcd.md",
    "configs/08r2_protocol_option2_narrow_ad.md",
    "run_08r1_real_model_verification.py",
    "run_08c_othello_state_probes.py",
    "run_08f_othello_state_causal_edits.py",
    "configs/toy_othello_08r2.json",
    "configs/08r1_fresh_holdout_manifest.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2 prepare / dependency audit.")
    p.add_argument("--config", default="configs/toy_othello_08r2.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def merge_cfg(parent: dict, child: dict) -> dict:
    out = dict(parent)
    out.update(child)
    return out


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    parent = load_config(args.parent_config)
    child = load_config(args.config)
    cfg = merge_cfg(parent, child)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    isolated = isolate_r2_results_dir(cfg["results_dir"])
    out_dir = Path(isolated["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = Path(cfg.get("holdout_manifest", "configs/08r1_fresh_holdout_manifest.json"))
    holdout = load_holdout_game_ids(holdout_path)
    file_hash = holdout["file_sha256"]
    if holdout["split"] != CONFIRMATION_SPLIT:
        raise SystemExit(f"holdout split mismatch: {holdout['split']}")
    if not holdout["hash_ok"]:
        raise SystemExit(
            "holdout payload hash mismatch: "
            f"payload={holdout['payload_sha256']} recorded={holdout['manifest_sha256']}"
        )

    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    ckpt_cands = []
    if args.checkpoint:
        ckpt_cands.append(Path(args.checkpoint))
    ckpt_cands.append(Path(cfg["checkpoints_dir"]) / cfg.get("checkpoint_file", "synthetic_model.pth"))
    ckpt = next((p for p in ckpt_cands if p.is_file()), ckpt_cands[0])

    data_dir = Path(cfg["data_dir"])
    data_files = {
        "games": data_dir / "games.json",
        "prefixes_train": data_dir / "prefixes_train.jsonl",
        "prefixes_val": data_dir / "prefixes_val.jsonl",
        "prefixes_test": data_dir / "prefixes_test.jsonl",
    }

    source_hashes = {}
    missing_sources = []
    for rel in SOURCE_FILES:
        path = root / rel
        digest = sha256_file(path)
        source_hashes[rel] = {"path": str(path), "exists": path.is_file(), "sha256": digest}
        if not path.is_file():
            missing_sources.append(rel)

    pairs = []
    pair_counts = {}
    val_rows = []
    if data_files["prefixes_val"].is_file():
        val_rows = add_labels(load_split(data_dir, "val"))
        require_no_confirmation(val_rows, holdout["game_ids"])
        pairs = r2_pair_categories(val_rows, int(cfg["n_val_pairs"]), int(cfg["pair_seed"]), holdout["game_ids"])
        pair_counts = dict(Counter(p["r2_type"] for p in pairs))
        census = pair_census_public(val_rows, holdout["game_ids"])
        write_json(
            out_dir / "08r2_validation_pairs.json",
            {
                "n": len(pairs),
                "counts": pair_counts,
                "census": census,
                "seed": int(cfg["pair_seed"]),
                "split": "val",
                "used_confirmation": 0,
                "protocol_choice": "UNFROZEN_AWAITING_REVIEW",
                "pairs": pairs,
            },
        )
    else:
        write_json(
            out_dir / "08r2_validation_pairs.json",
            {
                "n": 0,
                "counts": {},
                "seed": int(cfg["pair_seed"]),
                "split": "val",
                "used_confirmation": 0,
                "pairs": [],
                "status": "BLOCKED_MISSING_ARTIFACT",
                "artifact": str(data_files["prefixes_val"]),
            },
        )

    audit = {
        "experiment": "08R2",
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
        "git_hash": git_hash(),
        "site": cfg["site"],
        "results_dir": str(out_dir),
        "isolation": isolated,
        "confirmation_sealed": True,
        "confirmation_guard": confirmation_guard_text(),
        "holdout": {
            "path": str(holdout_path),
            "split": holdout["split"],
            "n_games": holdout["n_games"],
            "file_sha256": file_hash,
            "payload_sha256": holdout["payload_sha256"],
            "recorded_sha256": holdout["manifest_sha256"],
            "hash_ok": holdout["hash_ok"],
            "used_in_validation": 0,
        },
        "checkpoint": {
            "path": str(ckpt),
            "exists": ckpt.is_file(),
            "sha256": sha256_file(ckpt) if ckpt.is_file() else None,
            "status": "FOUND" if ckpt.is_file() else "BLOCKED_MISSING_ARTIFACT",
        },
        "probe": {
            **{k: probe_info[k] for k in probe_info},
            "sha256": sha256_file(Path(probe_info["probe_path"])) if probe_info["exists"] else None,
            "status": "FOUND" if probe_info["exists"] else "BLOCKED_MISSING_ARTIFACT",
        },
        "data": {
            name: {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}
            for name, path in data_files.items()
        },
        "source_hashes": source_hashes,
        "missing_sources": missing_sources,
        "validation_pairs": {
            "n": len(pairs),
            "counts": pair_counts,
            "n_val_rows": len(val_rows),
        },
        "real_model_metrics": "PENDING",
        "note": "Prepare does not run the frozen model and does not fabricate metric tables.",
    }
    write_json(out_dir / "08r2_dependency_audit.json", audit)

    protocol = {
        "status": "SEALED",
        "execute": False,
        "reason": confirmation_guard_text(),
        "holdout_path": str(holdout_path),
        "holdout_sha256": file_hash,
        "frozen_editor": None,
        "requires": [
            "08R2_VALIDATION_COMPLETE",
            "frozen editor configuration",
            "explicit --execute-sealed-confirmation",
        ],
    }
    write_json(out_dir / "08r2_confirmation_protocol.json", protocol)
    prereg = root / "configs/08r2_preregistration.md"
    if prereg.is_file():
        (out_dir / "08r2_preregistration.md").write_text(prereg.read_text(encoding="utf-8"), encoding="utf-8")

    missing = [k for k, v in audit["data"].items() if not v["exists"]]
    if not audit["checkpoint"]["exists"]:
        missing.append("checkpoint")
    if not audit["probe"]["exists"]:
        missing.append("probe")
    print("08R2 prepare complete")
    print(f"  results_dir={out_dir}")
    print(f"  pairs={len(pairs)} counts={pair_counts}")
    print(f"  missing_real_artifacts={missing}")
    print(f"  official letter unchanged: {OFFICIAL_0A_LETTER}")
    print("  confirmation sealed: not evaluated")
    if missing_sources:
        raise SystemExit(f"required source files missing: {missing_sources}")


if __name__ == "__main__":
    main()
