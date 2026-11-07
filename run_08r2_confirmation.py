"""08R2 confirmation guard.

CONFIRMATION_EVALUATOR_NOT_IMPLEMENTED.

This file refuses to evaluate the sealed split. It is not a complete
scientific confirmation runner. Do not treat a sealed exit as an evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.othello_r1_eval import sha256_file
from src.othello_r2 import confirmation_guard_text, isolate_r2_results_dir, load_holdout_game_ids
from src.othello_runutil import write_json
from src.toy_othello import load_config


CONFIRMATION_STATUS = "CONFIRMATION_EVALUATOR_NOT_IMPLEMENTED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2 confirmation guard (evaluator not implemented).")
    p.add_argument("--config", default="configs/toy_othello_08r2.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--execute-sealed-confirmation", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parent = load_config(args.parent_config)
    child = load_config(args.config)
    cfg = dict(parent)
    cfg.update(child)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    out_dir = Path(isolate_r2_results_dir(cfg["results_dir"])["results_dir"])
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    write_json(
        out_dir / "08r2_confirmation_protocol.json",
        {
            "status": CONFIRMATION_STATUS,
            "execute": False,
            "reason": confirmation_guard_text(),
            "holdout_sha256": holdout["manifest_sha256"],
            "payload_sha256": holdout["payload_sha256"],
            "file_sha256": holdout["file_sha256"],
            "hash_ok": holdout["hash_ok"],
            "file_bytes_sha256": sha256_file(cfg["holdout_manifest"]),
        },
    )
    raise SystemExit(
        f"{CONFIRMATION_STATUS}. {confirmation_guard_text()} "
        "A complete confirmation evaluator has not been implemented."
    )


if __name__ == "__main__":
    main()
