"""Print FOUND/MISSING for every file 08R1 needs. Does not run the model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.othello_r1_eval import isolate_r1_results_dir, resolve_probe_path


def _status(path: Path | None) -> str:
    if path is None:
        return "MISSING"
    p = Path(path)
    if p.is_file() or p.is_dir():
        return "FOUND"
    return "MISSING"


def _sha(path: Path | None) -> str:
    if path is None or not Path(path).is_file():
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R1 preflight FOUND/MISSING check.")
    p.add_argument("--config", default="configs/toy_othello.json")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--runner", default="run_08r1_real_model_verification.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    ckpt = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "synthetic_model.pth"
    probe = resolve_probe_path(args.probe_path, ckpt_dir)
    isolation = isolate_r1_results_dir(args.results_dir)
    required = [
        ("runner", Path(args.runner)),
        ("othello_r1.py", Path("src/othello_r1.py")),
        ("othello_r1_eval.py", Path("src/othello_r1_eval.py")),
        ("config", Path(args.config)),
        ("prefixes_train.jsonl", data / "prefixes_train.jsonl"),
        ("prefixes_val.jsonl", data / "prefixes_val.jsonl"),
        ("prefixes_test.jsonl", data / "prefixes_test.jsonl"),
        ("games.json", data / "games.json"),
        ("checkpoint", ckpt),
        ("probe", Path(probe["probe_path"])),
    ]
    outputs = [
        ("results_parent", Path(isolation["requested"])),
        ("results_dir_08r1", Path(isolation["results_dir"])),
    ]
    missing = 0
    print("=== 08R1 preflight ===")
    print(f"probe_source={probe['source']} fallback_used={probe['fallback_used']}")
    print(f"results_requested={isolation['requested']}")
    print(f"results_actual={isolation['results_dir']} nested={isolation['nested']}")
    print(f"historical_in_requested={len(isolation['historical_artifacts_seen'])}")
    for name, path in required + outputs:
        st = _status(path)
        required_miss = name in {n for n, _ in required} and st == "MISSING"
        if required_miss:
            missing += 1
        extra = _sha(path)[:12] if st == "FOUND" and path.is_file() else ""
        kind = "required" if name in {n for n, _ in required} else "output"
        print(f"{st:8s} {kind:8s} {name:24s} {path} {extra}")
    print(f"missing_count={missing}")
    print("Official 0A letter unchanged: C / PARTIAL")
    print("This preflight does not claim 08R1_COMPLETE.")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
