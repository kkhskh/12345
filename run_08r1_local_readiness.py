"""Static 08R1 source-readiness check. Never writes 08r1_real_* measurements."""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

from src.othello_r1_eval import local_source_readiness
from src.othello_runutil import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="08R1 local source readiness (not COMPLETE).")
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--out", default="artifacts/results/toy_othello/08r1_engineering_local_readiness.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runner = root / "run_08r1_real_model_verification.py"
    helper = root / "src/othello_r1_eval.py"
    for path in (runner, helper, root / "src/othello_r1.py"):
        py_compile.compile(str(path), doraise=True)
    decision = local_source_readiness(
        runner.read_text(encoding="utf-8"),
        helper.read_text(encoding="utf-8"),
        tests_passed=bool(args.tests_passed),
    )
    write_json(args.out, decision)
    print(decision["decision"])
    print("Official 0A letter unchanged: C / PARTIAL")
    print("This script does not claim 08R1_COMPLETE and does not write 08r1_real_* files.")


if __name__ == "__main__":
    main()
