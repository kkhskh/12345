"""Run the whole pipeline in order, logging each step to logs/ with scripts/run_logged.py.

Usage (from the package root):
    python scripts/run_all.py                # regenerate analysis, tables, figures; verify; test; manifest
    python scripts/run_all.py --rebuild-pdf  # also rebuild the TeX and PDF (needs tectonic)
    python scripts/run_all.py --pdf-text     # also check the PDF text (needs pymupdf)
Stops at the first step with a nonzero exit code and returns that code.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-pdf", action="store_true")
    ap.add_argument("--pdf-text", action="store_true")
    ap.add_argument("--prefix", default="run", help="label prefix for the log files")
    a = ap.parse_args()
    steps = [
        ("recompute_analysis", ["python", "scripts/recompute_analysis.py"]),
        ("reproduce_tables", ["python", "scripts/reproduce_tables.py"]),
        ("reproduce_figures", ["python", "scripts/reproduce_figures.py"]),
    ]
    if a.rebuild_pdf:
        steps.append(("build_pdf", ["python", "scripts/build_pdf.py"]))
    steps += [
        ("verify_paper_claims", ["python", "scripts/verify_paper_claims.py"]),
        ("pytest", ["python", "-m", "pytest", "tests/test_paper_reproducibility.py", "-q"]),
        ("independent_harness", ["python", "verification/independent_harness.py"]),
        ("build_release_manifest", ["python", "scripts/build_release_manifest.py"]),
        ("cross_artifact_check", ["python", "scripts/cross_artifact_check.py"] + (["--pdf-text"] if a.pdf_text else [])),
    ]
    for i, (name, cmd) in enumerate(steps, 1):
        rc = subprocess.run([sys.executable, "scripts/run_logged.py", f"{a.prefix}_{i:02d}_{name}", "--"] + cmd, cwd=PKG).returncode
        if rc != 0:
            print(f"step {name} failed with exit code {rc}; see logs/{a.prefix}_{i:02d}_{name}.txt")
            return rc
    print("all steps passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
