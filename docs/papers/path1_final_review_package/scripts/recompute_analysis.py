"""Layer A -> B: recompute every analysis quantity from data/raw and write analysis/.

Usage (from anywhere):
    python scripts/recompute_analysis.py
Fails with a nonzero exit code if any raw file is missing, any hash differs from
data/provenance_manifest.json, or any internal consistency requirement is violated.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1lib  # noqa: E402


def write_rows(path: Path, rows: list[dict]) -> None:
    p1lib.require(len(rows) > 0, f"refusing to write empty table {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def write_csvs(res: dict, out: Path) -> None:
    write_rows(out / "fixed_js_interventions.csv", res["fixed_js"]["interventions"])
    write_rows(out / "fixed_js_pairs.csv", res["fixed_js"]["pairs"])
    write_rows(out / "spec_pairs.csv", res["spec"]["rows"])
    write_rows(out / "nst_pairs.csv", res["nst"]["pairs"])
    write_rows(out / "fiber_pairs.csv", res["fiber"]["rows"])


def main() -> int:
    res = p1lib.compute_all()
    out = p1lib.ANALYSIS
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(p1lib.strip_rows(res), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csvs(res, out)
    fj = res["fixed_js"]["slices"]["pooled"]
    nst = res["nst"]["slices"]["pooled"]
    print(f"raw files checked: {res['raw_check']['files_checked']}")
    print(f"fixed-JS delta_specific={fj['delta_specific']['point']:.8f} delta_random={fj['delta_random']['point']:.8f}")
    print(f"NST learned-wrong={nst['wrong']['point']:.6f} learned-editorA={nst['editor']['point']:.6f}")
    print(f"stage share={res['stage_share']['share']:.6f}")
    print("analysis written to", out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except p1lib.AnalysisError as exc:
        print(f"ANALYSIS FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
