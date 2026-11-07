#!/usr/bin/env python3
"""Level-1 check of the 336-row fixed-JS verification table. No model forward pass."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).with_name("fixed_js_verified_data.csv")


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    assert len(rows) == 336
    pairs = {r["pair"] for r in rows}
    assert len(pairs) == 48
    js = [float(r["js"]) for r in rows]
    archive = [float(r["archive_js"]) for r in rows]
    probe = [float(r["probe_max_abs"]) for r in rows]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    d_wrong = []
    d_rand = []
    for rs in by.values():
        c = [float(x["y"]) for x in rs if x["method"] == "correct"]
        w = [float(x["y"]) for x in rs if x["method"] == "wrong"]
        rnd = [float(x["y"]) for x in rs if x["method"] == "random"]
        assert len(c) == 1 and len(w) == 3 and len(rnd) == 3
        d_wrong.append(c[0] - sum(w) / 3)
        d_rand.append(c[0] - sum(rnd) / 3)
    print("n_rows", len(rows))
    print("n_pairs", len(pairs))
    print("max_probe_abs", max(probe))
    print("max_js_error", max(abs(x - 0.01) for x in js))
    print("max_js_vs_archive", max(abs(a - b) for a, b in zip(js, archive)))
    print("delta_wrong", sum(d_wrong) / len(d_wrong))
    print("delta_random", sum(d_rand) / len(d_rand))
    print("wins_wrong", sum(d > 0 for d in d_wrong), "/", len(d_wrong))
    print("wins_random", sum(d > 0 for d in d_rand), "/", len(d_rand))
    assert max(probe) < 1e-12
    assert max(abs(x - 0.01) for x in js) < 2e-4
    assert abs(sum(d_wrong) / len(d_wrong) - 0.02655064) < 1e-8
    assert abs(sum(d_rand) / len(d_rand) - 0.02296572) < 1e-8
    print("PASS")


if __name__ == "__main__":
    main()
