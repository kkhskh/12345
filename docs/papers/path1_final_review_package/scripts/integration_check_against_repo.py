"""Integration check: compare package primitives with the research repository's own implementation.

This is the only file in the package that imports the research repository, and it is not part of
the package tests. It needs a checkout of the repository:

    python scripts/integration_check_against_repo.py --repo /path/to/ml_research

It compares, on every saved fixed-JS, SPEC and NST cell: Y, G and Q from the repository's
y_components; JS from the repository's js_divergence; the repository's square_to_token; and
the repository's cluster_bootstrap on the pooled contrasts. Exit code 1 on any disagreement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1lib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    repo = ap.parse_args().repo.resolve()
    p1lib.require((repo / "src" / "othello_r2_fiber.py").is_file(), f"not a research repository checkout: {repo}")
    sys.path.insert(0, str(repo))
    from src.othello_r1 import cluster_bootstrap as repo_boot
    from src.othello_r2_fiber import js_divergence as repo_js
    from src.othello_r2_fiber import softmax as repo_softmax
    from src.othello_r2_fiber_spec import y_components as repo_y
    from src.toy_othello import square_to_token as repo_tok

    bad = []
    n_tok = sum(repo_tok(s) == p1lib.square_to_token(s) for s in range(64) if s not in p1lib.CENTER_SQUARES)
    if n_tok != 60:
        bad.append(f"square_to_token disagrees on {60 - n_tok} squares")

    worst = {"Y": 0.0, "G": 0.0, "Q": 0.0, "JS": 0.0, "softmax": 0.0}
    n_cells = 0
    for study, logits_f, sets_f in (("fixed_js", "fixed_js/fixed_js_full_logits.npz", "spec/fiber_spec_target_legal_sets.json"),
                                    ("spec", "spec/fiber_spec_full_logits.npz", "spec/fiber_spec_target_legal_sets.json"),
                                    ("nst", "nst/null_state_full_logits.npz", "nst/null_state_target_legal_sets.json")):
        L = p1lib.load_npz(p1lib.RAW / logits_f)
        sets = p1lib.read_json(p1lib.RAW / sets_f)
        for key, logits in L.items():
            rg, tg, cell = key.split("__")
            s = sets[f"{rg}|{tg}"]
            if study == "nst":
                rec, tgt = s["recipient"]["legal_squares"], s["target"]["legal_squares"]
                gl, il = s["became_legal_squares"], s["became_illegal_squares"]
            else:
                rec, tgt, gl, il = s["recip_legal"], s["target_legal"], s["became_legal"], s["became_illegal"]
            p_pkg = p1lib.softmax(logits)
            p_repo = repo_softmax(logits)
            worst["softmax"] = max(worst["softmax"], float(np.abs(p_pkg - p_repo).max()))
            mine = p1lib.y_components(p_pkg, gl, il)
            theirs = repo_y(p_repo, rec, tgt)
            for k in ("Y", "G", "Q"):
                worst[k] = max(worst[k], abs(mine[k] - float(theirs[k])))
            ref = L[f"{rg}__{tg}__editor_A"]
            worst["JS"] = max(worst["JS"], abs(p1lib.js_divergence(p1lib.softmax(ref), p_pkg) - float(repo_js(repo_softmax(ref), p_repo))))
            n_cells += 1
    for k, v in worst.items():
        print(f"max |package - repository| {k:<8} {v:.3e}")
        if v > 1e-12:
            bad.append(f"{k} differs by {v:.3e}")

    res = p1lib.compute_all()
    for study, rows_key, metric_cols in (("fixed_js", "pairs", (("delta_specific", "delta_specific", 82), ("delta_random", "delta_random", 83))),
                                         ("nst", "pairs", (("wrong", "delta_wrong", 82), ("editor", "delta_editor", 85)))):
        rows = res[study][rows_key]
        for metric, col, seed in metric_cols:
            vals = np.array([float(r[col]) for r in rows])
            clusters = [r["recip_game"] for r in rows]
            theirs = repo_boot(vals, clusters, 2000, seed)
            mine = res[study]["slices"]["pooled"][metric]
            d = max(abs(theirs[k] - mine[k]) for k in ("point", "ci95_low", "ci95_high"))
            print(f"{study} {metric}: repository cluster_bootstrap vs package, max diff {d:.3e}")
            if d > 1e-12:
                bad.append(f"bootstrap {study} {metric} differs by {d:.3e}")
    print(f"\ncompared {n_cells} saved cells; " + ("all agree" if not bad else f"DISAGREEMENTS: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except p1lib.AnalysisError as exc:
        print(f"INTEGRATION CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
