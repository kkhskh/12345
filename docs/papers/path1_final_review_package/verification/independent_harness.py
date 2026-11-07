"""Independent re-derivation of the headline numbers, written without the package analysis library.

It shares no code with scripts/p1lib.py. It reads data/raw directly, builds its own token list,
uses scipy's Jensen-Shannon distance, implements its own clustered bootstrap with integer draws,
and then compares its numbers with analysis/results.json.

    python verification/independent_harness.py
Exit code 0 only if every comparison is within tolerance.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr, softmax

PKG = Path(__file__).resolve().parents[1]
RAW = PKG / "data" / "raw"
MOVE_SQUARES = [s for s in range(64) if s not in (27, 28, 35, 36)]
TOKEN_OF = {s: i + 1 for i, s in enumerate(MOVE_SQUARES)}


def js(p, q):
    """scipy's JS on the smoothed inputs, plus the original code's second smoothing inside each KL term."""
    sm = lambda x: (x + 1e-12) / (x + 1e-12).sum()  # noqa: E731
    p, q = sm(p), sm(q)
    single = float(jensenshannon(p, q, base=np.e) ** 2)
    m = 0.5 * (p + q)
    double = 0.5 * float(rel_entr(sm(p), sm(m)).sum()) + 0.5 * float(rel_entr(sm(q), sm(m)).sum())
    assert abs(single - double) < 1e-9, (single, double)
    return double


def y_score(logits, legal, illegal):
    p = softmax(np.asarray(logits, dtype=np.float64))
    return float(p[[TOKEN_OF[s] for s in legal]].sum() - p[[TOKEN_OF[s] for s in illegal]].sum())


def boot(values, clusters, seed, n_boot=2000):
    groups = defaultdict(list)
    order = []
    for v, c in zip(values, clusters):
        if c not in groups:
            order.append(c)
        groups[c].append(v)
    cm = np.array([np.mean(groups[c]) for c in order])
    rng = np.random.default_rng(seed)
    reps = np.array([cm[rng.integers(0, len(cm), len(cm))].mean() for _ in range(n_boot)])
    return {"point": float(np.mean(values)), "lo": float(np.quantile(reps, 0.025)), "hi": float(np.quantile(reps, 0.975)),
            "wins": int(np.sum(np.asarray(values) > 0))}


def pairs_from(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return [r for r in rows if r["cell"] == "pair_contrast"]


def fixed_js():
    L = np.load(RAW / "fixed_js" / "fixed_js_full_logits.npz")
    P = np.load(RAW / "fixed_js" / "fixed_js_probe_probabilities.npz")
    sets = json.load(open(RAW / "spec" / "fiber_spec_target_legal_sets.json", encoding="utf-8"))
    pairs = pairs_from(RAW / "fixed_js" / "fixed_js_interventions.csv")
    d_w, d_r, probe, jserr, recips, cats, exact = [], [], 0.0, 0.0, [], [], []
    for pr in pairs:
        k = f"{pr['recip_game']}__{pr['target_game']}__"
        s = sets[f"{pr['recip_game']}|{pr['target_game']}"]
        y = lambda c: y_score(L[k + c], s["became_legal"], s["became_illegal"])  # noqa: E731
        yc = y("correct_null")
        d_w.append(yc - np.mean([y(f"wrong_null_{j}") for j in range(3)]))
        d_r.append(yc - np.mean([y(f"random_null_{j}") for j in range(3)]))
        ref = softmax(L[k + "editor_A"])
        for c in ["correct_null"] + [f"wrong_null_{j}" for j in range(3)] + [f"random_null_{j}" for j in range(3)]:
            probe = max(probe, float(np.abs(P[k + c] - P[k + "editor_A"]).max()))
            jserr = max(jserr, abs(js(ref, softmax(L[k + c])) - 0.01))
        recips.append(pr["recip_game"])
        cats.append(pr["r2_type"][0])
        exact.append(float(pr["board_exact_editor"]) > 0.5)
    out = {"fixed_js/slices/pooled/delta_specific": boot(d_w, recips, 82), "fixed_js/slices/pooled/delta_random": boot(d_r, recips, 83)}
    for name, mask in (("A", [c == "A" for c in cats]), ("nonexact_editor", [not e for e in exact])):
        idx = [i for i, m in enumerate(mask) if m]
        out[f"fixed_js/slices/{name}/delta_random"] = boot([d_r[i] for i in idx], [recips[i] for i in idx], 83)
    return out, {"fixed_js/summary/probe_max_abs_vs_editor_A": probe, "fixed_js/summary/max_abs_js_minus_target": jserr,
                 "fixed_js/summary/n_pairs": len(pairs)}


def spec():
    L = np.load(RAW / "spec" / "fiber_spec_full_logits.npz")
    sets = json.load(open(RAW / "spec" / "fiber_spec_target_legal_sets.json", encoding="utf-8"))
    pairs = pairs_from(RAW / "spec" / "fiber_spec_interventions.csv")
    d_w, d_r, recips = [], [], []
    for pr in pairs:
        k = f"{pr['recip_game']}__{pr['target_game']}__"
        s = sets[f"{pr['recip_game']}|{pr['target_game']}"]
        y = lambda c: y_score(L[k + c], s["became_legal"], s["became_illegal"])  # noqa: E731
        d_w.append(y("correct_null") - np.mean([y(f"wrong_null_{j}") for j in range(3)]))
        d_r.append(y("correct_null") - np.mean([y(f"random_null_{j}") for j in range(3)]))
        recips.append(pr["recip_game"])
    return {"spec/slices/pooled/delta_specific": boot(d_w, recips, 82), "spec/slices/pooled/delta_random": boot(d_r, recips, 83)}


def nst():
    L = np.load(RAW / "nst" / "null_state_full_logits.npz")
    sets = json.load(open(RAW / "nst" / "null_state_target_legal_sets.json", encoding="utf-8"))
    pairs = pairs_from(RAW / "nst" / "null_state_interventions.csv")
    contrasts = {m: [] for m in ("wrong", "random", "mean", "editor")}
    cell = {"wrong": "wrong_target", "random": "random", "mean": "mean_state", "editor": "editor_A"}
    recips, y_learned, token_ok = [], [], True
    for pr in pairs:
        k = f"{pr['recip_game']}__{pr['target_game']}__"
        s = sets[f"{pr['recip_game']}|{pr['target_game']}"]
        token_ok &= [TOKEN_OF[q] for q in s["became_legal_squares"]] == s["became_legal_tokens"]
        token_ok &= [TOKEN_OF[q] for q in s["became_illegal_squares"]] == s["became_illegal_tokens"]
        y = lambda c: y_score(L[k + c], s["became_legal_squares"], s["became_illegal_squares"])  # noqa: E731
        yl = y("learned")
        y_learned.append(yl)
        for m, c in cell.items():
            contrasts[m].append(yl - y(c))
        recips.append(pr["recip_game"])
    out = {f"nst/slices/pooled/{m}": boot(v, recips, 82 + i) for i, (m, v) in enumerate(contrasts.items())}
    return out, {"nst/summary/mean_Y/learned": float(np.mean(y_learned)), "nst/summary/n_Y_learned_positive": int(np.sum(np.array(y_learned) > 0)),
                 "token_list_matches_saved_tokens": bool(token_ok)}, contrasts["editor"]


def stage_share(editor_contrast):
    a = {v["id"]: v["value"] for v in json.load(open(PKG / "data" / "archival" / "archival_values.json", encoding="utf-8"))["values"]}
    full_editor = float(np.mean(editor_contrast))
    return (full_editor - (a["sem.full_minus_legal"] + a["sem.legal_minus_stage"])) / full_editor


def main() -> int:
    res = json.load(open(PKG / "analysis" / "results.json", encoding="utf-8"))

    def get(path):
        cur = res
        for p in path.split("/"):
            cur = cur[p]
        return cur

    rows, bad = [], []

    def cmp(name, mine, theirs, tol):
        ok = abs(mine - theirs) <= tol
        rows.append((name, mine, theirs, abs(mine - theirs), ok))
        if not ok:
            bad.append(name)

    fj, fjs = fixed_js()
    sp = spec()
    ns, nss, editor_contrast = nst()
    for block in (fj, sp, ns):
        for path, b in block.items():
            ref = get(path)
            cmp(path + "/point", b["point"], ref["point"], 1e-9)
            cmp(path + "/ci95_low", b["lo"], ref["ci95_low"], 1e-9)
            cmp(path + "/ci95_high", b["hi"], ref["ci95_high"], 1e-9)
            cmp(path + "/wins", b["wins"], ref["wins"], 0)
    for path, v in list(fjs.items()) + [(k, v) for k, v in nss.items() if "/" in k]:
        cmp(path, float(v), float(get(path)), 1e-9 * max(1.0, abs(float(v))) if "probe" not in path and "js" not in path else 1e-9)
    cmp("stage_share/share", stage_share(editor_contrast), get("stage_share/share"), 1e-9)
    if not nss["token_list_matches_saved_tokens"]:
        bad.append("token list disagrees with the saved NST legal-token lists")
    w = max(len(r[0]) for r in rows)
    for name, mine, theirs, diff, ok in rows:
        print(f"{'OK ' if ok else 'BAD'} {name:<{w}}  harness={mine:.10g}  analysis={theirs:.10g}  |diff|={diff:.2e}")
    print(f"token list matches saved NST tokens: {nss['token_list_matches_saved_tokens']}")
    print(f"\n{len(rows) - len(bad)} of {len(rows)} comparisons agree" + ("" if not bad else f"; MISMATCH: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
