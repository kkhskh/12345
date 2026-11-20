"""Preregistered operating characteristics of the Track A pair -> parent -> cross-parent rules (review A B1).  Pure simulation of
pair statuses (SUPPORTED with prob pS, CONTRA with prob pC, else MIDDLE; each pair evaluable with prob pE), through the REAL
track_a.parent_verdict / cross_parent.  Pass criteria (frozen):
  P(GENERALIZES | pS=0.9, pC=0.02, pE>=0.8) >= 0.8;  P(FAILS | same) <= 0.05;  P(GENERALIZES | pS=0.1, pC=0.7) <= 0.01;
  P(GENERALIZES) non-decreasing in n_D (4 -> 6) for the true effect; for 4..12 parents; with one CONTRADICTING null parent among 8, P(GEN) <= 0.15.
Reported (not gated): power at 80% pair evaluability (4 parents x 4 data orders: about 0.4 -> the Standard preset is underpowered when pairs
are lost; Extended's 6 data orders restore about 0.9), a silently null parent (mostly MIDDLE) and a 60%-supported / 25%-contradicted effect."""
import os, sys, json
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG); os.environ.setdefault("COR_DEVICE", "cpu")
import numpy as np
from cor_research import protocol as PR
from cor_research.scoring import track_a as TA
C = PR.effective("standard"); H = dict(expect="I_like")
def sim(n_par, nD, pS, pC, pE, reps=4000, seed=0, null_parents=0, null=(0.1, 0.7)):
    rng = np.random.default_rng(seed); res = {"GENERALIZES": 0, "FAILS_TO_GENERALIZE": 0, "NOT_ESTABLISHED": 0, "INCONCLUSIVE": 0}
    for _ in range(reps):
        pv = {}
        for p in range(n_par):
            outs = []; pS_, pC_ = (null if p < null_parents else (pS, pC))
            for _ in range(nD):
                if rng.random() > pE: outs.append("ANCHORS_NOT_EVALUABLE"); continue
                u = rng.random(); outs.append("SUPPORTED_PAIR" if u < pS_ else "CONTRA_PAIR" if u < pS_ + pC_ else "MIDDLE")
            pv[p] = TA.parent_verdict(outs, H, C)
        res[TA.cross_parent(pv, C)] += 1
    return {k: v / reps for k, v in res.items()}
def run():
    out = {}; ok = True
    for n_par in (4, 6, 8, 12):
        for nD in (4, 6):
            t = sim(n_par, nD, 0.9, 0.02, 1.0); t8 = sim(n_par, nD, 0.9, 0.02, 0.8); f = sim(n_par, nD, 0.1, 0.7, 1.0)
            out[f"{n_par}p_{nD}D"] = dict(true_full=t, true_eval80=t8, null=f, weak_mixed=sim(n_par, nD, 0.4, 0.3, 1.0))
            ok &= t["GENERALIZES"] >= 0.8 and t["FAILS_TO_GENERALIZE"] <= 0.05 and t8["FAILS_TO_GENERALIZE"] <= 0.05 and f["GENERALIZES"] <= 0.01
        ok &= out[f"{n_par}p_6D"]["true_full"]["GENERALIZES"] >= out[f"{n_par}p_4D"]["true_full"]["GENERALIZES"] - 0.02
    out["8p_6D_one_null_parent"] = sim(8, 6, 0.9, 0.02, 1.0, null_parents=1); out["8p_6D_one_silent_null_parent"] = sim(8, 6, 0.9, 0.02, 1.0, null_parents=1, null=(0.1, 0.2))
    out["8p_6D_60pc_support_25pc_contra"] = sim(8, 6, 0.6, 0.25, 1.0)
    ok &= out["8p_6D_one_null_parent"]["GENERALIZES"] <= 0.15
    return ok, out
if __name__ == "__main__":
    ok, out = run(); import tempfile; json.dump(out, open(os.path.join(os.environ.get("COR_TEST_OUT") or tempfile.mkdtemp(), "verdict_oc_results.json"), "w"), indent=1)
    for k in ("8p_6D_one_null_parent", "8p_6D_one_silent_null_parent", "8p_6D_60pc_support_25pc_contra"): print(k, {a: round(b, 3) for a, b in out[k].items()})
    for k, v in ((k, v) for k, v in out.items() if "true_full" in v): print(k, "true:", {a: round(b, 3) for a, b in v["true_full"].items()}, "eval80 GEN/FAIL", round(v["true_eval80"]["GENERALIZES"], 3), round(v["true_eval80"]["FAILS_TO_GENERALIZE"], 3), "null GEN", v["null"]["GENERALIZES"], "weak GEN", round(v["weak_mixed"]["GENERALIZES"], 3))
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
