"""Track B operating characteristics v2 (review B r2 N3): on the REAL frozen-type discriminating set (tie-aware rules, T4 generator),
with model errors calibrated to development error rates (EARLY 2.6%, MIXTURE 1.4% of discriminating decisions; errors land uniformly on
the two non-Bayes options), parents with BOTH bases (native and induced organisations), between-parent SD 0.01 of every planted effect,
through the REAL track_b scorer.  Scenarios: precision difference only; + directed effect toward 'local' (MIXTURE class); + history-only
effect (blocked models differ, no class effect).  Gates: P(B1 | precision only) <= 0.05; P(B1 | history only) <= 0.05.
Writes to $COR_TEST_OUT (default: a temporary directory), never into the package."""
import os, sys, json, tempfile
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG); os.environ.setdefault("COR_DEVICE", "cpu")
import numpy as np
from cor_research import env as E
E.configure("cpu")
from cor_research.scoring import track_b as TB
from cor_research.generators import t4 as T4
Xd, qd, _ = T4.discriminating(20000, 91_410_000); Rd = T4.rules(Xd, qd); RD = {k: T4.decide(Rd[k]) for k in ("bayes",) + T4.RULES}; B0 = RD["bayes"]; n = len(B0)
S_local = (RD["local"] >= 0) & (RD["local"] != B0)
EARLY = dict(KL=0.03, acc=0.85, nec_E=1.0, nec_M=0.005, ckpt_sha256="h"); MIX = dict(KL=0.012, acc=0.85, nec_E=0.001, nec_M=0.8, ckpt_sha256="h")
root = os.environ.get("COR_TEST_OUT") or tempfile.mkdtemp(); os.makedirs(os.path.join(root, "trackB"), exist_ok=True)
def model(rng, err, f_dir=0.0, hist=0.0, fixed=None):
    d = B0.copy(); m = rng.random(n) < err; d[m] = (B0[m] + rng.integers(1, 3, m.sum())) % 3
    if f_dir: mm = fixed["dir"]; d[mm] = RD["local"][mm]
    if hist: mm = fixed["hist"]; d[mm] = (B0[mm] + 2) % 3
    return d
def units(rng, nparents, f_dir=0.0, hist=0.0, nD=4):
    U = []
    for i in range(nparents):
        p = f"nP{9300 + i}"; fx = dict(dir=S_local & (rng.random(n) < max(f_dir + 0.01 * rng.standard_normal(), 0)), hist=rng.random(n) < max(hist + 0.01 * rng.standard_normal(), 0) if hist else None)
        for base in "EM":
            for d_ in range(2001, 2001 + nD):
                for arm in ("fullB", "fullI"):
                    cls_ = ("EARLY" if arm == "fullB" else "MIXTURE") if base == "E" else ("MIXTURE" if arm == "fullB" else "EARLY")
                    dec = model(rng, 0.026 if cls_ == "EARLY" else 0.014, f_dir if cls_ == "MIXTURE" else 0.0, hist if arm == "fullI" else 0.0, fx)
                    tg = f"oc_{p}_{base}{d_}_{arm}"
                    for lev in ("matched", "final"): np.savez(os.path.join(root, "trackB", f"{tg}_{lev}.npz"), dec=dec.astype(np.int8), **{f"rule_{k}": v for k, v in RD.items()})
                    ic = {f"{k}_{l}": dict(IIA_norm=0.5) for k in ("OQI", "OQI_unmatched", "RS") for l in range(4)}; ic["null_base_patch_max_logit_diff"] = 0.0
                    kl = (0.030 if cls_ == "EARLY" else 0.015) + 0.001 * rng.standard_normal()
                    fn = dict(ckpt_sha256="h", natural=dict(KL=kl, rules={k: dict(w=0.05) for k in T4.RULES}, nulls=dict(parametric=dict(rules={k: dict(w=0.04) for k in T4.RULES}))), interchange=ic)
                    U.append(dict(key=f"{p}:{base}{d_}:{arm}", parent=p, base=base, D=d_, arm=arm, tag=tg, status="PRESENT",
                                  rec=dict(analyses={lev: dict(step=1000, necessity=dict(EARLY if cls_ == "EARLY" else MIX), functional=fn, decisions_sha256=E.sha256(os.path.join(root, "trackB", f"{tg}_{lev}.npz"))) for lev in ("matched", "final")})))
    return ({}, U, [])
def rate(reps, **kw):
    out = {}
    for r in range(reps):
        v = TB.score("standard", root, frozen_code=None, inputs=units(np.random.default_rng(500 + r), **kw))["verdict"].split(" [")[0]; out[v] = out.get(v, 0) + 1 / reps
    return out
if __name__ == "__main__":
    R = {}
    for npar in (4, 6):
        R[f"{npar}p_precision_only"] = rate(10, nparents=npar); R[f"{npar}p_directed_0.15"] = rate(10, nparents=npar, f_dir=0.15); R[f"{npar}p_history_only_0.05"] = rate(10, nparents=npar, hist=0.05)
    R["discriminating_set"] = dict(n=int(n), frac_S_local=float(S_local.mean()))
    json.dump(R, open(os.path.join(root, "trackb_oc_results.json"), "w"), indent=1)
    for k, v in R.items(): print(k, {a: round(b, 2) for a, b in v.items()} if k != "discriminating_set" else v)
    ok = all(R[f"{n_}p_precision_only"].get("B1-FUNCTIONAL", 0) <= 0.05 and R[f"{n_}p_history_only_0.05"].get("B1-FUNCTIONAL", 0) <= 0.05 for n_ in (4, 6))
    print("PASS" if ok else "FAIL", "results:", os.path.join(root, "trackb_oc_results.json")); sys.exit(0 if ok else 1)
