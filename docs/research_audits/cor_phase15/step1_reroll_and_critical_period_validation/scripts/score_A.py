"""XIII-E Workstream A confirmation scorer (frozen in freeze A-2).  FAIL-CLOSED: it refuses to produce any verdict unless every output listed
in logs/manifest_A.json exists, has the expected configuration, and is complete; on refusal it writes nothing and exits with code 2.
Usage: score_A.py [LOGDIR] [CKPTDIR] -> LOGDIR/score_A.json
Statistics (all exact or Monte Carlo with fixed seeds):
  * per anchor x re-draw type: counts of EARLY / MIXTURE / HYBRID at 1211 with Clopper-Pearson 95% intervals; continuous lik_RO1, mix_mass.
  * conditional comparisons (A4): P(early) after W (L) re-draws from early vs mixture anchors, per parent, Newcombe 95% interval for the
    difference; homogeneity of P(early) across the four anchors within a re-draw type (exact Fisher-Freeman-Halton by Monte Carlo).
  * parent baselines: natural-diagonal class frequencies (XIII-B) vs re-draw frequencies.
  * R1 transfer tests (the frozen predictions): for each re-draw type T in {W, L}, every new seed s is observed in two contexts per parent
    (fixed components of the early anchor and of the mixture anchor).  Statistic rho_T = mean over parents of Spearman(mix_mass@1211 in the
    early-anchor context, mix_mass@1211 in the mixture-anchor context) across the 8 seeds.  Null (no transferable component of the re-drawn
    tensor = re-roll): seed labels exchangeable within parent; stratified permutation, 20 000 draws, seed 0; one-sided p (positive).
    Upper bound: 95th percentile of rho_T under a stratified bootstrap of seeds (10 000 draws, seed 1).
    Verdict per T: TRANSFER if p < 0.05 and rho_T >= 0.3; NO-TRANSFER if p >= 0.05 and the upper bound < 0.6; otherwise INCONCLUSIVE (bound chosen by the pre-freeze power simulation, logs/A_power.json).
    Frozen prediction R1-T (re-roll null, no transferable component): CONFIRMED iff NO-TRANSFER, FALSIFIED iff TRANSFER, else INCONCLUSIVE.
  * class concordance across contexts (exact): for each T, number of seeds with the same class in both contexts vs the expectation under
    independence given the context-specific rates.
  * A5 stability: 1211 class vs 4000 class for the continued runs.
  * threshold sensitivity: counts recomputed with the mix thresholds 0.5/0.8 shifted by +-0.05 and the lik threshold 0.4 by +-0.05.
  * exploratory (not a verdict): AUC of the step-0 statistic R_route (logs/r2_A.jsonl) for EARLY at 1211, direction pre-declared negative."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs")
CK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "checkpoints", "ft")
import numpy as np
from scipy.stats import spearmanr, beta
def refuse(msg): print("REFUSED:", msg, file=sys.stderr); sys.exit(2)
try: MAN = json.load(open(os.path.join(LD, "manifest_A.json")))
except Exception as ex: refuse(f"manifest unreadable: {ex}")
DES = MAN["design"]
# ---------- completeness and configuration checks (fail closed) ----------
for r in DES["runs"]:
    d = os.path.join(CK, r["tag"])
    for f in ("stop_done.json", "step1211.npz", "opt_step1211.pkl"):
        if not os.path.exists(os.path.join(d, f)): refuse(f"missing {r['tag']}/{f}")
    sd = json.load(open(os.path.join(d, "stop_done.json")))
    if sd.get("stop") != 1211: refuse(f"{r['tag']} stop_done has stop {sd.get('stop')}")
def jl(f):
    p = os.path.join(LD, f)
    if not os.path.exists(p): refuse(f"missing {f}")
    out = {}
    for l in open(p):
        try: x = json.loads(l)
        except Exception: refuse(f"malformed line in {f}")
        out[(x["unit"], x["step"])] = x
    return out
UM = jl("unit_metrics_A.jsonl")
for r in DES["runs"]:
    if (r["spec"], 1211) not in UM: refuse(f"no 1211 metrics for {r['spec']}")
    m = UM[(r["spec"], 1211)]; fac = m.get("factors")
    if fac is None or [fac[0], fac[1], fac[2], fac[3]] != [r["parent"], r["L"], r["W"], r["D"]]: refuse(f"configuration mismatch for {r['spec']}")
try: CONT = json.load(open(os.path.join(LD, "manifest_A_cont.json")))["continuation_final"]
except Exception: refuse("continuation set not finalised (logs/manifest_A_cont.json from select_continuation.py)")
if not set(MAN["design"]["continuation"]["fixed"]) <= set(CONT) or len(CONT) != len(MAN["design"]["continuation"]["fixed"]) + 2: refuse("continuation set does not follow the frozen rule")
UC = jl("unit_metrics_Acont.jsonl")
for tag in CONT:
    if not os.path.exists(os.path.join(CK, tag.replace("_stop1211", "") + "_br1211", "final.json")): refuse(f"continuation {tag} not finished")
    spec = next(r["spec"] for r in DES["runs"] if r["tag"] == tag).replace(":stop1211", ":br1211")
    if (spec, 4000) not in UC: refuse(f"no 4000 metrics for {spec}")
R2 = None
if os.path.exists(os.path.join(LD, "r2_A.jsonl")): R2 = {json.loads(l)["config"]: json.loads(l) for l in open(os.path.join(LD, "r2_A.jsonl"))}
# ---------- analysis ----------
TH = dict(e_mix=0.5, e_lik=0.4, m_mix=0.8, m_lik=0.15)
def cls(l, m, th=TH): return "EARLY" if (m < th["e_mix"] and l >= th["e_lik"]) else ("MIXTURE" if (m >= th["m_mix"] and l <= th["m_lik"]) else "HYBRID")
def cls4000(l, m): return "early" if (l >= 0.6 and m < 0.5) else ("mixture" if (m >= 0.8 and l <= 0.15) else "hybrid")
rows = []
for r in DES["runs"]:
    m = UM[(r["spec"], 1211)]; rows.append(dict(r, lik=m["lik_RO1"], mix=m["mix_mass"], KL=m["KL"], acc=m["acc"], route_Q=m.get("route_Q"), cls=cls(m["lik_RO1"], m["mix_mass"]), competent=m["KL"] <= 0.05))
def cp(k, n): return [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]
res = dict(table=rows, cells={}, comparisons={}, R1={}, concordance={}, stability={}, sensitivity={}, exploratory={})
A = {a["id"]: a for a in DES["anchors"]}
for a in A:
    for T in ("W", "L"):
        rs = [x for x in rows if x["anchor"] == a and x["type"] == T]; n = len(rs)
        c = {k: sum(x["cls"] == k for x in rs) for k in ("EARLY", "MIXTURE", "HYBRID")}
        res["cells"][f"{a}:{T}"] = dict(n=n, counts=c, P_early=c["EARLY"] / n, CI_early=cp(c["EARLY"], n), P_mixture=c["MIXTURE"] / n, CI_mixture=cp(c["MIXTURE"], n),
                                        lik=[x["lik"] for x in rs], mix=[x["mix"] for x in rs], n_competent=sum(x["competent"] for x in rs))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
for p in sorted({a["parent"] for a in A.values()}):
    e = next(k for k, v in A.items() if v["parent"] == p and v["known"] == "early"); m = next(k for k, v in A.items() if v["parent"] == p and v["known"] == "mixture")
    for T in ("W", "L"):
        k1, n1 = res["cells"][f"{e}:{T}"]["counts"]["EARLY"], 8; k2, n2 = res["cells"][f"{m}:{T}"]["counts"]["EARLY"], 8
        l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); d = k1 / n1 - k2 / n2
        lo = d - np.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2); hi = d + np.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2)
        res["comparisons"][f"{p}:{T}:early_anchor_minus_mixture_anchor"] = dict(diff=d, newcombe95=[lo, hi])
rng = np.random.default_rng(2)
for T in ("W", "L"):
    tab = np.array([[res["cells"][f"{a}:{T}"]["counts"][k] for k in ("EARLY", "MIXTURE", "HYBRID")] for a in A])
    def stat(tb):
        e = tb.sum(1, keepdims=True) * tb.sum(0, keepdims=True) / tb.sum(); m = e > 0; return float((((tb - e) ** 2)[m] / e[m]).sum())
    obs = stat(tab); lab = np.repeat(np.arange(4), tab.sum(1)); outc = np.concatenate([np.repeat(np.arange(3), r) for r in tab]); cnt = 0
    for _ in range(20000):
        pr = rng.permutation(outc); tb = np.zeros((4, 3));
        for i, j in zip(lab, pr): tb[i, j] += 1
        cnt += stat(tb) >= obs - 1e-12
    res["comparisons"][f"homogeneity_across_anchors:{T}"] = dict(chi2=obs, p_mc=(cnt + 1) / 20001)
res["comparisons"]["parent_baselines_natural_diagonals"] = MAN["parent_baselines"]
for T in ("W", "L"):
    seeds = DES[f"{T}_seeds"]; per = []; pairs = {}
    for p in sorted({a["parent"] for a in A.values()}):
        e = next(k for k, v in A.items() if v["parent"] == p and v["known"] == "early"); m = next(k for k, v in A.items() if v["parent"] == p and v["known"] == "mixture")
        get = lambda anc, s: next(x for x in rows if x["anchor"] == anc and x["type"] == T and x[T] == s)
        xe = np.array([get(e, s)["mix"] for s in seeds]); xm = np.array([get(m, s)["mix"] for s in seeds]); pairs[p] = (xe, xm)
        ce = [get(e, s)["cls"] for s in seeds]; cm = [get(m, s)["cls"] for s in seeds]
        same = sum(a == b for a, b in zip(ce, cm)); pe = {k: ce.count(k) / 8 for k in ("EARLY", "MIXTURE", "HYBRID")}; pm = {k: cm.count(k) / 8 for k in pe}
        res["concordance"][f"{p}:{T}"] = dict(same_class=same, expected_if_independent=8 * sum(pe[k] * pm[k] for k in pe), classes_early_ctx=ce, classes_mixture_ctx=cm)
    def rho(prs): return float(np.mean([0.0 if (np.ptp(a) == 0 or np.ptp(b) == 0) else spearmanr(a, b)[0] for a, b in prs.values()]))
    obs = rho(pairs); r0 = np.random.default_rng(0); ge = 0
    for _ in range(20000): ge += rho({p: (a, r0.permutation(b)) for p, (a, b) in pairs.items()}) >= obs - 1e-12
    pv = (ge + 1) / 20001; r1 = np.random.default_rng(1); bs = []
    for _ in range(10000):
        bp = {}
        for p, (a, b) in pairs.items(): i = r1.integers(0, 8, 8); bp[p] = (a[i], b[i])
        bs.append(rho(bp))
    ub = float(np.percentile(bs, 95))
    v = "TRANSFER" if (pv < 0.05 and obs >= 0.3) else ("NO-TRANSFER" if (pv >= 0.05 and ub < 0.6) else "INCONCLUSIVE")
    res["R1"][f"R1-{T}"] = dict(rho=obs, p_one_sided=pv, upper95=ub, outcome=v, verdict={"NO-TRANSFER": "CONFIRMED", "TRANSFER": "FALSIFIED"}.get(v, "INCONCLUSIVE"),
                                per_parent={p: (0.0 if (np.ptp(a) == 0 or np.ptp(b) == 0) else float(spearmanr(a, b)[0])) for p, (a, b) in pairs.items()})
st = []
for tag in CONT:
    r = next(x for x in rows if x["tag"] == tag); spec = r["spec"].replace(":stop1211", ":br1211"); m = UC[(spec, 4000)]
    st.append(dict(tag=tag, cls1211=r["cls"], lik1211=r["lik"], mix1211=r["mix"], cls4000=cls4000(m["lik_RO1"], m["mix_mass"]), lik4000=m["lik_RO1"], mix4000=m["mix_mass"], KL4000=m["KL"]))
agree = sum(s["cls1211"].lower() == s["cls4000"] for s in st); res["stability"] = dict(runs=st, agree=agree, n=len(st))
for nm, th in (("mix-0.05", dict(TH, e_mix=0.45, m_mix=0.75)), ("mix+0.05", dict(TH, e_mix=0.55, m_mix=0.85)), ("lik-0.05", dict(TH, e_lik=0.35)), ("lik+0.05", dict(TH, e_lik=0.45))):
    res["sensitivity"][nm] = {f"{a}:{T}": sum(cls(x["lik"], x["mix"], th) == "EARLY" for x in rows if x["anchor"] == a and x["type"] == T) for a in A for T in ("W", "L")}
if R2:
    byc = R2; x = []; yv = []
    for r in rows:
        c = f'{r["parent"]}:{r["L"]}:{r["W"]}'
        if c in byc: x.append(byc[c]["R_route"]); yv.append(r["cls"] == "EARLY")
    x = np.array(x); yv = np.array(yv)
    if yv.any() and (~yv).any(): res["exploratory"]["R_route_AUC_negative_direction"] = float(np.mean([(a < b) + 0.5 * (a == b) for a in x[yv] for b in x[~yv]]))
res["summary"] = {k: v["verdict"] for k, v in res["R1"].items()}
json.dump(res, open(os.path.join(LD, "score_A.json"), "w"), indent=1, default=float); print(res["summary"])
