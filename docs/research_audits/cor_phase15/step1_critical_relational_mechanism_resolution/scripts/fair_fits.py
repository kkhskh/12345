"""XIII-C Workstream B2 + B5: fair, nested decision-law fits with a proper scoring rule, multi-start, held-out evaluation, bootstrap
uncertainty, and a Shapley decomposition of each model's deviation from Bayes over PARAMETER GROUPS.

Common parametrisation (every family nests exact Bayes):
  weights   w_g = softmax_g( a log W_g + k log max(f_g,1) + d [f_g = 0] )      Bayes: a = 1, k = 1, d = -inf (here -40)
  combine   M(y) = (1/rho) log sum_g w_g L_g(y)^rho                             Bayes: rho = 1   (T2: == misstated reliability alpha'(rho) at q)
  pooled    P(y) = sum_s log c_s(y)  over the 6 sources at q, UNGATED            (early-pooling hypothesis; Bayes: gamma = 0)
  output    log p = tau * [(1 - gamma) M(y) + gamma P(y)]  then lapse  p <- (1 - lam) p + lam/3      Bayes: tau = 1, gamma = 0, lam = 0
Parameter groups for B5:  CAL = {tau},  COMB = {rho},  REL = {a},  ELIG = {k, d},  POOL = {gamma},  LAPSE = {lam}.
Objective: KL(model || fit) on 3000 training episodes (the model's own predictive distribution is the target: a proper scoring rule);
reported on 3000 held-out episodes: KL, mean |Δp|, top-candidate log-prob error, KL split by the number of eligible groups S, and calibration
of the fit against the model (ECE of fitted top probability vs model's top probability).
Families reported (B2): BAYES, TEMP{CAL}, MIX{CAL,REL,ELIG}, PM{+COMB}, MIXL{MIX+LAPSE}, PML{PM+LAPSE}, NPOE{MIX with rho = 1e-3}, HYB{MIX+POOL},
  FULL{all six groups}.  Bootstrap (B=8 resamples of the training set) for rho in PM and PML.
B5: Shapley shares over the six groups of the KL reduction KL(model||BAYES) - KL(model||fit_S), all 64 subsets (2 starts each).
Convergence: every family uses >= 2 starts; FULL additionally starts from the best of PML/HYB/PM (it nests them) and Shapley subsets start
also from the FULL optimum restricted to their free parameters (added after development showed FULL < PML on one unit: a local optimum).
Usage: FF_STREAM=86300000 FF_TAG=_dev python3 scripts/fair_fits.py SPEC...   -> logs/fair_fits<TAG>.json (resumable per unit)"""
import json, os, sys, itertools, math, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax
from scipy.optimize import minimize
from scipy.special import logsumexp as lse
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore"); import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(int(os.environ.get("FF_STREAM", 86_300_000))); N_ = 6000
X = NV.Bm.sample_fast(NV.ENV, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
W, Lg, pf, e, S = RP.parts(X, q); f = S - e; lW, lL = np.log(W), np.log(Lg); Sn = S[:, 0].astype(int)
a_ = X.reshape(N_, 6, 3, 2)[ii, q].reshape(N_, 6); y = np.arange(3); al = 0.8
POOLED = np.where(a_[..., None] == y, np.log(al), np.log((1 - al) / 2)).sum(1)          # (N,3) ungated sum of per-source log-evidence
tr, te = ii[:3000], ii[3000:]
GROUPS = dict(CAL=["tau"], COMB=["rho"], REL=["a"], ELIG=["k", "d"], POOL=["gamma"], LAPSE=["lam"])
DEF = dict(a=1.0, k=1.0, d=-40.0, tau=1.0, rho=1.0, gamma=0.0, lam=0.0)
def logq(p, idx):
    z = p["a"] * lW[idx] + p["k"] * np.log(np.maximum(f[idx], 1)) + p["d"] * (f[idx] == 0); lw = z - lse(z, 1, keepdims=True)
    rho = float(np.clip(p["rho"], 1e-3, 6)); Mx = lse(lw[..., None] + rho * lL[idx], 1) / rho
    s = p["tau"] * ((1 - p["gamma"]) * Mx + p["gamma"] * POOLED[idx]); lp = s - lse(s, -1, keepdims=True)
    lam = float(np.clip(p["lam"], 0, 0.5))
    return np.log((1 - lam) * np.exp(lp) + lam / 3) if lam > 0 else lp
def kl(P, lq, idx): return float((P[idx] * (np.log(P[idx] + 1e-300) - lq)).sum(-1).mean())
def fit(P, free, idx=tr, starts=2, fixed=None, extra=None):
    base = dict(DEF, **(fixed or {}))
    if not free: return base, kl(P, logq(base, idx), idx)
    x0s = [[base[k] for k in free]] + ([[extra.get(k, base[k]) for k in free]] if extra else [])
    alt = dict(a=0.8, k=0.6, d=-6.0, tau=1.2, rho=0.6, gamma=0.2, lam=0.02); x0s.append([alt[k] for k in free])
    if starts > 2: alt2 = dict(a=1.2, k=1.4, d=-3.0, tau=0.8, rho=1.5, gamma=0.5, lam=0.05); x0s.append([alt2[k] for k in free])
    best = None; funs = []
    for x0 in x0s[:max(starts, 2) + (1 if extra else 0)]:
        r = minimize(lambda x: kl(P, logq(dict(base, **dict(zip(free, x))), idx), idx), x0, method="Powell", options=dict(maxiter=6000, xtol=1e-4, ftol=1e-9))
        funs.append(r.fun)
        if best is None or r.fun < best.fun: best = r
    p = dict(base, **dict(zip(free, best.x))); p["_spread"] = float(np.ptp(funs)); return p, float(best.fun)
def report(P, p):
    lq = logq(p, te); Q = np.exp(lq); top = P[te].argmax(-1); j = np.arange(len(te))
    byS = {str(s): kl(P[te][Sn[te] == s], lq[Sn[te] == s], slice(None)) for s in (1, 2, 3) if (Sn[te] == s).sum() > 20}
    ece = float(np.abs(Q[j, top] - P[te][j, top]).mean())
    return dict(KL_test=kl(P, lq, te), mad=float(np.abs(Q - P[te]).mean()), top_logp_err=float(np.abs(lq[j, top] - np.log(P[te][j, top] + 1e-300)).mean()),
                KL_by_S=byS, top_prob_gap=ece, params={k: float(v) for k, v in p.items() if not k.startswith("_")}, start_spread=p.get("_spread", 0.0))
FAMS = dict(BAYES=[], TEMP=["tau"], MIX=["tau", "a", "k", "d"], PM=["tau", "a", "k", "d", "rho"], MIXL=["tau", "a", "k", "d", "lam"],
            PML=["tau", "a", "k", "d", "rho", "lam"], HYB=["tau", "a", "k", "d", "gamma"], FULL=["tau", "a", "k", "d", "rho", "gamma", "lam"])
outp = os.path.join(HERE, "..", "logs", "fair_fits" + os.environ.get("FF_TAG", "") + ".json"); res = json.load(open(outp)) if os.path.exists(outp) else {}
PB = GL.posterior_closed(RP.ENV1, X, q)
targets = [("BAYES_SELF", None)] + [(sp, sp) for sp in sys.argv[1:]]
for name, sp in targets:
    if name in res: continue
    t0 = time.time()
    if sp is None: P = PB
    else:
        M = xc.load(sp); L = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), X))[ii, q]; P = RP.sm(L)
    r = dict(fams={}, KL_model_vs_bayes_test=kl(P, np.log(PB[te] + 1e-300), te))
    best_p = {}
    for fam, free in FAMS.items():
        ext = None
        if fam == "FULL":                                                     # nested-start: FULL must be at least as good as PML and HYB
            ext = min((best_p[f] for f in ("PML", "HYB", "PM")), key=lambda pp: kl(P, logq(pp, tr), tr))
        p, _ = fit(P, free, starts=3 if fam in ("PM", "PML", "FULL") else 2, extra=ext); best_p[fam] = p; r["fams"][fam] = report(P, p)
        r["fams"][fam]["KL_train"] = kl(P, logq(p, tr), tr)
    nf = dict(NPOE=dict(rho=1e-3)); p, _ = fit(P, ["tau", "a", "k", "d"], fixed=nf["NPOE"]); r["fams"]["NPOE"] = report(P, p)
    boot = {}
    for fam in ("PM", "PML"):
        rr = []
        for b in range(8):
            bi = np.random.default_rng(1000 + b).choice(tr, len(tr)); p, _ = fit(P, FAMS[fam], idx=bi); rr.append(float(p["rho"]))
        boot[fam] = dict(rho_boot=rr, lo=float(np.percentile(rr, 5)), hi=float(np.percentile(rr, 95)))
    r["bootstrap"] = boot
    if sp is not None and os.environ.get("FF_SHAPLEY", "1") == "1":
        gnames = list(GROUPS); base_kl = kl(P, logq(DEF, te), te); v = {}
        for k_ in range(len(gnames) + 1):
            for sub in itertools.combinations(gnames, k_):
                free = sum((GROUPS[g] for g in sub), []); p, _ = fit(P, free, extra=best_p["FULL"]); v["+".join(sub)] = base_kl - kl(P, logq(p, te), te)
        key = lambda s: "+".join(g for g in gnames if g in s); G = len(gnames)
        sh = {g: sum(math.factorial(len(s)) * math.factorial(G - len(s) - 1) / math.factorial(G) * (v[key(set(s) | {g})] - v[key(set(s))])
                     for k_ in range(G) for s in itertools.combinations([o for o in gnames if o != g], k_)) for g in gnames}
        r["shapley"] = dict(base_KL=base_kl, shares={g: sh[g] / base_kl for g in gnames}, explained_full=v["+".join(gnames)] / base_kl, subset_gain=v)
    r["sec"] = time.time() - t0; res[name] = r; json.dump(res, open(outp, "w"), indent=1)
    print(name, "KLvsB", round(r["KL_model_vs_bayes_test"], 4), {fm: (round(x["KL_test"], 5), round(x["params"]["rho"], 2)) for fm, x in r["fams"].items()},
          {k: (round(b["lo"], 2), round(b["hi"], 2)) for k, b in boot.items()}, {g: round(s, 2) for g, s in r.get("shapley", {}).get("shares", {}).items()},
          round(r["sec"]), flush=True)
