"""Per-model forward-pass data for Investigation V.  Usage: cri_run.py PHASE MODEL [MODEL ...]
PHASE dev  : development models (Investigation IV P2-P4, T2, T10, T11 and P0, P1, T0, T1); streams 13_000_000 (eval),
             13_010_000 (encoder fit)
PHASE conf : independent cohort cT60..cT67, cP60..cP67; streams 13_100_000 (eval), 13_110_000 (encoder fit)
One scored decision q per episode (fixed RNG).  Generator-side data (identical for every model of a phase) are written once
to logs/PHASE/generator.npz; per-model raw arrays to logs/PHASE/model_<name>.npz.  Everything is a function of the generator
and the model; no quantity is selected using model outputs.

Families at the scored decision (all with X_q fixed; support labels recorded per pair):
  N    Bayesian-null minimal edit (two non-colluder counts shifted together)          exact Bayes unchanged, same world
  N2   Bayesian-null resample: X_{-q} re-realised with another count vector of the same pi-class (fresh report values)
  R    Bayesian-relevant minimal edit (one non-colluder agreement toggled), same world, Bayes changes
  R2   Bayesian-relevant resample: X_{-q} re-realised with a random same-world count vector with different pi
  C1   relabel (counts preserved)   C2 within-pair source swap   C3 X_{-q} resampled with the same agreement pattern
  Cm   matched-content control for N: the same report tokens touched, agreement status (hence V, Bayes) unchanged
  Nabs, Rabs   abstract V interventions at the model's Investigation IV site (P: readout 36+q; T: q's six source tokens)
               with covariate-corrected encoders, using exactly the count vectors of the N and R pairs
  path  N and R responses with the layer-2 input patched (counterfactual values) only at position sets
        Rq = {36+q}, Qs = {6q..6q+5}, Oth = all other positions; and with each set blocked (all others patched)
  surf  input-level V-surface: for the first NS episodes, every same-world count vector (36) realised KR times; mean
        centred logits per (episode, vector)
Quality on the natural set: accuracy, KL(Bayes || model), ECE, fitted temperature, mean |centred log-odds error|."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import cri as R
E = R.E
PH = sys.argv[1]; names = sys.argv[2:]
EVAL, FIT = {"dev": (13_000_000, 13_010_000), "conf": (13_100_000, 13_110_000)}[PH]
NE, NF, NS, KR = int(os.environ.get("NE", 2000)), 4000, int(os.environ.get("NS", 300)), 4
OUT = os.path.join(HERE, "..", "logs", PH); os.makedirs(OUT, exist_ok=True)
t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

# ------------------------------------------------------------------ generator side (deterministic)
gfile = os.path.join(OUT, "generator.npz")
D = E.sample(NE, EVAL); X, Y, cfg = D["X"], D["Y"], D["cfg"]
rng = np.random.default_rng(EVAL + 1); qsel = rng.integers(0, R.Q, NE); idx = np.arange(NE)
Xf = E.sample(NF, FIT)["X"]
fam = {k: X.copy() for k in ("N", "N2", "R", "R2", "C1", "C2", "C3", "Cm")}; ok = {k: np.zeros(NE, bool) for k in fam}
for q in range(R.Q):
    s = qsel == q; Xs = X[s]; r = np.random.default_rng(EVAL + 10 + q)
    a, o = R.family_N(Xs, q, r); fam["N"][s], ok["N"][s] = a, o
    a, o = R.family_R(Xs, q, cfg[s], r); fam["R"][s], ok["R"][s] = a, o
    for c in ("C1", "C2", "C3"): fam[c][s] = R.family_C(Xs, q, r, c); ok[c][s] = True
    fam["Cm"][s] = R.matched_content_control(Xs, fam["N"][s], r); ok["Cm"][s] = ok["N"][s]
    n = E.loo(Xs)[:, q]; P = E.pi_of_n(n); h = R.true_h(cfg[s])
    v2 = n.copy(); o2 = np.zeros(len(Xs), bool); v3 = n.copy(); o3 = np.zeros(len(Xs), bool)
    for i in range(len(Xs)):
        sv = R.surface_vs(h[i]); Ps = E.pi_of_n(sv); same = np.abs(Ps - P[i]).max(-1) < 1e-12
        cand = sv[same & (np.abs(sv - n[i]).sum(1) > 0)]
        if len(cand): v2[i] = cand[r.integers(len(cand))]; o2[i] = True
        cand = sv[~same]; v3[i] = cand[r.integers(len(cand))]; o3[i] = True
    fam["N2"][s] = R.realize(Xs, q, v2, r); ok["N2"][s] = o2
    fam["R2"][s] = R.realize(Xs, q, v3, r); ok["R2"][s] = o3
post0 = E.bayes_from(X, E.loo(X)); cB = {"base": R.bayes_logodds(post0[idx, qsel])}
sup = {}
for k, Xp in fam.items():
    pk = E.bayes_from(Xp, E.loo(Xp)); cB[k] = R.bayes_logodds(pk[idx, qsel])
    lab = {}
    for q in range(R.Q):
        s = qsel == q; L = R.support_labels(X[s], Xp[s], q, cfg[s])
        for kk, v in L.items(): lab.setdefault(kk, np.zeros(NE, v.dtype))[s] = v
    sup[k] = lab
# surface
Si = np.arange(NS); surfV = np.stack([R.surface_vs(R.true_h(cfg[i:i + 1])[0]) for i in Si])          # NS,36,3
rs = np.random.default_rng(EVAL + 99); Xsurf = np.repeat(X[Si], 36 * KR, 0); qs = np.repeat(qsel[Si], 36 * KR)
Vflat = np.repeat(surfV.reshape(-1, 3), KR, 0)
for q in range(R.Q):
    s = qs == q
    if s.any(): Xsurf[s] = R.realize(Xsurf[s], q, Vflat[s], rs)
Psurf = E.bayes_from(Xsurf, E.loo(Xsurf)); cBsurf = R.bayes_logodds(Psurf[np.arange(len(Xsurf)), qs]).reshape(NS, 36, KR, 3).mean(2)
assert (E.loo(Xsurf)[np.arange(len(Xsurf)), qs] == Vflat).all() and E.same_world(Xsurf, np.repeat(cfg[Si], 36 * KR)).all()
np.savez_compressed(gfile, X=X, Y=Y, cfg=cfg, qsel=qsel, Xfit=Xf, surfV=surfV, cBsurf=cBsurf,
                    **{f"X_{k}": v for k, v in fam.items()}, **{f"ok_{k}": v for k, v in ok.items()},
                    **{f"cB_{k}": v for k, v in cB.items()}, **{f"sup_{k}_{kk}": v for k, d in sup.items() for kk, v in d.items()})
log("generator data written; ok fractions " + json.dumps({k: float(v.mean()) for k, v in ok.items()}))

# ------------------------------------------------------------------ per model
def at_q(L): return L[idx, qsel]
for m in names:
    M = R.Model(m); k = M.kind; d = M.d; res = {}
    A0, L0, _, _ = M.acts(X); c0 = E.clogit(at_q(L0)); res["c_base"] = c0
    p0 = E.softmax(at_q(L0)); pB = post0[idx, qsel]
    acts = {}
    for f, Xp in fam.items():
        Ap, Lp, _, _ = M.acts(Xp); res[f"c_{f}"] = E.clogit(at_q(Lp)); acts[f] = Ap
    # quality
    Lall = E.clogit(L0.reshape(-1, 3)); PB = post0.reshape(-1, 3); cBall = R.bayes_logodds(PB)
    kl = lambda T: float((PB * (np.log(PB + 1e-12) - np.log(E.softmax(Lall / T) + 1e-12))).sum(-1).mean())
    Ts = np.exp(np.linspace(np.log(0.3), np.log(3), 121)); Tstar = float(Ts[np.argmin([kl(T) for T in Ts])])
    Pm = E.softmax(Lall); conf = Pm.max(-1); corr = (Pm.argmax(-1) == Y.reshape(-1)); bins = np.minimum((conf * 10).astype(int), 9)
    ece = float(sum(abs(corr[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    qual = dict(acc=float(corr.mean()), acc_bayes=float((PB.argmax(-1) == Y.reshape(-1)).mean()), KL_bayes=kl(1.0), T_star=Tstar,
                KL_bayes_calibrated=kl(Tstar), ECE=ece, mean_abs_logodds_err=float(np.abs(Lall - cBall).mean()),
                mean_abs_logodds_err_calibrated=float(np.abs(Lall / Tstar - cBall).mean()))
    # abstract interventions (covariate-corrected encoder at the Inv IV site)
    Af = M.acts(Xf)[0]; ek = "P_full" if k == "P" else "T_own"; enc = R.fit_encoder_cov(ek, Xf, Af)
    for f in ("N", "R"):
        cab = np.zeros((NE, 3))
        for q in range(R.Q):
            s = qsel == q; pos = E.site_positions(k, q)
            dl = E.encoder_delta(enc, ek, X[s], E.loo(X[s]), E.loo(fam[f][s]), d, pos)
            cab[s] = E.clogit(M.run(X[s], dl)[:, q])
        res[f"c_{f}abs"] = cab
    # path patching
    for f in ("N", "R"):
        for setname in ("Rq", "Qs", "Oth"):
            pa, bl = np.zeros((NE, 3)), np.zeros((NE, 3))
            for q in range(R.Q):
                s = qsel == q; mask = np.zeros(E.T, bool)
                if setname == "Rq": mask[36 + q] = True
                elif setname == "Qs": mask[6 * q:6 * q + 6] = True
                else: mask[:] = True; mask[36 + q] = False; mask[6 * q:6 * q + 6] = False
                full = acts[f][s] - A0[s]
                pa[s] = E.clogit(M.run(X[s], full * mask[None, :, None])[:, q])
                bl[s] = E.clogit(M.run(X[s], full * (~mask)[None, :, None])[:, q])
            res[f"c_{f}_patch_{setname}"] = pa; res[f"c_{f}_block_{setname}"] = bl
    # V-surface (input level)
    Ls = M.acts(Xsurf)[1][np.arange(len(Xsurf)), qs]; S4 = E.clogit(Ls).reshape(NS, 36, KR, 3); surf = S4.mean(2); res["surf"] = surf
    res["surf_half_a"], res["surf_half_b"] = S4[:, :, :KR // 2].mean(2), S4[:, :, KR // 2:].mean(2)   # split halves (noise ceiling)
    Jq = np.concatenate([R.J_at(X[i:i + 1], qsel[i]) for i in Si]); Jrep = np.repeat(Jq, 36, 0); Vs = surfV.reshape(-1, 3)
    Xqr = np.repeat(X[Si, qsel[Si]], 36, 0); tgt = surf.reshape(-1, 3); grp = np.repeat(Si, 36)
    fits = {}
    ha, hb = res["surf_half_a"].reshape(-1, 3), res["surf_half_b"].reshape(-1, 3)
    rel = R.r2(ha, hb, grp)["r2_within"]; fits["split_half_r2_within"] = rel     # R² of one half predicting the other
    for fk in ("mix", "vote"):
        th, pred = R.fit_family(fk, tgt, Jrep, Xqr, Vs); fits[fk] = dict(theta=th.tolist(), **R.r2(pred, tgt, grp))
        res[f"surf_pred_{fk}"] = pred.reshape(NS, 36, 3)
    ex = R.mix_exact(Jrep, Vs); beta = float((ex * tgt).sum() / (ex * ex).sum())
    fits["exact_bayes_scaled"] = dict(scale=beta, **R.r2(beta * ex, tgt, grp)); res["surf_pred_exact"] = (beta * ex).reshape(NS, 36, 3)
    np.savez_compressed(os.path.join(OUT, f"model_{m}.npz"), **res)
    json.dump(dict(model=m, kind=k, quality=qual, fits=fits), open(os.path.join(OUT, f"model_{m}.json"), "w"), indent=1)
    log(f"{m} ({k}): acc {qual['acc']:.3f} KL {qual['KL_bayes']:.3f} T* {Tstar:.2f}; R2_within mix {fits['mix']['r2_within']:.3f} vote {fits['vote']['r2_within']:.3f} exact {fits['exact_bayes_scaled']['r2_within']:.3f} split-half {rel:.3f}")
