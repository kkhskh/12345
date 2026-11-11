"""Observable boundary responses for one or more models.  Usage: boundary_run.py PHASE MODEL [MODEL ...]
PHASE dev  : stream 15_000_000 (development models)      PHASE conf : stream 15_100_000 (Investigation VI cohort)
Generator side (written once to logs/PHASE/generator.npz): 6000 natural episodes, one scored decision q each (fixed RNG);
families F45, F34, F23 (one non-colluder group's leave-q count raised by one through a single report; same world, X_q fixed) and
F0 (the same token as the F45 edit changed without changing agreement; V and Bayes unchanged).  Orientation: the edited group's
own candidate c_g at q (defined when its two q-sources agree; otherwise the pair is excluded from oriented statistics).
Per model and pair: oriented log-probability-ratio margin m = log p(c_g) − mean log p(other); model response Δm_M, exact Δm_B,
excess E = Δm_M − Δm_B; probability response Δp(c_g) model and Bayes; rank changes.  Quality on the natural set: accuracy, KL to
Bayes, ECE (10 bins), KL-optimal temperature, mean |centred logit| (margin scale).  Writes logs/PHASE/model_<name>.{npz,json}."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import boundary as B
R, E, G = B.R, B.E, B.G
PH, names = sys.argv[1], sys.argv[2:]
EVAL = {"dev": 15_000_000, "conf": 15_100_000}[PH]; NE = int(os.environ.get("NE", 6000))
OUT = os.path.join(HERE, "..", "logs", PH); os.makedirs(OUT, exist_ok=True); t0 = time.time()
D = E.sample(NE, EVAL); X, Y, cfg = D["X"], D["Y"], D["cfg"]; q = np.random.default_rng(EVAL + 1).integers(0, 6, NE); idx = np.arange(NE)
fam, meta = {}, {}
for name, (frm, to) in {"F45": (4, 5), "F34": (3, 4), "F23": (2, 3)}.items():
    Xp = X.copy(); gs = np.full(NE, -1); ks = gs.copy(); ok = np.zeros(NE, bool)
    for qq in range(6):
        s = q == qq; a, g_, k_, o = B.count_edit(X[s], qq, cfg[s], np.random.default_rng(EVAL + 10 + qq + 100 * frm), frm, to); Xp[s], gs[s], ks[s], ok[s] = a, g_, k_, o
    fam[name] = Xp; vote = np.array([B.group_vote(X[i:i + 1], q[i], max(gs[i], 0))[0] for i in idx]); meta[name] = dict(g=gs, k=ks, ok=ok, vote=vote, orient=ok & (vote >= 0))
    if name == "F45": fam["F0"] = B.token_control(X, gs, ks); meta["F0"] = dict(g=gs, k=ks, ok=ok, vote=vote, orient=ok & (vote >= 0))
c0B = B.bayes_c(X, q); cB = {f: B.bayes_c(Xf, q) for f, Xf in fam.items()}
# regimes (for F45-type orientation): relation of c_g to other groups' votes at q, number of count-5 groups before the edit
n0 = E.loo(X)[idx, q]; Hb = n0 == 5; Vq = np.stack([np.array([B.group_vote(X[i:i + 1], q[i], g)[0] for i in idx]) for g in range(3)], 1)
regime = {}
for f in fam:
    rg = np.full(NE, "", object); gs = meta[f]["g"]; vote = meta[f]["vote"]
    for i in np.where(meta[f]["orient"])[0]:
        oth = [h for h in range(3) if h != gs[i]]; col = [h for h in oth if Hb[i, h]]; thr = [h for h in oth if not Hb[i, h]]
        rg[i] = "R1_with_colluder_candidate" if any(Vq[i, h] == vote[i] for h in col) else ("R2_with_third_group" if any(Vq[i, h] == vote[i] for h in thr) else "R3_with_no_group")
    regime[f] = rg
gfile = os.path.join(OUT, "generator.npz")
np.savez_compressed(gfile, X=X, Y=Y, cfg=cfg, q=q, nH_before=Hb.sum(1), **{f"X_{f}": v for f, v in fam.items()},
                    **{f"{k}_{f}": v for f, m in meta.items() for k, v in m.items()}, **{f"regime_{f}": v.astype(str) for f, v in regime.items()},
                    cB_base=c0B, **{f"cB_{f}": v for f, v in cB.items()})
print(f"[{time.time() - t0:.0f}s] generator: oriented fractions " + json.dumps({f: float(m['orient'].mean()) for f, m in meta.items()}), flush=True)
post = E.bayes_from(X, E.loo(X))
for m in names:
    M = B.Model(m); res = {}
    L0 = M.acts(X)[1]; c0 = E.clogit(L0[idx, q]); res["c_base"] = c0
    for f, Xf in fam.items(): res[f"c_{f}"] = E.clogit(M.acts(Xf)[1][idx, q])
    summ = {}
    for f in fam:
        o = meta[f]["orient"]; v = np.maximum(meta[f]["vote"], 0)
        dmM = B.oriented_margin(res[f"c_{f}"], v) - B.oriented_margin(c0, v); dmB = B.oriented_margin(cB[f], v) - B.oriented_margin(c0B, v)
        sm = lambda c: np.exp(c) / np.exp(c).sum(-1, keepdims=True)
        dpM = sm(res[f"c_{f}"])[idx, v] - sm(c0)[idx, v]; dpB = sm(cB[f])[idx, v] - sm(c0B)[idx, v]
        exc = (dmM - dmB)[o]; ep = (dpM - dpB)[o]
        summ[f] = dict(n=int(o.sum()), B_logratio=float(exc.mean()), B_prob=float(ep.mean()), mean_dmM=float(dmM[o].mean()), mean_dmB=float(dmB[o].mean()),
                       slope_dmM_on_dmB=float(np.polyfit(dmB[o], dmM[o], 1)[0]) if dmB[o].std() > 0 else float("nan"),
                       rank_change_model=float((res[f"c_{f}"].argmax(-1) != c0.argmax(-1))[o].mean()), rank_change_bayes=float((cB[f].argmax(-1) != c0B.argmax(-1))[o].mean()),
                       by_regime={r: dict(n=int(((regime[f] == r) & o).sum()), B_logratio=float((dmM - dmB)[(regime[f] == r) & o].mean()), mean_dmB=float(dmB[(regime[f] == r) & o].mean()))
                                  for r in ("R1_with_colluder_candidate", "R2_with_third_group", "R3_with_no_group") if ((regime[f] == r) & o).sum() >= 10})
        res[f"E_{f}"] = np.where(o, dmM - dmB, np.nan); res[f"Ep_{f}"] = np.where(o, dpM - dpB, np.nan)
    Lall = E.clogit(L0.reshape(-1, 3)); PB = post.reshape(-1, 3); Pm = E.softmax(Lall)
    kl = lambda T: float((PB * (np.log(PB + 1e-12) - np.log(E.softmax(Lall / T) + 1e-12))).sum(-1).mean())
    Ts = np.exp(np.linspace(np.log(0.3), np.log(3), 241)); Tst = float(Ts[np.argmin([kl(T) for T in Ts])])
    conf = Pm.max(-1); corr = Pm.argmax(-1) == Y.reshape(-1); bins = np.minimum((conf * 10).astype(int), 9)
    qual = dict(acc=float(corr.mean()), KL_bayes=kl(1.0), T_star=Tst, KL_calibrated=kl(Tst), margin_scale=float(np.abs(Lall).mean()),
                ECE=float(sum(abs(corr[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())),
                NLL=float(-np.log(Pm[np.arange(len(Pm)), Y.reshape(-1)] + 1e-12).mean()))
    np.savez_compressed(os.path.join(OUT, f"model_{m.replace('@', '_')}.npz"), **res)
    json.dump(dict(model=m, kind=B.kind_of(m), quality=qual, families=summ), open(os.path.join(OUT, f"model_{m.replace('@', '_')}.json"), "w"), indent=1)
    print(f"[{time.time() - t0:.0f}s] {m} ({B.kind_of(m)}): acc {qual['acc']:.3f} KL {qual['KL_bayes']:.3f}  B45 {summ['F45']['B_logratio']:+.3f}  B34 {summ['F34']['B_logratio']:+.3f}  B23 {summ['F23']['B_logratio']:+.3f}  B0 {summ['F0']['B_logratio']:+.3f}", flush=True)
