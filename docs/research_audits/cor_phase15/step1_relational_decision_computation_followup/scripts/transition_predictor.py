"""Workstream A (EXPLORATORY, not frozen): does an INTERNAL quantity that is not computed from the model's outputs predict the behavioural
A2 -> B transition during full_c1 training?
Internal predictor (attention only, no logits): ALIGN(t) = max over layer-2 heads of corr(per-group readout-token attention mass at q, Bayes
responsibility r_g), on natural episodes (stream 74_800_000), restricted to heads with mass on q-tokens >= 0.5 at that checkpoint (else 0).
Behavioural metric (decision metric): on the on-support discriminating families D1/D2 (algorithm_discrimination.py), gap(t) = dist_to_A2 - dist_to_B
(> 0: closer to B) and KL(Bayes || model) on natural episodes.
Reported per model: the curves, t50 of ALIGN (first checkpoint reaching 50% of its final value) and t50 of the behavioural gap, and the
lead = t50_gap - t50_align (positive: the internal alignment rises first).  -> logs/transition_predictor<TAG>.json
Usage: TP_MODELS=spec,spec python3 scripts/transition_predictor.py  (eval set for families: TP_EVAL, default dev)"""
import json, os, sys, re, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
src = open(os.path.join(HERE, "algorithm_discrimination.py")).read(); src = src[:src.index("out = dict(n=")]
sys.argv = ["x", os.environ.get("TP_EVAL", "dev")]; g = {"__file__": os.path.join(HERE, "algorithm_discrimination.py"), "__name__": "alg"}; exec(compile(src, "alg[def]", "exec"), g)
fams, pred = g["fams"], g["pred"]
env = GL.Env(3, 1); rng = np.random.default_rng(74_800_000); N_ = 1500
d = NV.Bm.sample_fast(NV.ENV, N_, rng); X = d["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(N_, 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1, keepdims=True)
rB = W * (S - e) * Lg.sum(-1); rB = rB / rB.sum(1, keepdims=True); PB = GL.posterior_closed(env, X, q)
tq = q[:, None] * 6 + np.arange(6)[None]
out = {}
for sp in os.environ["TP_MODELS"].split(","):
    parent, spec, seed, suf = sp.split(":"); D = F.model_dir(f"{parent}_{spec}_s4000_seed{seed}{suf}")
    steps = sorted(int(re.findall(r"step(\d+)", os.path.basename(f))[0]) for f in glob.glob(os.path.join(D, "step*.npz"))); rows = []
    for k in steps:
        M = F.load_model(parent, spec, seed=int(seed), suffix=suf, step=k); r = {"step": k}
        rec = {}; L = np.asarray(F.forward(M, jnp.asarray(X), rec=rec)); A2 = np.asarray(rec["pattern2"])
        P = np.exp(L[ii, q] - L[ii, q].max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True); r["KL"] = float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean())
        al = []
        for h in range(4):
            row = A2[ii, h, 36 + q]; mq = np.take_along_axis(row, tq, 1); mass = mq.sum(1).mean()
            pg = np.stack([mq[:, 2 * gg:2 * gg + 2].sum(1) for gg in range(3)], 1); pg = pg / np.maximum(pg.sum(1, keepdims=True), 1e-12)
            al.append(float(np.corrcoef(pg.ravel(), rB.ravel())[0, 1]) if mass >= 0.5 else 0.0)
        r["ALIGN"] = max(al); r["align_per_head"] = al
        fj = jax.jit(lambda X_: F.forward(M, X_)); gaps = []
        for f in ("D1", "D2"):
            Xb, Xa, qq = fams[f]; i = np.arange(len(qq)); D_ = F.centred(F.batched(fj, Xa)[i, qq]) - F.centred(F.batched(fj, Xb)[i, qq])
            gaps.append(float(np.abs(D_ - pred[f]["A2"]).mean() - np.abs(D_ - pred[f]["B"]).mean()))
        r["gap_A2_minus_B"] = float(np.mean(gaps)); rows.append(r)
        print(sp[:22], k, "ALIGN", round(r["ALIGN"], 3), "gap", round(r["gap_A2_minus_B"], 3), "KL", round(r["KL"], 3), flush=True)
    def t50(v):
        v = np.array(v); lo, hi = v[0], v[-1]
        if hi - lo <= 1e-6: return None
        return int(steps[int(np.argmax(v >= lo + 0.5 * (hi - lo)))])
    ta, tg = t50([r["ALIGN"] for r in rows]), t50([r["gap_A2_minus_B"] for r in rows])
    out[sp] = dict(rows=rows, t50_align=ta, t50_gap=tg, lead=(None if ta is None or tg is None else tg - ta))
    print(sp, "t50 align", ta, "t50 gap", tg, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "transition_predictor" + os.environ.get("TP_TAG", "") + ".json"), "w"), indent=1)
