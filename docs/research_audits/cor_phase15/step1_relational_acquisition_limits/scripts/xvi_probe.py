"""XVI cue-conditioned probes and causal use (mandate D1).  EXPLORATORY.  Usage: xvi_probe.py COND CKPT OUT.json [LABEL]
One question q per episode (3000 episodes, stream 97_900_000).  Sites: residual stream at the inputs of layers 1 and 2 and after layer 2,
averaged over the gate tokens (G), the evidence tokens (E) or taken at the readout of q (R).
  natural probe   logistic probe for g on natural inputs, trained on x0 in {0,1}, tested on held-out x0 = 2.
                  Baselines (no gate computation): AGREE = best accuracy predicting g from [pair agrees] alone; EVID = best accuracy
                  predicting g from the full evidence (pair + single), exact, from the task's joint law.  Excess = probe - EVID.
  decorrelated    the same probe trained and tested on GATE-RANDOMISED inputs (gate sources redrawn so that the visible g is independent
  probe           of the evidence regime): the evidence then carries no information about g, chance is exactly 0.5, and accuracy
                  above 0.5 requires g computed from the gate tokens.  (Inputs are off-distribution for a trained model; this is a
                  representational test, not a behavioural one.)
  causal use      flip transfer: on Bayes-decisive gate flips, the residual at the evidence tokens (layer-2 input) or the readout is
                  replaced by the donor's (gate-flipped) residual; the fraction of decisions moving to the flipped Bayes answer.  The
                  same-site transfer of a gate-randomised donor that keeps g is reported as a baseline.
Expected values: a gate-free (agreement-only) model: natural probe <= EVID, decorrelated probe 0.5, causal use = baseline; a model that
computes g: decorrelated probe near 1 at sites carrying g, causal use near 1."""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", ".."); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp, gatelib as GL, xvitask as XT, xvidiag as XD
from sklearn.linear_model import LogisticRegression
cond, ck, outp = sys.argv[1], sys.argv[2], sys.argv[3]; label = sys.argv[4] if len(sys.argv) > 4 else ck
a, b1, b0, nu = XD.parse(cond); K = 3; n = 3000
rng = np.random.default_rng(97_900_000); d = XT.sample(n, rng, a, b1, b0, nu); X = d["X"]; q = rng.integers(0, 6, n); I = np.arange(n)
def regate(Xin):
    Xr = Xin.copy(); m, Q = Xin.shape[:2]
    for s in range(a - 1): Xr[:, :, s] = rng.integers(0, K, (m, Q))
    part = Xr[:, :, :a - 1].sum(-1) % K; g = rng.random((m, Q)) < 0.5; op = (-part) % K; Xr[:, :, a - 1] = np.where(g, op, (op + rng.integers(1, K, (m, Q))) % K); return Xr
Xd = regate(X); Xf = X.copy(); Xf[I, q] = XT.gate_flip(X[I, q][:, None], a, rng)[:, 0]
B0, Bf = XT.bayes(X, a, b1, b0)[I, q], XT.bayes(Xf, a, b1, b0)[I, q]; dec = GL.unique(B0) & GL.unique(Bf) & (B0.argmax(-1) != Bf.argmax(-1))
gq, gd = XT.gate(X, a)[I, q], XT.gate(Xd, a)[I, q]; x0, x0d = X[I, q, 0], Xd[I, q, 0]
# exact baselines from the joint law (one question)
px, PB = XT.joint(a, b1, b0, nu); XA = XT.xall(6); gv = (XA[:, :a].sum(1) % K) == 0
def best(keyf):
    j = {}
    for x, g, p_ in zip(XA, gv, px): k = keyf(x); j.setdefault(k, [0.0, 0.0]); j[k][int(g)] += p_
    return float(sum(max(v) for v in j.values()))
AGREE = best(lambda x: x[3] == x[4]); EVID = best(lambda x: (x[3], x[4], x[5]))
SITES = ("resid_pre1", "resid_pre2", "resid_post2")
def toks(kind): return {"G": np.stack([6 * q + s for s in range(a)], 1), "E": np.stack([6 * q + s for s in (3, 4, 5)], 1), "R": (36 + q)[:, None]}[kind]
def run(p, Xin, patch=None):
    rec = {}
    def hk(nm, v):
        if nm in SITES:
            if patch is not None and nm == patch[0]: v = patch[1](v)
            rec[nm] = v
        return v
    o = jax.nn.softmax(GL.F3.forward(p, jnp.asarray(Xin), hk)["answer"], -1); return np.asarray(o)[I, q], {k: np.asarray(v) for k, v in rec.items()}
p = GL.load(ck); P0, R0 = run(p, X); Pf, Rf = run(p, Xf); _, Rd = run(p, Xd)
res = dict(cond=cond, ckpt=label, baselines=dict(AGREE=AGREE, EVID=EVID, decorrelated_chance=0.5), natural={}, decorrelated={}, use={}, use_baseline={})
res["follow"] = float(((Pf.argmax(-1) == Bf.argmax(-1)) & (Pf.argmax(-1) != P0.argmax(-1)))[dec].mean())
probe = lambda H, y, xx: float(LogisticRegression(max_iter=3000, C=1.0).fit(H[xx != 2], y[xx != 2]).score(H[xx == 2], y[xx == 2]))
for s in SITES:
    for kind in "GER":
        res["natural"][f"{s}@{kind}"] = probe(R0[s][I[:, None], toks(kind)].mean(1), gq, x0)
        res["decorrelated"][f"{s}@{kind}"] = probe(Rd[s][I[:, None], toks(kind)].mean(1), gd, x0d)
for s, kind in (("resid_pre2", "E"), ("resid_pre2", "R"), ("resid_post2", "R")):
    T = toks(kind)
    def f(v, s=s, T=T, src=Rf):
        v = np.array(v); v[I[:, None], T] = src[s][I[:, None], T]; return jnp.asarray(v)
    Pp, _ = run(p, X, (s, f)); res["use"][f"{s}@{kind}"] = float((Pp.argmax(-1) == Bf.argmax(-1))[dec].mean())
res["use_baseline_no_patch"] = float((P0.argmax(-1) == Bf.argmax(-1))[dec].mean())
res["excess_natural_over_EVID"] = {k: v - EVID for k, v in res["natural"].items()}
json.dump(res, open(outp, "w"), indent=1)
print(label, "follow %.2f" % res["follow"], "EVID %.3f AGREE %.3f" % (EVID, AGREE), {k: round(v, 2) for k, v in res["natural"].items() if "pre1" not in k}, "decor", {k: round(v, 2) for k, v in res["decorrelated"].items() if "pre1" not in k}, "use", {k: round(v, 2) for k, v in res["use"].items()}, "base %.2f" % res["use_baseline_no_patch"], flush=True)
