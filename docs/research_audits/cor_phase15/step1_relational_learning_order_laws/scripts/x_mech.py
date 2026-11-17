"""XV Workstream D (exploratory, selection rule declared in FREEZE XV): around the gate transition of one model.
Usage: x_mech.py COND RUNDIR T_G OUT.json   Checkpoints: t_G-200, t_G-50, t_G, t_G+100, 4000 (nearest existing snapshots).
At each: (i) presence: logistic probe for g (trained on x0 in {0,1}, tested on held-out x0 = 2; chance 0.5) on the residual at the inputs
of layers 1 and 2 and after layer 2, at the gate tokens (mean), the evidence tokens 3-5 (mean) and the readout of the edited question;
(ii) use: flip transfer of a full residual patch from the gate-flipped donor at the gate tokens / evidence tokens / readout (layer-1 and
layer-2 inputs), on Bayes-decisive flips; (iii) behaviour: G and accuracy."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp, gatelib as GL, xvtask as XT
from sklearn.linear_model import LogisticRegression
cond, rd, tG, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; a, b, nu = int(cond.split(":")[0]), float(cond.split(":")[1]), float(cond.split(":")[2])
rng = np.random.default_rng(97_900_000); n = 1500; d = XT.sample(n, rng, a, b, nu); X = d["X"]; q = rng.integers(0, 6, n); I = np.arange(n)
Xf = X.copy(); Xf[I, q] = XT.gate_flip(X[I, q][:, None], a, rng)[:, 0]
B0, Bf = XT.bayes(X, a, b)[I, q], XT.bayes(Xf, a, b)[I, q]; dec = GL.unique(B0) & GL.unique(Bf) & (B0.argmax(-1) != Bf.argmax(-1)); gq = XT.gate(X, a)[I, q]; x0 = X[I, q, 0]
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
avail = GL.snap_steps(rd); pick = sorted({min(avail, key=lambda s: abs(s - t)) for t in (tG - 200, tG - 50, tG, tG + 100, 4000)}); res = dict(cond=cond, run=rd, t_G=tG, points=[])
for k in pick:
    p = GL.load(os.path.join(rd, f"step{k}.npz")); P0, R0 = run(p, X); Pf, Rf = run(p, Xf); pt = dict(step=k, presence={}, use={})
    pt["follow"] = float(((Pf.argmax(-1) == Bf.argmax(-1)) & (Pf.argmax(-1) != P0.argmax(-1)))[dec].mean())
    for s in SITES:
        for kind in "GER":
            H = R0[s][I[:, None], toks(kind)].mean(1); tr, te = x0 != 2, x0 == 2
            pt["presence"][f"{s}@{kind}"] = float(LogisticRegression(max_iter=2000).fit(H[tr], gq[tr]).score(H[te], gq[te]))
            if s != "resid_post2":
                T = toks(kind)
                def f(v, s=s, T=T):
                    v = np.array(v); v[I[:, None], T] = Rf[s][I[:, None], T]; return jnp.asarray(v)
                Pp, _ = run(p, X, (s, f)); pt["use"][f"{s}@{kind}"] = float((Pp.argmax(-1) == Bf.argmax(-1))[dec].mean())
    res["points"].append(pt); print(k, pt["follow"], {kk: round(v, 2) for kk, v in pt["presence"].items() if "E" in kk or "R" in kk}, {kk: round(v, 2) for kk, v in pt["use"].items()}, flush=True)
json.dump(res, open(outp, "w"), indent=1)
