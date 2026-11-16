"""XIV Stage III GATE mechanisms (presence / access / use / mediation) for one model.  Usage: g_mech.py NAME CKPT STREAM OUT.json
One question q per episode is edited (q uniform); scores are on q.  Edits: gate flip (x1 redrawn, on-support), relation-preserving shift
(x0+k, x1-k: g unchanged), decoy toggle (closed gates only: x1 moved to the other closed value; g unchanged, the decoy relation
d = [x0 == x1] may change).
(1) behaviour: KL to the exact rules; gate-flip slope G; |response| to the shift and the decoy toggle relative to the gate flip.
(2) presence: logistic probes for g at resid_pre1 / resid_pre2 / resid_post2 for the G, E (mean of the 4) and R tokens of q, trained on
    x0 in {0,1}, tested on held-out x0 = 2 (abstraction); same for the decoy relation d and the single value x0.
(3) use: residual patching from the gate-flipped donor at each (site, token set); flip transfer on Bayes-decisive cases.
(4) mediation at the best E site: 1-d interchange per E token along the g class-mean direction; controls: the decoy-relation direction,
    variance-matched random directions (10), and the relation-preserving null donor (decision change rate; ideal 0).
    Ideal-mechanism references (a model that stores g in one direction at E and decides by it): g-direction 1, decoy 0, random 0, null 0."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, gatelib as GL
from sklearn.linear_model import LogisticRegression
name, ck, stream, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
p = GL.load(ck); rng = np.random.default_rng(stream); n = 2000
d = GL.TX.sample_gate(n, rng); X = d["X"]; q = rng.integers(0, 6, n); I = np.arange(n)
def edit_q(X, fn):
    Xe = X.copy(); Xe[I, q] = fn(X[I, q][:, None])[:, 0]; return Xe
Xf = edit_q(X, lambda Z: GL.gate_flip(Z, rng)); Xs = edit_q(X, lambda Z: GL.gate_keep_shift(Z, rng))
def toggle(Z):
    Zt = Z.copy(); g = GL.gate(Z); x0, x1 = Z[:, :, 0], Z[:, :, 1]; other = (-x0 - (x1 + x0) % 3) % 3   # the other closed value
    # closed values are x1 with x0+x1 in {1,2}; the other closed value is x1' with x0+x1' = 3 - (x0+x1) mod 3
    alt = (3 - (x0 + x1) % 3 - x0) % 3; Zt[:, :, 1] = np.where(g, x1, alt); return Zt
Xd = edit_q(X, toggle)
gq = GL.gate(X)[I, q]; dq = (X[I, q, 0] == X[I, q, 1]); x0q = X[I, q, 0]
SITES = ("resid_pre1", "resid_pre2", "resid_post2")
def toks(kind): return {"G": np.stack([6 * q, 6 * q + 1], 1), "E": np.stack([6 * q + s for s in (2, 3, 4, 5)], 1), "R": (36 + q)[:, None]}[kind]
def run(Xin, patch=None):
    rec = {}
    def hk(nm, a):
        if nm in SITES:
            if patch is not None and nm == patch[0]: a = patch[1](a)
            rec[nm] = a
        return a
    out = jax.nn.softmax(GL.F3.forward(p, jnp.asarray(Xin), hk)["answer"], -1); return np.asarray(out)[I, q], {k: np.asarray(v) for k, v in rec.items()}
P0, R0 = run(X); Pf, Rf = run(Xf); Ps, _ = run(Xs); Pd, _ = run(Xd)
B0, Bf = GL.rule("bayes", X)[I, q], GL.rule("bayes", Xf)[I, q]
def cen(P): L = np.log(P + 1e-12); return L - L.mean(-1, keepdims=True)
dB = cen(Bf) - cen(B0); dm = cen(Pf) - cen(P0)
res = dict(name=name, ckpt=ck, n=n)
res["behaviour"] = dict(KL_bayes=GL.kl(B0[:, None], P0[:, None]), KL_gatefree=GL.kl(GL.rule("gatefree", X)[I, q][:, None], P0[:, None]),
                        G_slope=float((dm * dB).sum() / (dB * dB).sum()), flip_mean_abs=float(np.abs(dm).mean()),
                        shift_mean_abs=float(np.abs(cen(Ps) - cen(P0)).mean()), decoy_toggle_mean_abs=float(np.abs(cen(Pd) - cen(P0)).mean()),
                        shift_decision_change=float((Ps.argmax(-1) != P0.argmax(-1)).mean()), decoy_decision_change=float((Pd.argmax(-1) != P0.argmax(-1)).mean()))
# (2) presence
pres = {}
for s in SITES:
    for kind in ("G", "E", "R"):
        H = R0[s][I[:, None], toks(kind)].mean(1); tr, te = x0q != 2, x0q == 2; r = {}
        for lab, y in (("g", gq), ("decoy", dq)):
            c = LogisticRegression(max_iter=2000, C=1.0).fit(H[tr], y[tr]); r[lab] = float(c.score(H[te], y[te]))
        c = LogisticRegression(max_iter=2000).fit(H[: n // 2], x0q[: n // 2]); r["x0_value"] = float(c.score(H[n // 2:], x0q[n // 2:]))
        pres[f"{s}@{kind}"] = r
res["presence_heldout_x0"] = pres
# (3) use: residual patching
dec = GL.unique(B0) & GL.unique(Bf) & (B0.argmax(-1) != Bf.argmax(-1)); use = {}
def patch_tokens(s, T, src):
    def f(a):
        a = np.array(a); a[I[:, None], T] = src[s][I[:, None], T]; return jnp.asarray(a)
    return (s, f)
for s in SITES:
    for kind in ("G", "E", "R"):
        Pp, _ = run(X, patch_tokens(s, toks(kind), Rf)); use[f"{s}@{kind}"] = float((Pp.argmax(-1) == Bf.argmax(-1))[dec].mean())
res["use_flip_transfer"] = use; res["n_decisive"] = int(dec.sum())
# (4) mediation at the best E site (max E transfer)
best = max(SITES, key=lambda s: use[f"{s}@E"]); T = toks("E"); H0, Hf = R0[best], Rf[best]; half = I < n // 2
def classdir(lab):
    D = []
    for j in range(4):
        h = H0[I, T[:, j]]; u = h[half & lab].mean(0) - h[half & ~lab].mean(0); D.append(u / np.linalg.norm(u))
    return np.stack(D)
def interchange(D, src):
    def f(a):
        a = np.array(a)
        for j in range(4):
            u = D[j]; h, hs = a[I, T[:, j]], src[I, T[:, j]]; a[I, T[:, j]] = h + np.outer((hs - h) @ u, u)
        return jnp.asarray(a)
    return (best, f)
ev = ~half & dec
Ug = classdir(gq); Ud = classdir(dq)
def score(D, src=Hf): Pp, _ = run(X, interchange(D, src)); return float((Pp.argmax(-1) == Bf.argmax(-1))[ev].mean())
med = dict(site=best, g_direction=score(Ug), decoy_direction=score(Ud))
rr = np.random.default_rng(1); rnd = []; vr = []
def vm_dir(C, target):
    """random unit direction with u'Cu = target exactly: rotate a random direction towards the eigenvector on the other side."""
    w, V = np.linalg.eigh(C); a = rr.normal(size=C.shape[0]); a -= V[:, -1] * (a @ V[:, -1]); a -= V[:, 0] * (a @ V[:, 0]); a /= np.linalg.norm(a)
    va = a @ C @ a; bvec = V[:, -1] if va < target else V[:, 0]; vb = bvec @ C @ bvec
    t = np.clip((target - va) / (vb - va), 0, 1); return np.sqrt(1 - t) * a + np.sqrt(t) * bvec     # a, b orthogonal and C-orthogonal (b eigvec)
for t in range(10):
    D = []
    for j in range(4):
        h = H0[I, T[:, j]]; C = np.cov(h.T); tg = float(Ug[j] @ C @ Ug[j]); u = vm_dir(C, tg); D.append(u); vr.append(float(u @ C @ u / tg))
    rnd.append(score(np.stack(D)))
# value-conditional (conjunctive) directions: u_{j,v} = mean(h | g=1, x_s=v) - mean(h | g=0, x_s=v), applied with the token's own value
xs = np.stack([X[I, q, s] for s in (2, 3, 4, 5)], 1)
Uc = np.zeros((4, 3, H0.shape[-1]))
for j in range(4):
    h = H0[I, T[:, j]]
    for v in range(3):
        m1, m0 = half & gq & (xs[:, j] == v), half & ~gq & (xs[:, j] == v); u = h[m1].mean(0) - h[m0].mean(0); Uc[j, v] = u / np.linalg.norm(u)
def interchange_cond(src):
    def f(a):
        a = np.array(a)
        for j in range(4):
            U = Uc[j, xs[:, j]]; h, hs = a[I, T[:, j]], src[I, T[:, j]]; a[I, T[:, j]] = h + U * ((hs - h) * U).sum(1, keepdims=True)
        return jnp.asarray(a)
    return (best, f)
Pc, _ = run(X, interchange_cond(Hf)); med["value_conditional_direction"] = float((Pc.argmax(-1) == Bf.argmax(-1))[ev].mean())
med["cos_g_vs_conditional"] = [[float(Ug[j] @ Uc[j, v]) for v in range(3)] for j in range(4)]
med.update(random_vm_mean=float(np.mean(rnd)), random_vm_max=float(np.max(rnd)), random_var_ratio_mean=float(np.mean(vr)))
_, Rs = run(Xs); Pn, _ = run(X, interchange(Ug, Rs[best])); med["null_donor_decision_change"] = float((Pn.argmax(-1) != P0.argmax(-1))[~half].mean())
Pfull, _ = run(X, patch_tokens(best, T, Rf)); med["full_E_patch"] = float((Pfull.argmax(-1) == Bf.argmax(-1))[ev].mean())
med["ideal_reference"] = dict(g_direction=1.0, decoy_direction=0.0, random=0.0, null_donor=0.0)
res["mediation"] = med
json.dump(res, open(outp, "w"), indent=1, default=float)
print(json.dumps(dict(behaviour=res["behaviour"], mediation=med), default=float))
