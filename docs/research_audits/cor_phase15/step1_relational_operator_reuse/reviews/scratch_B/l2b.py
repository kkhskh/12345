import sys, json, numpy as np
sys.argv=[sys.argv[0]]
exec(open('/tmp/revB17/checks.py').read().split("res = {}")[0])
conds = {"A3": "3:0.9:0.9:0.3333333333333333", "A2": "2:0.9:0.9:0.05"}; Ds = {k: XS.SlotDiag(c, n=600) for k, c in conds.items()}
def fwd2(p, X, pre=None, at2=None):
    """pre: delta dict for window [0,2) (relabel); at2: dict slot-> (7,d) array added at layer-2 input (after undo)."""
    x = N.embed(p, jnp.asarray(X)); L = len(p["layers"])
    for l, lay in enumerate(p["layers"]):
        if pre is not None and l == 0:
            for q, v in pre.items(): x = x.at[:, tok(q)].add(v)
        if l == 2:
            if pre is not None:
                for q, v in pre.items(): x = x.at[:, tok(q)].add(-v)
            if at2 is not None:
                for q, v in at2.items(): x = x.at[:, tok(q)].add(jnp.asarray(v))
        z = N.ln(x, lay["g1"], lay["b1"])
        qq = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        a = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qq, k) / jnp.sqrt(qq.shape[-1]), -1)
        x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
        x = x + jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
    zf = N.ln(x, p["gf"], p["bf"]); return np.asarray(zf[:, S * Q:S * Q + Q] @ p["WA"])
out={}
for name in ["A2_s704","A3_s804"]:
    A = json.load(open(f"{PKG}/logs/analysis/{name}.json")); D = Ds[name[:2]]
    p = GL.load(f"{PKG}/confirm_outputs/{name}/step{A['selected']}.npz"); E = np.asarray(p["Eq"])
    _, _, c = fwd(p, D.X, capture=True); h = c[2]; Qb, _ = np.linalg.qr(E.T); proj = lambda z: z - (z @ Qb) @ Qb.T
    cent = np.stack([proj(h[:, tok(q)]).mean(0) for q in range(6)]); r={}
    for i,j in A["pairs"][:3]:
        sw = {i: (E[j] - E[i])[None] + cent[j] - cent[i], j: (E[i] - E[j])[None] + cent[i] - cent[j]}
        rng = np.random.default_rng(2); rd={}
        for q in (i,j):
            R = rng.normal(size=sw[q].shape); rd[q] = R / np.linalg.norm(R, axis=-1, keepdims=True) * np.linalg.norm(sw[q], axis=-1, keepdims=True)
        # U<->U layer-2 identity swap control (j with another U k, if any)
        r[f"{i}->{j}"] = dict(fullid_rec=FG(D, j, lambda X: fwd2(p, X, at2=sw)), random_rec=FG(D, j, lambda X: fwd2(p, X, at2=rd)), fullid_donor=FG(D, i, lambda X: fwd2(p, X, at2=sw)))
    out[name]=r; print(name, json.dumps(r), flush=True)
json.dump(out, open("/tmp/revB17/l2b.json","w"), indent=1)
