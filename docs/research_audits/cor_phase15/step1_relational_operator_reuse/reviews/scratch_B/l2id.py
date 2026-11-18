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
res = {}
for name in ["A3_s804", "A3_s803", "A2_s705", "A2_s704"]:
    A = json.load(open(f"{PKG}/logs/analysis/{name}.json")); D = Ds[name[:2]]
    p = GL.load(f"{PKG}/confirm_outputs/{name}/step{A['selected']}.npz"); E = np.asarray(p["Eq"]); i, j = A["pairs"][0]
    _, _, c = fwd(p, D.X, capture=True); h = c[2]                      # layer-2 input, N x T x d
    Qb, _ = np.linalg.qr(E.T); proj = lambda z: z - (z @ Qb) @ Qb.T
    cent = np.stack([proj(h[:, tok(q)]).mean(0) for q in range(6)])     # 6 x 7 x d (non-Eq slot centroids per token type)
    dl = {i: jnp.asarray(E[j] - E[i]), j: jnp.asarray(E[i] - E[j])}
    r = dict(pair=[i, j])
    r["relabel02_FG_j"] = FG(D, j, lambda X: fwd2(p, X, pre=dl))
    # after the [0,2) relabel, move recipient's derived identity at layer 2 back from i to j (and donor's from j to i)
    r["relabel02_plus_L2_derived_id_restored_FG_j"] = FG(D, j, lambda X: fwd2(p, X, pre=dl, at2={j: cent[j] - cent[i], i: cent[i] - cent[j]}))
    # baseline, layer-2-only full identity swap (raw code + derived-identity centroid shift) for donor i and recipient j
    sw = {i: (E[j] - E[i])[None] + cent[j] - cent[i], j: (E[i] - E[j])[None] + cent[i] - cent[j]}
    r["L2_only_fullid_swap_FG_donor_i"] = FG(D, i, lambda X: fwd2(p, X, at2=sw)); r["L2_only_fullid_swap_FG_recipient_j"] = FG(D, j, lambda X: fwd2(p, X, at2=sw))
    rawonly = {i: (E[j] - E[i])[None] + 0 * cent[0], j: (E[i] - E[j])[None] + 0 * cent[0]}
    r["L2_only_rawcode_swap_FG_donor_i(=pkg [2,3) w/o readout)"] = FG(D, i, lambda X: fwd2(p, X, at2=rawonly))
    # control: shift of equal norm along a random direction
    rng = np.random.default_rng(1); rd = {}
    for q in (i, j):
        dv = sw[q]; R = rng.normal(size=dv.shape); R = R / np.linalg.norm(R, axis=-1, keepdims=True) * np.linalg.norm(dv, axis=-1, keepdims=True); rd[q] = R
    r["L2_only_random_equal_norm_FG_donor_i"] = FG(D, i, lambda X: fwd2(p, X, at2=rd))
    r["derived_id_norm_over_resid_norm_L2"] = float(np.linalg.norm(cent[i] - cent[j], axis=-1).mean() / np.linalg.norm(h[:, tok(i)], axis=-1).mean())
    res[name] = r; print(name, json.dumps(r), flush=True)
json.dump(res, open("/tmp/revB17/l2id.json", "w"), indent=1)
