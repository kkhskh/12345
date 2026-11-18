import os, sys, json, numpy as np
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_operator_reuse"
sys.path.insert(0, PKG + "/src"); sys.path.insert(0, PKG + "/../step1_cross_family_relational_foundations/src")
import jax, jax.numpy as jnp, gatelib as GL, xviislot as XS, nets as N
S, Q = 6, 6; tok = XS.tok_index
def fwd(p, X, delta=None, lo=0, hi=2, attn_over=None, capture=False):
    """delta: dict slot->vector added to slot tokens before layer lo, removed after layer hi-1 (hi<L).
    attn_over: dict layer->(rows, A) replace attention rows (token idx array) with given A[N,H,len(rows),T]."""
    x = N.embed(p, jnp.asarray(X)); L = len(p["layers"]); caps = {}; atts = {}
    def add(x, s):
        for q, v in delta.items(): x = x.at[:, tok(q)].add(s * v)
        return x
    for l, lay in enumerate(p["layers"]):
        if delta is not None and l == lo: x = add(x, 1.0)
        if capture: caps[l] = np.asarray(x)
        z = N.ln(x, lay["g1"], lay["b1"])
        q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        a = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1]), -1)
        if attn_over is not None and l in attn_over:
            rows, A = attn_over[l]; a = a.at[:, :, rows].set(A)
        atts[l] = a
        x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
        x = x + jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
        if delta is not None and l == hi - 1 and hi < L: x = add(x, -1.0)
    if capture: caps[L] = np.asarray(x)
    zf = N.ln(x, p["gf"], p["bf"]); out = np.asarray(zf[:, S * Q:S * Q + Q] @ p["WA"])
    return out, atts, caps
def FG(D, j, f):
    a0 = f(D.X)[:, j].argmax(-1); af = f(D.Xf[j])[:, j].argmax(-1)
    return float(((af != a0) & (af == D.newG[j]))[D.dec[j]].mean())
res = {}
conds = {"A3": "3:0.9:0.9:0.3333333333333333", "A2": "2:0.9:0.9:0.05"}
Ds = {k: XS.SlotDiag(c, n=600) for k, c in conds.items()}
for name in ["A3_s804", "A3_s803", "A2_s704", "A2_s705"]:
    A = json.load(open(f"{PKG}/logs/analysis/{name}.json")); D = Ds[name[:2]]
    p = GL.load(f"{PKG}/confirm_outputs/{name}/step{A['selected']}.npz"); E = np.asarray(p["Eq"]); r = {}
    i, j = A["pairs"][0]; r["pair"] = [i, j]
    # (a) complement identity
    pi = np.arange(6); pi[i], pi[j] = j, i; Xp = D.X[:, pi]
    o1 = np.asarray(XS.forward(p, jnp.asarray(D.X), (i, j, 2, 3))); o2 = np.asarray(XS.forward(p, jnp.asarray(Xp), (i, j, 0, 2)))
    r["complement_maxdiff_[2,3)_vs_pi[0,2)"] = float(np.abs(o1[:, pi] - o2).max())
    # sanity: my fwd reproduces package relabel
    dl = {i: jnp.asarray(E[j] - E[i]), j: jnp.asarray(E[i] - E[j])}
    r["myfwd_vs_pkg"] = float(np.abs(fwd(p, D.X, dl)[0] - np.asarray(XS.forward(p, jnp.asarray(D.X), (i, j, 0, 2)))).max())
    r["base_FG_j"] = FG(D, j, lambda X: fwd(p, X)[0]); r["relabel02_FG_j"] = FG(D, j, lambda X: fwd(p, X, dl)[0])
    # (d) one-sided controls in window [0,2): j gets random norm-matched code; j gets i's code without exchange (duplicate)
    rng = np.random.default_rng(0); rr = []
    for t in range(5):
        v = rng.normal(size=E.shape[1]); v = v / np.linalg.norm(v) * np.linalg.norm(E[j]); rr.append(FG(D, j, lambda X: fwd(p, X, {j: jnp.asarray(v - E[j])})[0]))
    r["random_code_02_FG_j"] = rr
    r["dup_code_i_02_FG_j"] = FG(D, j, lambda X: fwd(p, X, {j: jnp.asarray(E[i] - E[j])})[0])
    r["dup_code_i_02_FG_i"] = FG(D, i, lambda X: fwd(p, X, {j: jnp.asarray(E[i] - E[j])})[0])
    r["dup_code_i_03_FG_j"] = FG(D, j, lambda X: fwd(p, X, {j: jnp.asarray(E[i] - E[j])}, 0, 3)[0])
    # (c) content-matched routing: recipient rows from the relabel-[0,2) run on the SAME input, inserted into the baseline run
    rows = tok(j)
    def routed(X, layers):
        _, at, _ = fwd(p, X, dl); return fwd(p, X, attn_over={l: (rows, at[l][:, :, rows]) for l in layers})[0]
    def relabel_own_route(X, layers):
        _, at, _ = fwd(p, X); return fwd(p, X, dl, attn_over={l: (rows, at[l][:, :, rows]) for l in layers})[0]
    r["matched_route_L01_FG_j"] = FG(D, j, lambda X: routed(X, (0, 1)))
    r["relabel02_but_own_route_L01_FG_j"] = FG(D, j, lambda X: relabel_own_route(X, (0, 1)))
    # (b) slot identity at layer-2 input after projecting out span(E_q): nearest-centroid decoding on readout tokens and source tokens
    _, _, c = fwd(p, D.X[:300], capture=True); _, _, cr = fwd(p, D.X[:300], dl, capture=True)
    Qb, _ = np.linalg.qr(E.T); proj = lambda h: h - (h @ Qb) @ Qb.T
    for lay in (1, 2, 3):
        H = proj(c[lay][:, 36:42]); cent = H.mean(0)  # 6 x d
        Hc = H - cent.mean(0); cen_c = cent - cent.mean(0)
        acc = float((np.argmin(((Hc[:, :, None, :] - cen_c[None, None]) ** 2).sum(-1), -1) == np.arange(6)).mean())
        Hr = proj(cr[lay][:, 36:42]) - cent.mean(0); lab = np.argmin(((Hr[:, :, None, :] - cen_c[None, None]) ** 2).sum(-1), -1)
        r[f"L{lay}in_readout_slotdecode_acc_noEq"] = acc
        r[f"L{lay}in_relabel_recipient_decoded_as_i_frac"] = float((lab[:, j] == i).mean()); r[f"L{lay}in_relabel_recipient_decoded_as_j_frac"] = float((lab[:, j] == j).mean())
        r[f"L{lay}in_Eq_norm_frac"] = float(np.linalg.norm(c[lay][:, 36:42] @ Qb, axis=-1).mean() / np.linalg.norm(c[lay][:, 36:42], axis=-1).mean())
    res[name] = r; print(name, json.dumps(r), flush=True)
json.dump(res, open("/tmp/revB17/checks.json", "w"), indent=1)
