"""XIII-D Workstream E3: are the relational routes Q / R / O separate computations, redundant backups, or additive parts of one mechanism?
(1) Path-patching as XIII-C d_routes on SMALL (one relational edit) and LARGE (the same group's agreement flipped at TWO other questions)
    on-support pairs: sufficiencies, redundancy, and the pairwise second-order interactions I_XY = s(cut XY) - s(cut X) - s(cut Y) + s(none).
    Under local linearity interactions are ~0 for small edits; growth with edit size indicates genuine non-additive interaction.
(2) FUNCTIONAL blocking (forward-time attention masks, all episodes): BLOCK_R = layer-1 readout -> source attention masked; BLOCK_Q = layer-1
    attention from q's source tokens to other questions' source tokens masked; BLOCK_O = layer-2 readout -> other-question source attention
    masked; every subset.  Reported: KL(Bayes || model) and Bayes-argmax accuracy on natural episodes, and the relational response slope.
    Separate-and-sufficient routes predict that one open route alone supports near-normal relational behaviour; redundant backups predict
    little loss when one is blocked; additive components predict losses proportional to their shares.
Usage: DT2_TAG=_dev python3 scripts/d_routes2.py SPEC... -> logs/d_routes2<TAG>.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
ST = int(os.environ.get("DT2_STREAM", 90_700_000)); rng = np.random.default_rng(ST)
SM = RP.relational_pairs(600, np.random.default_rng(ST + 1))
def large_pairs(n=600):
    Xa, Xb, qq, hh, qps = [], [], [], [], []
    while len(qq) < n:
        X = NV.Bm.sample_fast(NV.ENV, 1, rng)["X"][0]; q = rng.integers(6); h = rng.integers(3); oth = [p for p in range(6) if p != q]; p1, p2 = rng.choice(oth, 2, replace=False)
        x = X.copy()
        for p in (p1, p2):
            if x[p, 2 * h] == x[p, 2 * h + 1]: x[p, 2 * h + 1] = (x[p, 2 * h] + 1 + rng.integers(2)) % 3
            else: x[p, 2 * h + 1] = x[p, 2 * h]
        PBa = GL.posterior_closed(RP.ENV1, X[None], np.array([q])); PBb = GL.posterior_closed(RP.ENV1, x[None], np.array([q]))
        if not (np.isfinite(PBa).all() and np.isfinite(PBb).all()) or np.abs(RP.cen(PBb) - RP.cen(PBa)).max() < 0.05: continue
        Xa.append(X); Xb.append(x); qq.append(q); hh.append(h); qps.append((p1, p2))
    Xa, Xb, qq = np.array(Xa), np.array(Xb), np.array(qq)
    return dict(Xb=Xa, Xd=Xb, q=qq, h=np.array(hh), qps=np.array(qps), PB=GL.posterior_closed(RP.ENV1, Xa, qq), PBd=GL.posterior_closed(RP.ENV1, Xb, qq))
LG = large_pairs()
Xn = NV.Bm.sample_fast(NV.ENV, 2000, rng)["X"]; qn = rng.integers(0, 6, 2000); PBn = GL.posterior_closed(RP.ENV1, Xn, qn)
def path_patch(M, P_, large):
    n = len(P_["q"]); i = np.arange(n); q_ = P_["q"]; h = P_["h"]; rB, rD = {}, {}
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), rec=rB))[i, q_]; F.forward(M, jnp.asarray(P_["Xd"]), rec=rD); z2b = jnp.asarray(rB["resid2"])
    up = np.zeros((n, 43), bool)
    qpl = P_["qps"] if large else P_["qp"][:, None]
    for c in range(qpl.shape[1]): up[i, qpl[:, c] * 6 + 2 * h] = True; up[i, qpl[:, c] * 6 + 2 * h + 1] = True
    mQ = np.zeros((n, 43), bool); mQ[i[:, None], q_[:, None] * 6 + np.arange(6)[None]] = True; mR = np.zeros((n, 43), bool); mR[i, 36 + q_] = True
    mO = np.zeros((n, 43), bool); mO[:, :36] = True; mO &= ~mQ; pred = RP.cen(P_["PBd"]) - RP.cen(P_["PB"])
    def tr(cut):
        hp = {(0, hd): (jnp.asarray(up), rD["heads0"][:, hd]) for hd in range(4)}; mp = {0: (jnp.asarray(up), rD["mlp_out0"])}; m1 = np.zeros((n, 43), bool)
        for c, mm in (("Q", mQ), ("R", mR), ("O", mO)):
            if c in cut: m1 |= mm
        if m1.any():
            for hd in range(4): hp[(1, hd)] = (jnp.asarray(m1), rB["heads1"][:, hd])
        rp = {2: (lambda x: jnp.where(jnp.asarray(up)[..., None], z2b, x))} if "O" in cut else None
        L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), head_patch=hp, mlp_patch=mp, resid_patch=rp))[i, q_]; return RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)), pred)["slope"]
    s = {"".join(c) or "none": tr(set(c)) for k in range(4) for c in itertools.combinations("QRO", k)}; tot = s["none"]; key = lambda c: "".join(x for x in "QRO" if x in c) or "none"
    suff = {r: s[key(set("QRO") - {r})] / tot for r in "QRO"}
    inter = {a + b: (s[key({a, b})] - s[key({a})] - s[key({b})] + s["none"]) / tot for a, b in (("Q", "R"), ("Q", "O"), ("R", "O"))}
    return dict(total=tot, sufficiency=suff, redundancy=sum(suff.values()) - 1, interactions=inter)
MSK = {}
def masks(q):
    n = len(q); m = {k: np.zeros((n, 43, 43), np.float32) for k in ("R1", "Q1", "O2")}; i = np.arange(n)
    for j in range(n):
        m["R1"][j, 36 + q[j], :36] = -1e9
        qs = np.arange(q[j] * 6, q[j] * 6 + 6); oth = np.setdiff1d(np.arange(36), qs)
        m["Q1"][j][np.ix_(qs, oth)] = -1e9; m["O2"][j, 36 + q[j], oth] = -1e9
    return m
def fwd_block(M, X, q, block):
    mk = masks(q); x = F.N.embed(dict(M["emb"]), jnp.asarray(X)); n = len(X)
    for l in range(3):
        lay = M["layers"][l]; z = F.N.ln(x, lay["g1"], lay["b1"]); b = np.zeros((n, 43, 43), np.float32)
        if l == 1 and "R" in block: b += mk["R1"]
        if l == 1 and "Q" in block: b += mk["Q1"]
        if l == 2 and "O" in block: b += mk["O2"]
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]) + jnp.asarray(b)[:, None], -1)
        x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"])
        x = x + jax.nn.gelu(F.N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
    return np.asarray(F.N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[np.arange(n), q]
def bt(M, X, q, block): return np.concatenate([fwd_block(M, X[a:a + 250], q[a:a + 250], block) for a in range(0, len(X), 250)])
out = {}
for sp in sys.argv[1:]:
    M = xc.load(sp); r = dict(small=path_patch(M, SM, False), large=path_patch(M, LG, True), block={})
    for k in range(4):
        for c in itertools.combinations("QRO", k):
            P = RP.sm(bt(M, Xn, qn, set(c))); La = bt(M, SM["Xb"], SM["q"], set(c)); Lb = bt(M, SM["Xd"], SM["q"], set(c))
            r["block"]["".join(c) or "none"] = dict(KL=float((PBn * (np.log(PBn + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()), acc=float((P.argmax(-1) == PBn.argmax(-1)).mean()),
                                                   rel_slope=RP.fid(RP.cen(RP.sm(Lb)) - RP.cen(RP.sm(La)), RP.cen(SM["PBd"]) - RP.cen(SM["PB"]))["slope"])
    out[sp] = r; print(sp, "small", {k: v for k, v in r["small"].items()}, "\n large", r["large"], "\n block", {k: {a: round(b, 3) for a, b in v.items()} for k, v in r["block"].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "d_routes2" + os.environ.get("DT2_TAG", "") + ".json"), "w"), indent=1, default=float)
