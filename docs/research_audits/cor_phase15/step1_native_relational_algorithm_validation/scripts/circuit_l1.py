"""Workstream B1: weight-level decomposition of HOW layer 1 forms the perfectness x agreement-at-q conjunction.
Design: Investigation XI 2x2 within-episode factorial (TRUE: n_h(q)=4, A = agree at q, P = agree at the one other disagreeing question;
e_h = 1 only in xAP).  dd(.) = (.)_AP - (.)_A - (.)_P + (.)_00.
(1) Per layer-1 head h, at the q7 query tokens t, the output is o_t = sum_s a_ts u_s with u_s = LN1(x_s) W_V W_O (x = resid1).  Write
    a_c = a_00 + da_c, u_c = u_00 + du_c.  Then EXACTLY
        dd(o) = [dd(a) u_00]  +  [a_00 dd(u)]  +  [da_AP du_AP - da_A du_A - da_P du_P]
              =   PATTERN     +    VALUE      +        CROSS (pattern change x value change)
    Reported: norm of each term projected on the E-subspace Q (direct path to resid2) and their share of <dd(o)_E, total dd_E>.
    Key-token attribution of PATTERN: which key groups' attention probabilities change jointly with A and P:
        hq  = the two source tokens of group h at q;  hoth = group h's source tokens at the other 5 questions;
        other = all other source tokens;  ro = readout / episode tokens.
    Score decomposition: dd of the pre-softmax scores (q_t . k_s) by key group (are scores themselves interactive, or is the
    interaction created by softmax normalisation of additive score changes?):  ddS (scores) vs dd(a) under additive scores.
(2) MLP1: input m_c = resid1_c + attn1_c.  own = MLP1(m_A + m_P - m_00) vs transmitted = MLP1(m_AP) - MLP1(m_A + m_P - m_00);
    neuron-level: fraction of the E-projected MLP1 interaction carried by 'threshold-AND' neurons (pre-activation < 0 in 00, A, P and > 0 in AP)
    versus neurons in a smooth regime.
(3) LayerNorm: interaction created by LN2's data-dependent scale: MLP1 with LN2 statistics frozen at the 00 cell vs actual.
Usage: circuit_l1.py RSUF SPEC... -> logs/circuit_l1<TAG>.json   (FA_EVAL as factorial.py)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
EVP = os.environ.get("FA_EVAL", "dev")
if EVP.startswith("stream:"):
    _r = np.random.default_rng(int(EVP[7:])); ev = dict(X=NV.Bm.sample_fast(NV.ENV, 6000, _r)["X"]); ev["q"] = _r.integers(0, 6, 6000)
else: ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP)
X = ev["X"]; q = ev["q"]; n = len(X); C = GL.counts(X, 3); a = X.reshape(n, 6, 3, 2); eq = a[..., 0] == a[..., 1]; rng = np.random.default_rng(1104)
cells = {k: [] for k in ("x00", "xA", "xP", "xAP")}; qq, hh = [], []
for i in range(n):
    hs = [h for h in range(3) if not eq[i, q[i], h] and C[i, q[i], h] == 4]
    if not hs: continue
    h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
    x0 = X[i].copy(); xA = x0.copy(); xA[q[i], 2 * h + 1] = xA[q[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
    for k, x in zip(cells, (x0, xA, xP, xAP)): cells[k].append(x)
    qq.append(q[i]); hh.append(h)
    if len(qq) >= 400: break
cells = {k: np.array(v) for k, v in cells.items()}; qq = np.array(qq); hh = np.array(hh); N_ = len(qq); ii = np.arange(N_)
tok = np.concatenate([(36 + qq)[:, None], qq[:, None] * 6 + np.arange(6)[None]], 1)              # (N,7) query tokens
grp = np.full((N_, 43), 3)                                                                       # key groups: 0 hq, 1 hoth, 2 other, 3 ro/episode
for k in range(N_):
    for qp in range(6):
        for s in range(6):
            t = qp * 6 + s; g = s // 2
            grp[k, t] = 0 if (g == hh[k] and qp == qq[k]) else 1 if g == hh[k] else 2
def ln(x, g, b): mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True); return (x - mu) / np.sqrt(v + 1e-5) * g + b
gelu = lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
dd = lambda d: d["xAP"] - d["xA"] - d["xP"] + d["x00"]
out = {}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy")))
    L1 = {k: np.asarray(v) for k, v in M["layers"][1].items()}; recs = {}
    for c, Xc in cells.items():
        r = {}; F.forward(M, jnp.asarray(Xc), rec=r); recs[c] = {k: np.asarray(v) for k, v in r.items()}
    projE = lambda Z: Z[ii[:, None], tok].reshape(N_, -1) @ Q                                      # (N,43,64) -> (N,6)
    ddE_total = projE(dd({c: recs[c]["resid2"] for c in cells}))
    res = dict(n=N_, heads={})
    for h in range(4):
        z = {c: ln(recs[c]["resid1"], L1["g1"], L1["b1"]) for c in cells}
        u = {c: np.einsum("ntd,de,ef->ntf", z[c], L1["WV"][h], L1["WO"][h]) for c in cells}         # (N,43,64)
        A = {c: recs[c]["pattern1"][:, h] for c in cells}                                           # (N,43,43)
        Aq = {c: A[c][ii[:, None], tok] for c in cells}                                              # (N,7,43)
        o = lambda Ac, uc: np.einsum("nts,nsd->ntd", Ac, uc)
        da = {c: Aq[c] - Aq["x00"] for c in cells}; du = {c: u[c] - u["x00"] for c in cells}
        PAT = o(dd(Aq), u["x00"]); VAL = o(Aq["x00"], dd(u)); CRS = o(da["xAP"], du["xAP"]) - o(da["xA"], du["xA"]) - o(da["xP"], du["xP"])
        TOT = dd({c: o(Aq[c], u[c]) for c in cells}); chk = float(np.abs(PAT + VAL + CRS - TOT).max())
        full = lambda Zt: np.zeros((N_, 43, 64))
        def E7(Z7):                                                                                   # Z7 (N,7,64) at q7 tokens -> E coords
            return Z7.reshape(N_, -1) @ Q
        tE = E7(TOT); den = float((tE * ddE_total).sum())
        terms = {nm: E7(Z) for nm, Z in (("pattern", PAT), ("value", VAL), ("cross", CRS))}
        # key-group attribution of the PATTERN term (in E): split dd(a) by key group
        kg = {}
        for gi, gname in enumerate(("hq", "hoth", "other", "ro")):
            m = (grp == gi)[:, None, :]; kg[gname] = float((E7(o(dd(Aq) * m, u["x00"])) * ddE_total).sum() / (float((ddE_total ** 2).sum()) + 1e-12))
        # cross term by key group (which keys' attention change x which keys' value change)
        kc = {}
        for gi, gname in enumerate(("hq", "hoth", "other", "ro")):
            m = (grp == gi)[:, None, :]; Z = o(da["xAP"] * m, du["xAP"]) - o(da["xA"] * m, du["xA"]) - o(da["xP"] * m, du["xP"])
            kc[gname] = float((E7(Z) * ddE_total).sum() / (float((ddE_total ** 2).sum()) + 1e-12))
        # scores: is the pre-softmax score interactive?
        Wq, Wk = L1["WQ"][h], L1["WK"][h]
        S = {c: np.einsum("nte,nse->nts", np.einsum("ntd,de->nte", z[c][ii[:, None], tok], Wq), np.einsum("nsd,de->nse", z[c], Wk)) / 4.0 for c in cells}
        ddS = dd(S); mainS = np.abs(S["xA"] - S["x00"]).mean() + np.abs(S["xP"] - S["x00"]).mean()
        Sadd = S["xA"] + S["xP"] - S["x00"]; sm = lambda s: np.exp(s - s.max(-1, keepdims=True)) / np.exp(s - s.max(-1, keepdims=True)).sum(-1, keepdims=True)
        Aadd = sm(Sadd); PAT_softmax_only = o(Aadd - Aq["xA"] - Aq["xP"] + Aq["x00"], u["x00"])        # interaction of a created by softmax of ADDITIVE scores
        tot2 = float((ddE_total ** 2).sum()) + 1e-12
        sc = {}
        for gi, gname in enumerate(("hq", "hoth", "other")):
            mk = np.broadcast_to((grp == gi)[:, None, :], S["x00"].shape)
            sc[gname] = dict(A=float((S["xA"] - S["x00"])[mk].mean()), P=float((S["xP"] - S["x00"])[mk].mean()), dd=float(ddS[mk].mean()),
                             attn00=float(Aq["x00"][mk].reshape(N_, 7, -1).sum(-1).mean()))
        qshare = {}
        for qi, qn in ((0, "readout"), (None, "sources")):
            msk = np.zeros((N_, 7, 1)); msk[:, 0 if qi == 0 else slice(1, 7)] = 1
            qshare[qn] = float((E7(PAT * msk) * ddE_total).sum() / (float((ddE_total ** 2).sum()) + 1e-12))
        res["heads"][h] = dict(score_main_by_key=sc, pattern_by_query=qshare, decomp_check=chk, share_of_total_E=den / tot2,
                               pattern=float((terms["pattern"] * ddE_total).sum() / tot2), value=float((terms["value"] * ddE_total).sum() / tot2), cross=float((terms["cross"] * ddE_total).sum() / tot2),
                               pattern_by_key=kg, cross_by_key=kc, score_rel_interaction=float(np.abs(ddS).mean() / mainS),
                               pattern_from_softmax_of_additive_scores=float((E7(PAT_softmax_only) * ddE_total).sum() / tot2))
    # MLP1
    m = {c: recs[c]["resid1"] + recs[c]["heads1"].sum(1) for c in cells}
    def mlp(x, freeze=None):
        if freeze is None: zz = ln(x, L1["g2"], L1["b2"])
        else: mu, sd = freeze; zz = (x - mu) / sd * L1["g2"] + L1["b2"]
        pre = zz @ L1["W1"] + L1["c1"]; return gelu(pre) @ L1["W2"] + L1["c2"], pre
    madd = m["xA"] + m["xP"] - m["x00"]; tot2 = float((ddE_total ** 2).sum()) + 1e-12
    out_c = {c: mlp(m[c]) for c in cells}; own = mlp(madd)[0] - out_c["xA"][0] - out_c["xP"][0] + out_c["x00"][0]; trans = out_c["xAP"][0] - mlp(madd)[0]
    E43 = lambda Z: Z[ii[:, None], tok].reshape(N_, -1) @ Q
    pre = {c: out_c[c][1][ii[:, None], tok] for c in cells}                                        # (N,7,256)
    andm = (pre["x00"] < 0) & (pre["xA"] < 0) & (pre["xP"] < 0) & (pre["xAP"] > 0)
    hidd = {c: gelu(pre[c]) for c in cells}; ddh = dd(hidd); W2 = L1["W2"]
    contrib_and = np.einsum("ntk,kd->ntd", ddh * andm, W2).reshape(N_, -1) @ Q; contrib_all = np.einsum("ntk,kd->ntd", ddh, W2).reshape(N_, -1) @ Q
    # LN2 frozen at x00 statistics
    mu0 = m["x00"].mean(-1, keepdims=True); sd0 = np.sqrt(((m["x00"] - mu0) ** 2).mean(-1, keepdims=True) + 1e-5)
    ddfrz = dd({c: mlp(m[c], (mu0, sd0))[0] for c in cells})
    res["mlp1"] = dict(share_of_total_E=float((E43(dd({c: out_c[c][0] for c in cells})) * ddE_total).sum() / tot2),
                       own_nonlinear=float((E43(own) * ddE_total).sum() / tot2), transmitted_from_attention=float((E43(trans) * ddE_total).sum() / tot2),
                       frac_threshold_AND_neurons=float(andm.mean()), share_by_AND_neurons=float((contrib_and * ddE_total).sum() / ((contrib_all * ddE_total).sum() + 1e-12)),
                       LN2_frozen_vs_actual_E=float((E43(ddfrz) * ddE_total).sum() / ((E43(dd({c: out_c[c][0] for c in cells})) * ddE_total).sum() + 1e-12)))
    res["attn_sum_share"] = float(sum(v["share_of_total_E"] for v in res["heads"].values()))
    out[sp] = res
    hs_ = {h: (round(v["share_of_total_E"], 2), round(v["pattern"], 2), round(v["value"], 2), round(v["cross"], 2), round(v["pattern_from_softmax_of_additive_scores"], 2), round(v["score_rel_interaction"], 3)) for h, v in res["heads"].items()}
    print(sp, "heads(share,pat,val,cross,softmaxAdd,scoreI)", hs_, "| mlp1", {k: round(v, 3) for k, v in res["mlp1"].items()}, flush=True)
    for h, v in res["heads"].items():
        if abs(v["share_of_total_E"]) > 0.1: print("   head", h, "pat_by_key", {k: round(x, 2) for k, x in v["pattern_by_key"].items()}, "pat_by_query", {k: round(x, 2) for k, x in v["pattern_by_query"].items()}, "scores", {k: {kk: round(x, 3) for kk, x in d.items()} for k, d in v["score_main_by_key"].items()})
json.dump(out, open(os.path.join(HERE, "..", "logs", "circuit_l1" + os.environ.get("FA_TAG", "") + ".json"), "w"), indent=1)
