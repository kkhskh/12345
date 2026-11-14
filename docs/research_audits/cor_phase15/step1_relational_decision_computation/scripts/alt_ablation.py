"""Workstream B3/B4: does the layer-1 mechanism survive intervention designs OTHER than Investigation XI's additive completion, and is
there compensation?  TRUE 2x2 factorial cells (x00, xA, xP, xAP; e_h = 1 only in xAP).  Outcome at xAP:
   KL(Bayes || model), KL(cf_e0 || model) (cf_e0 = Bayes formula with e_h forced to 0), pref_e0, E-interaction remaining.
Interventions on the xAP run (all downstream recomputed unless stated):
  addcomp      : Investigation XI additive completion of every layer-1 head + MLP1 (hybrid activation)       [reference]
  path_inA     : PATH patching with natural activations: layer 1 (attention + MLP1) receives resid1 from xA (h agrees at q but is NOT
                 perfect); the residual bypass still carries xAP's resid1 -> P reaches layer 2 only around layer 1
  path_inP     : same with resid1 from xP (perfect, not agreeing at q)
  resamp_A     : layer-1 outputs (heads + MLP1, all tokens) replaced by those of the natural xA run (the P main effect of layer 1 is lost too)
  mean_L1      : layer-1 outputs replaced by their mean over 2000 natural episodes (token-wise)                  [non-specific control]
  pattern_A    : layer-1 attention PATTERNS replaced by the xA patterns (values from xAP); MLP1 recomputed       [removes P's effect on patterns]
  pattern_P    : patterns from xP (removes A's effect on patterns);  values_A / values_P: layer-1 VALUES computed from xA / xP resid1
  pattern_addcomp(_mlp_frozen): patterns replaced by their additive completion (pattern interaction only); attn_addcomp_mlp_{recomputed,frozen};
                 mlp_addcomp_attn_intact: decompositions of the XI additive completion into its attention and MLP parts (direct vs total)
  scrub_match  : E-subspace coordinates (resid2, q7) replaced by those of a random NATURAL donor episode with the SAME eligibility vector e
  scrub_e0     : same, donor with e_h = 0 and the other groups' e unchanged
  path_inA_L0dir: path_inA but layer 1 receives xA input ONLY at the q7 query tokens (keys/values from xAP)    [query-side vs key-side]
Compensation (single components, natural path patching): for each layer-1 head j: head j's input from xA (others intact);
  report total effect on the E-interaction vs head j's direct share (total < direct => compensation; total > direct => amplification).
Usage: alt_ablation.py RSUF SPEC... -> logs/alt_ablation<TAG>.json   (FA_EVAL as in factorial.py)"""
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
cells = {k: np.array(v) for k, v in cells.items()}; qq = np.array(qq); hh = np.array(hh); N_ = len(qq); ii = np.arange(N_); Xp = cells["xAP"]
env = GL.Env(3, 1); rel, col, noi = GL._group_lik(env, Xp); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, qq]); Lg = rel[ii, qq] / noi
def evec(Xc, qc):
    j = np.arange(len(qc)); A_ = Xc.reshape(len(Xc), 6, 3, 2)[j, qc]; return ((GL.counts(Xc, 3)[j, qc] == 5) & (A_[..., 0] == A_[..., 1])).astype(float)
e = evec(Xp, qq); nz = lambda P: P / P.sum(-1, keepdims=True)
cfB = nz((W[..., None] * (e.sum(1, keepdims=True) - e)[..., None] * Lg).sum(1)); e0 = e.copy(); e0[ii, hh] = 0
cf0 = nz(np.maximum((W[..., None] * (e0.sum(1, keepdims=True) - e0)[..., None] * Lg).sum(1), 1e-30))
# natural donor pool (for scrubbing and mean ablation)
rngd = np.random.default_rng(77_000_000); D = NV.Bm.sample_fast(NV.ENV, 20000, rngd); XD = D["X"]; qD = rngd.integers(0, 6, len(XD)); eD = evec(XD, qD)
key = lambda v: (v * np.array([1, 2, 4])).sum(-1).astype(int); kD = key(eD)
don_m = np.array([rng.choice(np.where(kD == k)[0]) for k in key(e)]); don_0 = np.array([rng.choice(np.where(kD == k)[0]) for k in key(e0)])
tok = lambda qc: np.concatenate([(36 + qc)[:, None], qc[:, None] * 6 + np.arange(6)[None]], 1)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
def run_layers(M, x_in, L1_in=None, L1_out=None, mlp_out=None, pat=None, q7_only_in=None, resid2_fn=None, attn_override=None, val_in=None):
    """Manual forward from resid1 (x_in).  L1_in: input to layer-1 sublayers (default x_in); q7_only_in: (input, mask) mix at query tokens only.
    L1_out: replacement (attn_sum, mlp) for layer 1; pat: replacement attention patterns (N,H,43,43)."""
    lay = M["layers"][1]; x = jnp.asarray(x_in)
    if L1_out is not None: x2 = x + L1_out[0] + L1_out[1]
    else:
        xin = jnp.asarray(L1_in) if L1_in is not None else x
        z = F.N.ln(xin, lay["g1"], lay["b1"])
        if q7_only_in is not None:                                                     # queries from q7_only_in at mask tokens, keys/values from xin
            zq = jnp.where(q7_only_in[1][..., None], F.N.ln(jnp.asarray(q7_only_in[0]), lay["g1"], lay["b1"]), z)
        else: zq = z
        qh = jnp.einsum("ntd,hde->nhte", zq, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z if val_in is None else F.N.ln(jnp.asarray(val_in), lay["g1"], lay["b1"]), lay["WV"])
        A_ = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1) if pat is None else jnp.asarray(pat)
        oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A_, vh), lay["WO"]).sum(1)
        if attn_override is not None: oh = jnp.asarray(attn_override)
        xa = x + oh
        if mlp_out is not None: mo = jnp.asarray(mlp_out)
        else:
            hid = jax.nn.gelu(F.N.ln(xa if L1_in is None else (xin + oh), lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); mo = hid @ lay["W2"] + lay["c2"]
        x2 = xa + mo
    if resid2_fn is not None: x2 = resid2_fn(x2)
    x3 = F.block_heads(M["layers"][2], x2, None, 2, None, None)
    return np.asarray(F.N.ln(x3, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[ii, qq], np.asarray(x2)
out = {}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy"))); Qj = jnp.asarray(Q)
    R = {}
    for c, Xc in cells.items(): r = {}; F.forward(M, jnp.asarray(Xc), rec=r); R[c] = {k: np.asarray(v) for k, v in r.items()}
    q7E = lambda Z: Z[ii[:, None], tok(qq)].reshape(N_, -1) @ Q
    dd0 = q7E(R["xAP"]["resid2"]) - q7E(R["xA"]["resid2"]) - q7E(R["xP"]["resid2"]) + q7E(R["x00"]["resid2"]); I0 = np.linalg.norm(dd0, axis=1).mean()
    def score(L, x2):
        P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True); kB, k0 = kl(cfB, P), kl(cf0, P)
        rem = np.linalg.norm(q7E(x2) - q7E(R["xA"]["resid2"]) - q7E(R["xP"]["resid2"]) + q7E(R["x00"]["resid2"]), axis=1).mean() / I0
        return dict(KL_bayes=float(kB.mean()), KL_e0=float(k0.mean()), pref_e0=float((k0 < kB).mean()), E_interaction_remaining=float(rem),
                    per_episode_KLdiff=(k0 - kB).tolist())
    xr1 = R["xAP"]["resid1"]; res = {}
    res["intact"] = score(*run_layers(M, xr1))
    att = lambda c: R[c]["heads1"].sum(1); mlp = lambda c: R[c]["mlp_out1"]
    res["addcomp"] = score(*run_layers(M, xr1, L1_out=(att("xA") + att("xP") - att("x00"), mlp("xA") + mlp("xP") - mlp("x00"))))
    res["path_inA"] = score(*run_layers(M, xr1, L1_in=R["xA"]["resid1"])); res["path_inP"] = score(*run_layers(M, xr1, L1_in=R["xP"]["resid1"]))
    res["resamp_A"] = score(*run_layers(M, xr1, L1_out=(att("xA"), mlp("xA"))))
    rD = {}; F.forward(M, jnp.asarray(XD[:2000]), rec=rD); mA = np.asarray(rD["heads1"]).sum(1).mean(0); mM = np.asarray(rD["mlp_out1"]).mean(0)
    res["mean_L1"] = score(*run_layers(M, xr1, L1_out=(np.broadcast_to(mA, xr1.shape), np.broadcast_to(mM, xr1.shape))))
    res["pattern_A"] = score(*run_layers(M, xr1, pat=R["xA"]["pattern1"]))
    res["pattern_P"] = score(*run_layers(M, xr1, pat=R["xP"]["pattern1"]))                   # removes A's effect on attention patterns
    res["values_A"] = score(*run_layers(M, xr1, val_in=R["xA"]["resid1"]))                    # removes P's effect on layer-1 values
    res["values_P"] = score(*run_layers(M, xr1, val_in=R["xP"]["resid1"]))                    # removes A's effect on layer-1 values
    padd = np.clip(R["xA"]["pattern1"] + R["xP"]["pattern1"] - R["x00"]["pattern1"], 0, None); padd = padd / padd.sum(-1, keepdims=True)
    res["pattern_addcomp"] = score(*run_layers(M, xr1, pat=padd))                                  # only the PATTERN interaction removed; MLP1 recomputed
    # direct vs total for the pattern interaction: MLP1 output frozen at its intact xAP value
    res["pattern_addcomp_mlp_frozen"] = score(*run_layers(M, xr1, pat=padd, mlp_out=R["xAP"]["mlp_out1"]))
    # attention interaction removed (additive completion of head outputs), MLP1 recomputed vs frozen
    res["attn_addcomp_mlp_recomputed"] = score(*run_layers(M, xr1, L1_out=(att("xA") + att("xP") - att("x00"), None)) if False else run_layers(M, xr1, pat=None, mlp_out=None, L1_in=None, L1_out=None))
    res.pop("attn_addcomp_mlp_recomputed")
    res["attn_addcomp_mlp_recomputed"] = score(*run_layers(M, xr1, attn_override=att("xA") + att("xP") - att("x00")))
    res["attn_addcomp_mlp_frozen"] = score(*run_layers(M, xr1, attn_override=att("xA") + att("xP") - att("x00"), mlp_out=R["xAP"]["mlp_out1"]))
    res["mlp_addcomp_attn_intact"] = score(*run_layers(M, xr1, mlp_out=mlp("xA") + mlp("xP") - mlp("x00")))
    msk = np.zeros((N_, 43), bool); msk[ii[:, None], tok(qq)] = True
    res["path_inA_queries_only"] = score(*run_layers(M, xr1, q7_only_in=(R["xA"]["resid1"], jnp.asarray(msk))))
    res["path_inA_keys_only"] = score(*run_layers(M, R["xAP"]["resid1"], L1_in=R["xA"]["resid1"], q7_only_in=(R["xAP"]["resid1"], jnp.asarray(msk))))
    for nm, don in (("scrub_match", don_m), ("scrub_e0", don_0)):
        rd = {}; F.forward(M, jnp.asarray(XD[don]), rec=rd); zD = np.asarray(rd["resid2"])[ii[:, None], tok(qD[don])].reshape(N_, -1)
        tq = jnp.asarray(tok(qq)); iij = jnp.asarray(ii)[:, None]; zDj = jnp.asarray(zD)
        def fn(x2, zDj=zDj):
            vb = x2[iij, tq].reshape(N_, -1); vp = vb + ((zDj - vb) @ Qj) @ Qj.T; return x2.at[iij, tq].set(vp.reshape(N_, 7, 64))
        res[nm] = score(*run_layers(M, xr1, resid2_fn=fn))
    # compensation: single heads, natural path patching of head j's input from xA
    comp = {}
    lay = M["layers"][1]
    for j in range(4):
        z_ap = F.N.ln(jnp.asarray(xr1), lay["g1"], lay["b1"]); z_a = F.N.ln(jnp.asarray(R["xA"]["resid1"]), lay["g1"], lay["b1"])
        def head_out(zin, h):
            qh = zin @ lay["WQ"][h]; kh = zin @ lay["WK"][h]; vh = zin @ lay["WV"][h]; A_ = jax.nn.softmax(jnp.einsum("nte,nse->nts", qh, kh) / 4.0, -1); return (A_ @ vh) @ lay["WO"][h]
        oh = [head_out(z_ap, h) for h in range(4)]; oh[j] = head_out(z_a, j); attn = sum(oh); xa = jnp.asarray(xr1) + attn
        hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); mo = hid @ lay["W2"] + lay["c2"]
        s_ = score(*run_layers(M, xr1, L1_out=(attn, mo)))
        # direct share of head j in the intact interaction
        hj = {c: R[c]["heads1"][:, j] for c in cells}; ddj = q7E(hj["xAP"]) - q7E(hj["xA"]) - q7E(hj["xP"]) + q7E(hj["x00"])
        direct = float((ddj * dd0).sum() / (dd0 ** 2).sum())
        # head j's own interaction after its input is replaced (it keeps the A-direction main effects only) is ~ removed; total effect:
        comp[j] = dict(direct_share=direct, total_removed=1 - s_["E_interaction_remaining"], KL_bayes=s_["KL_bayes"], KL_e0=s_["KL_e0"])
    res["single_head_path_inA"] = comp
    out[sp] = res
    print(sp, {k: (round(v["E_interaction_remaining"], 2), round(v["KL_bayes"], 3), round(v["KL_e0"], 3), round(v["pref_e0"], 2)) for k, v in res.items() if k != "single_head_path_inA"}, flush=True)
    print("   single heads (direct, totalRemoved)", {j: (round(v["direct_share"], 2), round(v["total_removed"], 2)) for j, v in comp.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "alt_ablation" + os.environ.get("FA_TAG", "") + ".json"), "w"))
