"""POST-HOC (review 3, not frozen): class-agnostic decomposition of LAYER 2 for purely relational EA pairs (xA -> xAP; values at q identical),
to identify how the early-readout class combines evidence in layer 2.  Components from the natural donor, for ALL four layer-2 heads:
  A  = readout-token attention rows;  VR = value inputs at the readout token;  VQ = value inputs at the 6 q tokens;  VO = value inputs at the 30
  other source tokens;  M = MLP2 output at the readout token;  K = residual skip at the readout token.
All 64 subsets -> Shapley shares of the projection of the patched change on the total change, plus attention-vs-MLP pairwise interaction.
Stream 79_700_000.  Usage: l2_decomp_all.py SPEC... -> logs/l2_decomp_all<TAG>.json"""
import json, os, sys, itertools, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
_r = np.random.default_rng(int(os.environ.get("LD_STREAM", 79_700_000))); X0 = NV.Bm.sample_fast(NV.ENV, 8000, _r)["X"]; q0 = _r.integers(0, 6, 8000)
n0 = len(X0); C = GL.counts(X0, 3); a = X0.reshape(n0, 6, 3, 2); eq = a[..., 0] == a[..., 1]
rng = np.random.default_rng(1404); B, Dn, qq = [], [], []
for i in range(n0):
    hs = [h for h in range(3) if not eq[i, q0[i], h] and C[i, q0[i], h] == 4]
    if not hs: continue
    h = hs[rng.integers(len(hs))]; others = [p for p in range(6) if p != q0[i] and not eq[i, p, h]]; qp = others[rng.integers(len(others))]
    xA = X0[i].copy(); xA[q0[i], 2 * h + 1] = xA[q0[i], 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]; B.append(xA); Dn.append(xAP); qq.append(q0[i])
    if len(qq) >= 500: break
B, Dn, qq = np.array(B), np.array(Dn), np.array(qq); N_ = len(qq); ii = np.arange(N_); ro = 36 + qq
mR = np.zeros((N_, 43), bool); mR[ii, ro] = True; mQ = np.zeros((N_, 43), bool); mQ[ii[:, None], qq[:, None] * 6 + np.arange(6)[None]] = True
mO = np.zeros((N_, 43), bool); mO[:, :36] = True; mO &= ~mQ
comps = ("A", "VR", "VQ", "VO", "M", "K"); out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); lay = M["layers"][2]
    rb, rd = {}, {}; F.forward(M, jnp.asarray(B), rec=rb); F.forward(M, jnp.asarray(Dn), rec=rd)
    xb, xd = np.asarray(rb["resid2"]), np.asarray(rd["resid2"]); Ad = np.asarray(rd["pattern2"]); mlpd = np.asarray(rd["mlp_out2"])
    def run(sub):
        x = jnp.asarray(xb); z = F.N.ln(x, lay["g1"], lay["b1"]); zd = F.N.ln(jnp.asarray(xd), lay["g1"], lay["b1"])
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"]); vd = jnp.einsum("ntd,hde->nhte", zd, lay["WV"])
        for c, m in (("VR", mR), ("VQ", mQ), ("VO", mO)):
            if c in sub: vh = jnp.where(jnp.asarray(m)[:, None, :, None], vd, vh)
        A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1); t = jnp.asarray(ro); nn = jnp.arange(N_)
        if "A" in sub: A = A.at[nn, :, t].set(jnp.asarray(Ad[ii, :, ro]))
        oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"]).sum(1)
        xs = x if "K" not in sub else x.at[nn, t].set(jnp.asarray(xd[ii, ro]))
        xa = xs + oh; hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); mo = hid @ lay["W2"] + lay["c2"]
        if "M" in sub: mo = mo.at[nn, t].set(jnp.asarray(mlpd[ii, ro]))
        L = np.asarray(F.N.ln(xa + mo, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[ii, qq]; return RP.sm(L)
    P0 = run(()); TE = RP.cen(run(comps)) - RP.cen(P0); vals = {}
    for k in range(7):
        for sub in itertools.combinations(comps, k):
            nie = RP.cen(run(sub)) - RP.cen(P0); vals["+".join(sub)] = float((nie * TE).sum() / (TE * TE).sum())
    key = lambda s: "+".join(c for c in comps if c in s)
    sh = {c: sum(math.factorial(len(s)) * math.factorial(5 - len(s)) / math.factorial(6) * (vals[key(set(s) | {c})] - vals[key(set(s))])
                 for k in range(6) for s in itertools.combinations([o for o in comps if o != c], k)) for c in comps}
    att = sh["A"] + sh["VR"] + sh["VQ"] + sh["VO"]
    out[sp] = dict(n=int(N_), shapley=sh, attention_total=att, mlp=sh["M"], skip=sh["K"], subset_proj=vals, TE_mag=float(np.abs(TE).mean()))
    print(sp[:6], "shapley", {k: round(v, 2) for k, v in sh.items()}, "attention total", round(att, 2), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l2_decomp_all" + os.environ.get("LD_TAG", "") + ".json"), "w"), indent=1)
