"""[v2: RELATIONAL families, values at q identical between base and donor: EA = (xA -> xAP: e_h and pf_h both 0->1), P0 = (x0 -> xP: pf_h 0->1 while e_h stays 0,
the E-vs-A1 discriminating family), each with a _ctrl family (same edit, group stays imperfect).  Below, "xAP" = base cell, "xP" = donor cell.]
Workstreams B1 / B3 / B5 (XIII-B) on the XIII TRUE / CTRL factorial (cells xAP: group h eligible at q; xP: h perfect but NOT agreeing at q,
so e_h differs while the perfect-only variable A1 is the SAME).  Base run = xAP, donor run = xP (natural activations), episodes from stream
L2C_STREAM (default 75_400_000).  A custom layer-2 forward lets each layer-2 ingredient at the readout token be taken from the donor separately.
TE = cen(P_xP) - cen(P_xAP).  For an intervention, NIE = cen(P_patched) - cen(P_xAP); reported: proj = NIE.TE / TE.TE, fidelity to the
E-mixture prediction (Bayes responsibilities of xP with xAP likelihoods) and to the A1-mixture prediction.
B5 decomposition: components R (mixture-head attention rows), V (mixture-head VALUE inputs at the six q tokens), O (outputs of the other
layer-2 heads at RO), M (MLP2 output at RO), K (resid2 skip path at RO).  All 32 subsets -> Shapley values and pairwise interactions of proj.
B1 decoys for R (same transplant procedure): LOW (lowest q-mass head, XIII's decoy), TOP (non-mixture layer-2 head with the largest
direct-logit share), MAG (TOP head with its row change rescaled to the mixture heads' mean L1 row change), PERM (the mixture heads' donor rows with
the three groups' q-token masses cyclically permuted: identical mass, entropy, positions and change magnitude, wrong group assignment),
MEAN (source-free: mean donor row over episodes), NOOP (own rows).
B3 restoration: REMOVAL = run xP with the layer-1 attention output at the six q tokens replaced by its xAP value (route Q, which carried 65-70%
of relational transfer in path_chain.py).  RESTORE = in that removed run, insert the INTACT xP mixture-head rows; decoy restorations with LOW, TOP,
MAG, PERM, MEAN rows; plus a candidate-likelihood edit (another group's value at q changed) to check preservation of unrelated effects under
removal + restoration.  Single head rule: relpairs.mixture_heads (mass >= 0.8 and share >= 0.2); mass-only heads are reported and also run.
Usage: l2_components.py SPEC... -> logs/l2_components<TAG>.json"""
import json, os, sys, itertools, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
EVS = int(os.environ.get("L2C_STREAM", 75_400_000)); _r = np.random.default_rng(EVS); X0 = NV.Bm.sample_fast(NV.ENV, 8000, _r)["X"]; q0 = _r.integers(0, 6, 8000)
n0 = len(X0); C = GL.counts(X0, 3); a = X0.reshape(n0, 6, 3, 2); eq = a[..., 0] == a[..., 1]; env = GL.Env(3, 1)
def design(base_n, seed=1304, nmax=500):
    rng = np.random.default_rng(seed + base_n); cells = {k: [] for k in ("xP", "xAP", "xPL", "xAPL")}; qq = []; hs_ = []
    for i in range(n0):
        hs = [h for h in range(3) if not eq[i, q0[i], h] and C[i, q0[i], h] == base_n]
        if not hs: continue
        h = hs[rng.integers(len(hs))]; others = [qp for qp in range(6) if qp != q0[i] and not eq[i, qp, h]]; qp = others[rng.integers(len(others))]
        x0 = X0[i].copy(); xA = x0.copy(); xA[q0[i], 2 * h + 1] = xA[q0[i], 2 * h]; xP = x0.copy(); xP[qp, 2 * h + 1] = xP[qp, 2 * h]; xAP = xA.copy(); xAP[qp, 2 * h + 1] = xAP[qp, 2 * h]
        b, dn = (xA, xAP) if FAMKIND == "EA" else (x0, xP)                 # EA: e and pf both flip; P0: pf flips, e stays 0 (E vs A1 discriminating)
        g = [gg for gg in range(3) if gg != h][rng.integers(2)]; s = 2 * g + rng.integers(2)       # likelihood edit: another group's source at q
        bL, dL = b.copy(), dn.copy(); nv = (b[q0[i], s] + rng.integers(1, 3)) % 3; bL[q0[i], s] = nv; dL[q0[i], s] = nv
        for k, x in zip(cells, (dn, b, dL, bL)): cells[k].append(x)
        qq.append(q0[i]); hs_.append(h)
        if len(qq) >= nmax: break
    return {k: np.array(v) for k, v in cells.items()}, np.array(qq)
D = {}
for FAMKIND in ("EA", "P0"):
    D[FAMKIND] = design(4); D[FAMKIND + "_ctrl"] = design(3)          # ctrl: the same edit leaves the group imperfect (n_h = 4): no eligibility/perfectness change
def preds(cl, qq):
    out = {}
    for nm, which in (("E", "e"), ("A1", "pf")):
        Wp, Lp, pfp, ep, Sp = RP.parts(cl["xP"], qq); Wa, La, pfa, ea, Sa = RP.parts(cl["xAP"], qq)
        fp = (Sp - ep) if which == "e" else (pfp.sum(1, keepdims=True) - pfp); fa = (Sa - ea) if which == "e" else (pfa.sum(1, keepdims=True) - pfa)
        rp = RP.resp(Wp, Lp, fp); ra = RP.resp(Wa, La, fa); Ln = La / La.sum(-1, keepdims=True)
        mix = lambda r: (lambda P: P / np.maximum(P.sum(-1, keepdims=True), 1e-300))((r[..., None] * Ln).sum(1))
        out[nm] = RP.cen(mix(rp)) - RP.cen(mix(ra))
    return out
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); lay = M["layers"][2]
    # single head rule (natural episodes of the same stream)
    mass, share = RP.head_stats(M, X0[:2000], q0[:2000], F, jnp); MH, MH_mass = RP.mixture_heads(mass, share)
    res = dict(mass=mass.round(3).tolist(), share=np.round(share, 3).tolist(), mixture_heads=MH, mixture_heads_mass_only=MH_mass)
    heads_used = MH if MH else MH_mass
    if not heads_used: out[sp] = res; print(sp[:6], "no mixture head; mass", mass.round(2), "share", np.round(share, 2)); continue
    non = [h for h in range(4) if h not in heads_used]; LOW = [int(np.argmin(mass))]; TOP = [max(non, key=lambda h: share[h])]
    for nm, (cl, qq) in D.items():
        N_ = len(qq); ii = np.arange(N_); ro = 36 + qq; qs = qq[:, None] * 6 + np.arange(6)[None]
        rec = {}
        for k in cl: r = {}; F.forward(M, jnp.asarray(cl[k]), rec=r); rec[k] = {kk: np.asarray(v) for kk, v in r.items() if kk in ("resid2", "pattern2", "heads2", "mlp_out2", "heads1")}
        def layer2(x2, xv=None, vheads=(), rows=None, hout=None, mlp=None, skip=None):
            xb = jnp.asarray(x2); z = F.N.ln(xb, lay["g1"], lay["b1"])
            qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
            if xv is not None and vheads:
                zv = F.N.ln(jnp.asarray(xv), lay["g1"], lay["b1"]); vv = jnp.einsum("ntd,hde->nhte", zv, lay["WV"])
                mq = np.zeros((N_, 43), bool); mq[ii[:, None], qs] = True; mq = jnp.asarray(mq)
                for h in vheads: vh = vh.at[:, h].set(jnp.where(mq[..., None], vv[:, h], vh[:, h]))
            A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1); t = jnp.asarray(ro); nn = jnp.arange(N_)
            for h, rw in (rows or {}).items(): A = A.at[nn, h, t].set(jnp.asarray(rw))
            oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"])
            for h, ho in (hout or {}).items(): oh = oh.at[nn, h, t].set(jnp.asarray(ho))
            xs = xb if skip is None else xb.at[nn, t].set(jnp.asarray(skip))
            xa = xs + oh.sum(1); hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); mo = hid @ lay["W2"] + lay["c2"]
            if mlp is not None: mo = mo.at[nn, t].set(jnp.asarray(mlp))
            x3 = xa + mo; Lx = np.asarray(F.N.ln(x3, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[ii, qq]; return RP.sm(Lx)
        rowsD = {h: rec["xP"]["pattern2"][ii, h, ro] for h in range(4)}; rowsA = {h: rec["xAP"]["pattern2"][ii, h, ro] for h in range(4)}
        P_ap = layer2(rec["xAP"]["resid2"]); P_xp = layer2(rec["xP"]["resid2"]); TE = RP.cen(P_xp) - RP.cen(P_ap); PR = preds(cl, qq)
        def score(P, base=P_ap, te=TE):
            nie = RP.cen(P) - RP.cen(base); return dict(proj=float((nie * te).sum() / (te * te).sum()), mag=float(np.abs(nie).mean()), fid_E=RP.fid(nie, PR["E"]), fid_A1=RP.fid(nie, PR["A1"]))
        r = dict(n=int(N_), TE_mag=float(np.abs(TE).mean()), pred_E_mag=float(np.abs(PR["E"]).mean()), pred_A1_mag=float(np.abs(PR["A1"]).mean()), noop=score(layer2(rec["xAP"]["resid2"], rows=rowsA)))
        # ---- B5 decomposition (subsets of R, V, O, M, K)
        comps = ("R", "V", "O", "M", "K"); vals = {}
        for k in range(6):
            for sub in itertools.combinations(comps, k):
                kw = {}
                if "R" in sub: kw["rows"] = {h: rowsD[h] for h in heads_used}
                if "V" in sub: kw["xv"] = rec["xP"]["resid2"]; kw["vheads"] = heads_used
                if "O" in sub: kw["hout"] = {h: rec["xP"]["heads2"][ii, h, ro] for h in non}
                if "M" in sub: kw["mlp"] = rec["xP"]["mlp_out2"][ii, ro]
                if "K" in sub: kw["skip"] = rec["xP"]["resid2"][ii, ro]
                vals["".join(sub)] = score(layer2(rec["xAP"]["resid2"], **kw))["proj"]
        sh = {}
        for c in comps:
            tot = 0.0; others = [o for o in comps if o != c]
            for k in range(5):
                for sub in itertools.combinations(others, k):
                    wgt = math.factorial(k) * math.factorial(4 - k) / math.factorial(5)
                    key = lambda s: "".join(o for o in comps if o in s)
                    tot += wgt * (vals[key(set(sub) | {c})] - vals[key(set(sub))])
            sh[c] = tot
        inter = {f"{x}{y}": vals["".join(o for o in comps if o in (x, y))] - vals[x] - vals[y] + vals[""] for x, y in itertools.combinations(comps, 2)}
        r.update(subset_proj=vals, shapley=sh, pairwise_interaction=inter)
        # ---- B1 decoys
        l1 = lambda rw, h: np.abs(rw - rowsA[h]).sum(1).mean()
        mixchg = float(np.mean([l1(rowsD[h], h) for h in heads_used]))
        def perm_rows(h):
            rw = rowsD[h].copy(); blk = np.stack([rw[ii[:, None], qs[:, 2 * g:2 * g + 2]] for g in range(3)], 1)
            for g in range(3): rw[ii[:, None], qs[:, 2 * g:2 * g + 2]] = blk[:, (g + 1) % 3]
            return rw
        def mag_rows(h):
            dlt = rowsD[h] - rowsA[h]; lam = mixchg / max(l1(rowsD[h], h), 1e-9); rw = np.clip(rowsA[h] + lam * dlt, 0, None); return rw / rw.sum(1, keepdims=True), lam
        meanrow = {h: np.broadcast_to(rowsD[h].mean(0, keepdims=True), rowsD[h].shape) for h in range(4)}
        dec = {"MIX": {h: rowsD[h] for h in heads_used}, "LOW": {h: rowsD[h] for h in LOW}, "TOP": {h: rowsD[h] for h in TOP},
               "MAG": {TOP[0]: mag_rows(TOP[0])[0]}, "PERM": {h: perm_rows(h) for h in heads_used}, "MEAN": {h: meanrow[h] for h in heads_used}}
        r["decoys"] = {k: dict(score(layer2(rec["xAP"]["resid2"], rows=v)), row_change_L1=float(np.mean([l1(rw, h) for h, rw in v.items()]))) for k, v in dec.items()}
        r["decoy_meta"] = dict(LOW=LOW, TOP=TOP, TOP_share=float(share[TOP[0]]), MAG_lambda=float(mag_rows(TOP[0])[1]), mixture_row_change_L1=mixchg)
        # ---- B3 removal + restoration (route Q at layer 1 removed: L1 attention output at q tokens from xAP)
        mq = np.zeros((N_, 43), bool); mq[ii[:, None], qs] = True; mq = jnp.asarray(mq)
        def removed(src, ref):
            rr = {}; F.forward(M, jnp.asarray(cl[src]), head_patch={(1, h): (mq, jnp.asarray(rec[ref]["heads1"][:, h])) for h in range(4)}, rec=rr); return np.asarray(rr["resid2"]), np.asarray(rr["pattern2"])
        x2R, A2R = removed("xP", "xAP"); P_rem = layer2(x2R)
        rest = {"REMOVED_only": layer2(x2R), "RESTORE_MIX": layer2(x2R, rows={h: rowsD[h] for h in heads_used}), "RESTORE_LOW": layer2(x2R, rows={h: rowsD[h] for h in LOW}),
                "RESTORE_TOP": layer2(x2R, rows={h: rowsD[h] for h in TOP}), "RESTORE_MAG": layer2(x2R, rows={TOP[0]: mag_rows(TOP[0])[0]}),
                "RESTORE_PERM": layer2(x2R, rows={h: perm_rows(h) for h in heads_used}), "RESTORE_MEAN": layer2(x2R, rows={h: meanrow[h] for h in heads_used})}
        r["restoration"] = {k: score(P) for k, P in rest.items()}
        r["restoration"]["removal_fraction_removed"] = 1 - r["restoration"]["REMOVED_only"]["proj"]
        # preservation of an unrelated (likelihood) effect: the q-value edit's effect, intact vs removed+restored
        x2RL, _ = removed("xPL", "xAPL"); PL_int = layer2(rec["xPL"]["resid2"]); PL_rr = layer2(x2RL, rows={h: rec["xPL"]["pattern2"][ii, h, ro] for h in heads_used})
        d_int = RP.cen(PL_int) - RP.cen(P_xp); d_rr = RP.cen(PL_rr) - RP.cen(rest["RESTORE_MIX"])
        r["preservation_likelihood_edit"] = dict(intact_mag=float(np.abs(d_int).mean()), fid_rr_to_intact=RP.fid(d_rr, d_int))
        res[nm] = r
    res["heads_used"] = heads_used; out[sp] = res
    T = res["EA"]; Cc = res["EA_ctrl"]
    print(sp[:6], "MH", MH, "mass-only", MH_mass, "TE", round(T["TE_mag"], 2), "shapley", {k: round(v, 2) for k, v in T["shapley"].items()}, "all", round(T["subset_proj"]["RVOMK"], 2), flush=True)
    print("      decoys TRUE proj", {k: (round(v["proj"], 2), round(v["row_change_L1"], 2)) for k, v in T["decoys"].items()}, "CTRL", {k: round(v["proj"], 2) for k, v in Cc["decoys"].items()}, flush=True)
    P0r = res["P0"]; print("      P0 (E vs A1 discriminating): TE", round(P0r["TE_mag"], 2), "predE", round(P0r["pred_E_mag"], 2), "predA1", round(P0r["pred_A1_mag"], 2), "MIX rows fidE/A1", round(P0r["decoys"]["MIX"]["fid_E"]["R2"], 2), round(P0r["decoys"]["MIX"]["fid_A1"]["R2"], 2), "restore MIX fidE/A1", round(P0r["restoration"]["RESTORE_MIX"]["fid_E"]["R2"], 2), round(P0r["restoration"]["RESTORE_MIX"]["fid_A1"]["R2"], 2), flush=True)
    print("      restore EA", {k: (round(v["proj"], 2), round(v["fid_E"]["R2"], 2), round(v["fid_A1"]["R2"], 2)) for k, v in T["restoration"].items() if isinstance(v, dict)}, "removed frac", round(T["restoration"]["removal_fraction_removed"], 2), "preserve", round(T["preservation_likelihood_edit"]["fid_rr_to_intact"]["slope"], 2), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "l2_components" + os.environ.get("L2C_TAG", "") + ".json"), "w"), indent=1, default=float)
