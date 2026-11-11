"""Stage III: exact forward-replay reconstruction of the layer-2 computation at the readout token t = 36+q for supported pairs
(F45 by regime, F54o, F34-R1, F0), development models.  Usage: layer2_reconstruction.py PHASE MODEL [...]   (PHASE dev|conf)
For each pair (before b, after a) the frozen input Z_b, Z_a (43x64) is computed.  Layer 2 at t is evaluated as an explicit function of
FIVE input copies, each set to its before- or after-value (32 combinations, all other computation recomputed exactly):
   D : the residual copy of z_t (enters the final sum and the MLP input),  Qy : the copy of z_t used to form the query,
   Kt/Ks : keys at (t) / at all other positions,  V : values at all positions.
Factorial (Moebius) coefficients of the oriented margin over these switches quantify main effects and interactions exactly
(the 00000 and 11111 corners are the natural before/after forwards).  Additional exact quantities:
   * MLP freezing: margin with the MLP output held at its before-value (attention+direct only) and with attention held (MLP+direct);
   * token-set attribution of the attention-output change at t, projected on the final-LN-linearised margin direction (midpoint
     Jacobian of LN_f; exact linear identity for the attention term only), split into re-weighting (dA v_b) and value change (A_a dv),
     for token sets: target-question tokens of the edited group g, of the other perfect group h*, of the third group f; other-question
     source tokens; readout tokens; episode token;
   * per-head one-at-a-time switches (add-to-before / remove-from-after) for Q, K, V.
Writes logs/recon_PHASE/<model>.json and per-pair arrays (.npz)."""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import nnlib as NL, r1lib as L
N = NL.N; B, E = L.B, L.E
PH, models = sys.argv[1], sys.argv[2:]
ev = dict(np.load(os.path.join(HERE, "..", "logs", f"eval_{'dev' if PH == 'dev' else 'conf'}.npz"), allow_pickle=True))
OUT = os.path.join(HERE, "..", "logs", f"recon_{PH}"); os.makedirs(OUT, exist_ok=True)
SW = ("D", "Qy", "Kt", "Ks", "V"); COMBOS = list(itertools.product((0, 1), repeat=5))
def head_terms(lay, zq, zk, zv, t):
    """attention at query position t: zq (n,64) query-input row; zk, zv (n,43,64) key/value inputs.  Returns per-head attention (n,H,43),
    per-head per-token value outputs after WO (n,H,43,64)."""
    lnq = N.ln(zq, lay["g1"], lay["b1"]); lnk = N.ln(zk, lay["g1"], lay["b1"]); lnv = N.ln(zv, lay["g1"], lay["b1"])
    q = jnp.einsum("nd,hde->nhe", lnq, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", lnk, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", lnv, lay["WV"])
    A = jax.nn.softmax(jnp.einsum("nhe,nhse->nhs", q, k) / jnp.sqrt(q.shape[-1]), -1)
    vo = jnp.einsum("nhse,hed->nhsd", v, lay["WO"]); return A, vo
def readout(p, x):  # x (n,64) final residual at t -> logits
    return N.ln(x, p["gf"], p["bf"]) @ p["WA"]
def layer2_at_t(p, zb, za, t, s, freeze=None, mlp_b=None, attn_b=None, head_sw=None):
    """s: dict switch -> 0/1.  zb, za: (n,43,64).  head_sw: optional per-head overrides {(h, 'Q'|'K'|'V'): 0/1}."""
    lay = p["layers"][2]; pick = lambda k: za if s[k] else zb
    n = zb.shape[0]; ar = jnp.arange(n)
    zt_d = pick("D")[ar, t]; zt_q = pick("Qy")[ar, t]
    zk = jnp.where(jnp.arange(43)[None, :, None] == t[:, None, None], pick("Kt"), pick("Ks")); zv = pick("V")
    if head_sw is None:
        A, vo = head_terms(lay, zt_q, zk, zv, t)
    else:
        As, vos = [], []
        for h in range(4):
            qq = za[ar, t] if head_sw.get((h, "Q"), s["Qy"]) else zb[ar, t]
            kk = za if head_sw.get((h, "K"), s["Ks"]) else zb; vv = za if head_sw.get((h, "V"), s["V"]) else zb
            Ah, voh = head_terms(lay, qq, kk, vv, t); As.append(Ah[:, h]); vos.append(voh[:, h])
        A, vo = jnp.stack(As, 1), jnp.stack(vos, 1)
    o = jnp.einsum("nhs,nhsd->nd", A, vo)
    if freeze == "attn": o = attn_b
    x = zt_d + o
    m = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
    if freeze == "mlp": m = mlp_b
    return readout(p, x + m), o, m, A, vo
out_all = {}
for mname in models:
    p = NL.load(NL.dev_ckpt(mname) if PH == "dev" else mname); fz = jax.jit(lambda X: NL.frozen_z(p, X)); res = {}
    pairs = {}
    for f in ("F45", "F34", "F0"):
        sel = ev[f"orient_{f}"]
        for rg in (("R1", "R2", "R3") if f == "F45" else ("R1",)):
            ss = sel & (ev[f"reg_{f}"] == rg); pairs[f"{f}_{rg}"] = (ev["X"][ss], ev[f"X_{f}"][ss], ev["q"][ss], ev[f"vote_{f}"][ss], ss)
    pairs["F54o"] = (ev["X_O"], ev["X_F54o"], ev["q_O"], ev["vote_F54o"], None)
    arrays = {}
    for pn, (Xb, Xa, q, v, ss) in pairs.items():
        zb, za = fz(jnp.asarray(Xb)), fz(jnp.asarray(Xa)); t = jnp.asarray(36 + q); n = len(q)
        marg = lambda lg: B.oriented_margin(np.asarray(lg) - np.asarray(lg).mean(-1, keepdims=True), v)
        M = {}
        for c in COMBOS:
            lg, o, m, A, vo = layer2_at_t(p, zb, za, t, dict(zip(SW, c))); M[c] = marg(lg)
            if c == (0,) * 5: ob, mb, Ab, vob = o, m, A, vo
            if c == (1,) * 5: oa, ma, Aa, voa = o, m, A, vo
        # natural forward check
        full_b = np.asarray(N.forward(p, jnp.asarray(Xb))["answer"])[np.arange(n), q]; full_a = np.asarray(N.forward(p, jnp.asarray(Xa))["answer"])[np.arange(n), q]
        chk = float(max(np.abs(marg(full_b) - M[(0,) * 5]).max(), np.abs(marg(full_a) - M[(1,) * 5]).max()))
        total = M[(1,) * 5] - M[(0,) * 5]
        # Moebius coefficients: effect of subset S = sum_{T subset S} (-1)^{|S|-|T|} M[T]
        mob = {}
        for c in COMBOS:
            S = [i for i in range(5) if c[i]]; val = 0
            for r_ in range(len(S) + 1):
                for T in itertools.combinations(S, r_):
                    cc = tuple(1 if i in T else 0 for i in range(5)); val = val + (-1) ** (len(S) - r_) * M[cc]
            mob["".join(SW[i] + "+" for i in S).rstrip("+") or "none"] = float(np.mean(val))
        # freezing
        all1 = dict(zip(SW, (1,) * 5))
        fr_mlp = marg(layer2_at_t(p, zb, za, t, all1, freeze="mlp", mlp_b=mb)[0]) - M[(0,) * 5]
        fr_att = marg(layer2_at_t(p, zb, za, t, all1, freeze="attn", attn_b=ob)[0]) - M[(0,) * 5]
        # token-set attribution of attention change, projected on LN_f-linearised margin direction at the midpoint
        xb = zb[jnp.arange(n), t] + ob + mb; xa = za[jnp.arange(n), t] + oa + ma
        def mfun(x, vv): lg = readout(p, x); return lg[vv] - (lg.sum() - lg[vv]) / 2
        J = jax.vmap(jax.grad(mfun))(0.5 * (xb + xa), jnp.asarray(v))           # (n,64)
        g_ = ev["X"][:0]  # placeholder
        # groups of tokens
        sets = {}
        Xb_np = np.asarray(Xb)
        if pn.startswith("F54o"): gsel = ev["drop_F54o"]; hsel = ev["keep_F54o"]
        else: gsel = None
        ge = np.full(n, -1); he = np.full(n, -1)
        if not pn.startswith("F54o"):
            # identify edited group g and other perfect group h* from the before/after counts
            nb = E.loo(Xb_np)[np.arange(n), q]; na = E.loo(np.asarray(Xa))[np.arange(n), q]; dg = (na != nb)
            ge = np.where(dg.any(1), dg.argmax(1), -1)
            for i in range(n):
                hs = [h for h in range(3) if h != ge[i] and nb[i, h] == 5]; he[i] = hs[0] if hs else -1
        else:
            ge, he = gsel, hsel
        fe = 3 - ge - he
        tokmask = {}
        for nm_, grp in (("q_tokens_edited_group", ge), ("q_tokens_other_perfect", he), ("q_tokens_third_group", fe)):
            mk = np.zeros((n, 43), bool)
            for i in range(n):
                if grp[i] < 0 or grp[i] > 2: continue
                a_, b_ = L.GROUPS[grp[i]]; mk[i, q[i] * 6 + a_] = mk[i, q[i] * 6 + b_] = True
            tokmask[nm_] = mk
        mk = np.zeros((n, 43), bool); mk[:, :36] = True
        for i in range(n): mk[i, q[i] * 6:(q[i] + 1) * 6] = False
        tokmask["other_question_sources"] = mk; mk = np.zeros((n, 43), bool); mk[:, 36:42] = True; tokmask["readout_tokens"] = mk
        mk = np.zeros((n, 43), bool); mk[:, 42] = True; tokmask["episode_token"] = mk
        Ab_, Aa_, vb_, va_ = np.asarray(Ab), np.asarray(Aa), np.asarray(vob), np.asarray(voa); Jn = np.asarray(J)
        proj_b = np.einsum("nhsd,nd->nhs", vb_, Jn); proj_a = np.einsum("nhsd,nd->nhs", va_, Jn)
        reweight = ((Aa_ - Ab_) * proj_b).sum(1); valchg = (Aa_ * (proj_a - proj_b)).sum(1)        # (n,43)
        attr = {k: dict(reweight=float((reweight * mk).sum(1).mean()), value=float((valchg * mk).sum(1).mean()),
                        attn_mass_before=float((Ab_.sum(1) / 4 * mk).sum(1).mean()), attn_mass_after=float((Aa_.sum(1) / 4 * mk).sum(1).mean()))
                for k, mk in tokmask.items()}
        lin_attn = float((reweight.sum(1) + valchg.sum(1)).mean()); exact_attn_proj = float(np.einsum("nd,nd->n", np.asarray(oa - ob), Jn).mean())
        # per-head one-at-a-time
        heads = {}
        for h in range(4):
            for comp in ("Q", "K", "V"):
                s0 = dict(zip(SW, (0,) * 5)); add = marg(layer2_at_t(p, zb, za, t, s0, head_sw={(h, comp): 1})[0]) - M[(0,) * 5]
                rem = M[(1,) * 5] - marg(layer2_at_t(p, zb, za, t, all1, head_sw={(h, comp): 0})[0])
                heads[f"h{h}_{comp}"] = dict(add_to_before=float(add.mean()), remove_from_after=float(rem.mean()))
        res[pn] = dict(n=n, natural_check_max_abs=chk, total=float(total.mean()), moebius=mob, freeze_mlp=float(fr_mlp.mean()), freeze_attn=float(fr_att.mean()),
                       direct_only=float((M[(1, 0, 0, 0, 0)] - M[(0,) * 5]).mean()), attn_token_attribution=attr, attn_linearised_total=lin_attn,
                       attn_exact_projection=exact_attn_proj, heads=heads)
        arrays[pn + "_total"] = total; arrays[pn + "_reweight"] = reweight; arrays[pn + "_value"] = valchg
    out_all[mname] = res; json.dump(res, open(os.path.join(OUT, f"{mname.replace('/', '_')[-40:]}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{mname.replace('/', '_')[-40:]}.npz"), **arrays)
    r = res["F45_R1"]; mo = r["moebius"]
    print(mname, "R1 total %+.3f chk %.1e | direct %+.3f fr_mlp(attn+dir) %+.3f fr_attn(mlp+dir) %+.3f | Moebius D %+.3f Qy %+.3f Kt %+.3f Ks %+.3f V %+.3f Ks+V %+.3f | third-grp reweight %+.3f value %+.3f | other-perf rw %+.3f val %+.3f | edited rw %+.3f val %+.3f | F54o total %+.3f" % (
        r["total"], r["natural_check_max_abs"], r["direct_only"], r["freeze_mlp"], r["freeze_attn"], mo["D"], mo["Qy"], mo["Kt"], mo["Ks"], mo["V"], mo["Ks+V"],
        r["attn_token_attribution"]["q_tokens_third_group"]["reweight"], r["attn_token_attribution"]["q_tokens_third_group"]["value"],
        r["attn_token_attribution"]["q_tokens_other_perfect"]["reweight"], r["attn_token_attribution"]["q_tokens_other_perfect"]["value"],
        r["attn_token_attribution"]["q_tokens_edited_group"]["reweight"], r["attn_token_attribution"]["q_tokens_edited_group"]["value"], res["F54o"]["total"]), flush=True)
