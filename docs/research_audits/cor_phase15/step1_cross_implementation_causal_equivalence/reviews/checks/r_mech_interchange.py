"""Reviewer check (mechanistic): within-model interchange variants on the conf streams with equiv_run.py's exact family RNG
(eval 12_400_000 x1000, rng(EVAL+100+q): cond_pairs then single_group_edit).
  (ceiling_site+qsrc: P site plus question q's six source tokens; equals ceiling_site for T)
  cond:  enc_orig (must reproduce results.json), enc_partial (encoder fitted with own-question covariates, count block
         only; see r_mech_marg_partial.py), floor_mismatched, ceiling = full-site patch, ceiling_site+readout (T: q's six
         source tokens AND readout 36+q patched), ceiling_all (every position patched: must be IIA 1).
  single (T only): pair tokens, one token of the pair, one token with 2x delta (dose-matched), encoder on all six tokens.
  pathway (T only): random-direction delta at q's source tokens (norm = encoder delta), readout layer-2 attention frozen:
         remaining fraction (architectural zero test).
Usage: r_mech_interchange.py M ..."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np
import eqv as E
from r_mech_marg_partial_lib import fit_partial
t0 = time.time(); FIT, EVAL, NF, NE = 12_300_000, 12_400_000, 4000, 1000
Xf = E.sample(NF, FIT)["X"]; De = E.sample(NE, EVAL); X, cfg = De["X"], De["cfg"]; n0 = E.loo(X)
fam = {}
for q in range(E.Q):
    r = np.random.default_rng(EVAL + 100 + q)
    fam[("cond", q)] = (E.cond_pairs(X, q, r), np.ones(NE, bool), None)
    Xs, gs, ok, _ = E.single_group_edit(X, q, r, cfg); fam[("single", q)] = (Xs, ok, gs)
sh = np.roll(np.arange(NE), 1)
for m in sys.argv[1:]:
    M = E.Model(m); k = m[0]; d = M.d; prim = "P_full" if k == "P" else "T_own"
    Af = M.acts(Xf)[0]; A0, L0, attn0, mlp0 = M.acts(X)
    enc, encp = E.fit_encoder(prim, Xf, Af), fit_partial(prim, Xf, Af); res = {}
    for f in ("cond",) + (("single",) if k == "T" else ()):
        agg = {}
        for q in range(E.Q):
            Xp, ok, gs = fam[(f, q)]; n1 = E.loo(Xp); Ap, Lp, _, _ = M.acts(Xp); pos = E.site_positions(k, q)
            C = {"enc_orig": E.encoder_delta(enc, prim, X, n0, n1, d, pos), "enc_partial": E.encoder_delta(encp, prim, X, n0, n1, d, pos),
                 "floor_mismatched": E.encoder_delta(enc, prim, X, n0, n1[sh], d, pos)}
            for nm, pp in (("ceiling_site", pos), ("ceiling_site+readout", sorted(set(pos + [36 + q]))), ("ceiling_site+qsrc", sorted(set(pos + list(range(6 * q, 6 * q + 6))))), ("ceiling_all", list(range(E.T)))):
                dl = np.zeros_like(A0); dl[:, pp] = Ap[:, pp] - A0[:, pp]; C[nm] = dl
            if f == "single":
                full = C["enc_orig"]; pair = np.zeros_like(A0); one = np.zeros_like(A0); one2 = np.zeros_like(A0)
                for i in np.where(ok)[0]:
                    tt = [6 * q + 2 * gs[i], 6 * q + 2 * gs[i] + 1]
                    pair[i, tt] = full[i, tt]; one[i, tt[0]] = full[i, tt[0]]; one2[i, tt[0]] = 2 * full[i, tt[0]]
                C.update(pair_tokens=pair, one_token=one, one_token_x2=one2)
            for nm, dl in C.items():
                agg.setdefault(nm, []).append((L0[ok, q], Lp[ok, q], M.run(X, dl)[ok, q]))
            if f == "cond" and k == "T":
                rng = np.random.default_rng(5 + q); dl = C["enc_orig"]; rd = np.zeros_like(dl)
                rd[:, pos] = rng.normal(size=dl[:, pos].shape); rd[:, pos] *= (np.linalg.norm(dl[:, pos], axis=-1, keepdims=True) / np.linalg.norm(rd[:, pos], axis=-1, keepdims=True))
                ma = np.zeros(E.T, np.float32); ma[36 + q] = 1; z = np.zeros(E.T, np.float32)
                full_ = np.abs(E.clogit(M.run(X, rd)[:, q]) - E.clogit(L0[:, q])).sum(-1)
                fr = np.abs(E.clogit(M.run_frozen(X, rd, attn0, mlp0, ma, z)[:, q]) - E.clogit(L0[:, q])).sum(-1)
                agg.setdefault("_pw", []).append((full_.mean(), fr.max()))
        pw = agg.pop("_pw", None)
        sc = {nm: E.score(*(np.concatenate([x[i] for x in v]) for i in range(3))) for nm, v in agg.items()}
        o = {nm: dict(iia=round(s["iia"], 4), r=round(s["r_dlogit"], 3), mabs=round(s["mean_abs_dlogit_patch"], 3)) for nm, s in sc.items()}
        fl, ce = sc["floor_mismatched"]["iia"], sc["ceiling_site"]["iia"]
        for nm in ("enc_orig", "enc_partial") + (("pair_tokens", "one_token", "one_token_x2") if f == "single" else ()):
            o[nm]["adj_site_ceiling"] = round(E.adjusted(sc[nm]["iia"], fl, ce), 3)
            o[nm]["adj_site+readout_ceiling"] = round(E.adjusted(sc[nm]["iia"], fl, sc["ceiling_site+readout"]["iia"]), 3)
        if pw: o["pathway_random_dir"] = dict(mean_full_effect=float(np.mean([a for a, _ in pw])), max_abs_effect_attn_frozen=float(max(b for _, b in pw)))
        res[f] = o
    print(m, json.dumps(res), f"[{time.time() - t0:.0f}s]", flush=True)
