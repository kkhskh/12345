"""Reviewer check: sampling stability of route_Q / route_R (class_features.routes, n=800 relational pairs) across streams."""
import os, sys, json, time
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore"); t0 = time.time()
def routes(M, P_):
    m = len(P_["q"]); i = np.arange(m); q_, qp, h = P_["q"], P_["qp"], P_["h"]; rB, rD = {}, {}
    L0 = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), rec=rB))[i, q_]; F.forward(M, jnp.asarray(P_["Xd"]), rec=rD); pred = RP.cen(P_["PBd"]) - RP.cen(P_["PB"])
    up = np.zeros((m, 43), bool); up[i, qp * 6 + 2 * h] = True; up[i, qp * 6 + 2 * h + 1] = True; up = jnp.asarray(up)
    def mk(kind):
        mm = np.zeros((m, 43), bool)
        if kind == "QS": mm[i[:, None], q_[:, None] * 6 + np.arange(6)[None]] = True
        else: mm[i, 36 + q_] = True
        return jnp.asarray(mm)
    def tr(freeze):
        hp = {(0, hd): (up, rD["heads0"][:, hd]) for hd in range(4)}; mp = {0: (up, rD["mlp_out0"])}
        for fz in freeze:
            for hd in range(4): hp[(1, hd)] = (mk(fz), rB["heads1"][:, hd])
        L = np.asarray(F.forward(M, jnp.asarray(P_["Xb"]), head_patch=hp, mlp_patch=mp))[i, q_]; return RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)), pred)["slope"]
    s0 = tr([]); return dict(transfer=s0, route_Q=1 - tr(["QS"]) / s0, route_R=1 - tr(["RO"]) / s0)
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); out[sp] = {}
    for st in (79_000_000, 81_000_001, 81_000_002):
        rng = np.random.default_rng(st); PR = RP.relational_pairs(800, rng); r = routes(M, PR)
        # bootstrap over pairs for the canonical stream: resample indices
        out[sp][st] = r; print(sp[:22], st, {k: round(v, 3) for k, v in r.items()}, round(time.time() - t0), "s", flush=True)
json.dump(out, open("/tmp/review13b_mech/route_stab.json", "w"), indent=1, default=str)
