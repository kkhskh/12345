"""Workstream A3 (XIII-B): causal CHAIN of the relational effect, path patching on relational pairs (class II natural activations).
Pairs: base x and donor x' differ only in one group's source pair at one question q' != q (values at q identical; Bayes change purely relational).
UPSTREAM intervention U: replace the outputs of ALL layer-0 heads and the layer-0 MLP at the six tokens of question q' by their donor values
(the 'source relationship' computed where the edit happened).  Everything downstream is recomputed, except components FROZEN at their base values:
  none | L1ATT@RO | L1ATT@QS | L1ATT@OS | L1MLP@RO | L1ATT@RO+L1MLP@RO | L2ATT@RO | L2MLP@RO | L2ATT@RO+L2MLP@RO | L1ATT@all-but-q'
Transfer = fidelity slope of the observed change (vs base) on the Bayes change.  The share of the transfer that flows THROUGH a component =
1 - slope(frozen)/slope(none).  Also: transfer when the edited question's layer-0 outputs are patched only at the edited GROUP's two tokens.
Usage: path_chain.py SPEC... (PC_STREAM default 75_200_000, PC_TAG) -> logs/path_chain<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, relpairs as RP
np.seterr(all="ignore")
rng = np.random.default_rng(int(os.environ.get("PC_STREAM", 75_200_000))); P_ = RP.relational_pairs(int(os.environ.get("PC_N", 800)), rng)
Xb, Xd, q, qp, hh = P_["Xb"], P_["Xd"], P_["q"], P_["qp"], P_["h"]; n = len(q); ii = np.arange(n); pred = RP.cen(P_["PBd"]) - RP.cen(P_["PB"])
T = 43
def mask(kind):
    m = np.zeros((n, T), bool)
    if kind == "RO": m[ii, 36 + q] = True
    elif kind == "QS": m[ii[:, None], q[:, None] * 6 + np.arange(6)[None]] = True
    elif kind == "QP": m[ii[:, None], qp[:, None] * 6 + np.arange(6)[None]] = True
    elif kind == "QPG": m[ii, qp * 6 + 2 * hh] = True; m[ii, qp * 6 + 2 * hh + 1] = True
    elif kind == "OS": m[:, :36] = True; m[ii[:, None], q[:, None] * 6 + np.arange(6)[None]] = False
    elif kind == "notQP": m[:] = True; m[ii[:, None], qp[:, None] * 6 + np.arange(6)[None]] = False
    return jnp.asarray(m)
out = dict(n_pairs=int(n), frac_e_change=float(P_["e_change"].mean()))
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    rB, rD = {}, {}; L0 = np.asarray(F.forward(M, jnp.asarray(Xb), rec=rB)); F.forward(M, jnp.asarray(Xd), rec=rD); P0 = RP.sm(L0[ii, q])
    LD = np.asarray(F.forward(M, jnp.asarray(Xd)))[ii, q]; full = RP.fid(RP.cen(RP.sm(LD)) - RP.cen(P0), pred)
    conds = {"none": [], "L1ATT@RO": [(1, "RO", "ATT")], "L1ATT@QS": [(1, "QS", "ATT")], "L1ATT@OS": [(1, "OS", "ATT")], "L1MLP@RO": [(1, "RO", "MLP")],
             "L1ATT+MLP@RO": [(1, "RO", "ATT"), (1, "RO", "MLP")], "L2ATT@RO": [(2, "RO", "ATT")], "L2MLP@RO": [(2, "RO", "MLP")],
             "L2ATT+MLP@RO": [(2, "RO", "ATT"), (2, "RO", "MLP")], "L1ATT@notQP": [(1, "notQP", "ATT")], "L1ATT+MLP@all": [(1, "notQP", "ATT"), (1, "notQP", "MLP")]}
    res = dict(full_donor_input=full)
    for upk in ("QP", "QPG"):
        up = mask(upk); base_slope = None
        for cn, fr in conds.items():
            hp = {(0, h): (up, rD["heads0"][:, h]) for h in range(4)}; mp = {0: (up, rD["mlp_out0"])}
            for (l, kind, comp) in fr:
                m = mask(kind)
                if comp == "ATT":
                    for h in range(4): hp[(l, h)] = (m, rB[f"heads{l}"][:, h])
                else: mp[l] = (m, rB[f"mlp_out{l}"])
            P = RP.sm(np.asarray(F.forward(M, jnp.asarray(Xb), head_patch=hp, mlp_patch=mp))[ii, q]); r = RP.fid(RP.cen(P) - RP.cen(P0), pred)
            if cn == "none": base_slope = r["slope"]
            r["share_through"] = (1 - r["slope"] / base_slope) if base_slope and abs(base_slope) > 1e-6 else None
            res[f"U{upk}|freeze {cn}"] = r
    out[sp] = res
    print(sp[:6], "full", round(full["slope"], 2), {k.replace("UQP|freeze ", ""): (round(v["slope"], 2), None if v["share_through"] is None else round(v["share_through"], 2)) for k, v in res.items() if k.startswith("UQP|")}, flush=True)
    print("      group-only U:", {k.replace("UQPG|freeze ", ""): round(v["slope"], 2) for k, v in res.items() if k.startswith("UQPG|") and k.split()[-1] in ("none", "L1ATT@RO", "L1ATT@QS", "L2ATT+MLP@RO")}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "path_chain" + os.environ.get("PC_TAG", "") + ".json"), "w"), indent=1)
