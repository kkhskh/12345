"""XIII-C Workstream D5: do genuinely PARALLEL causal routes carry the relational effect, or only one serial chain?
Purely relational pairs (edit of group h at q' != q).  The donor information is injected ONLY at its origin: layer-0 outputs (all heads + MLP0) at
the edited group's two tokens at q' are taken from the donor run (everything else is the base run).  Three candidate downstream routes:
  Q : layer-1 attention output written into the 6 source tokens of q          (-> read by layer-2 rows/values at q tokens)
  R : layer-1 attention output written into the readout token of q             (-> read at the readout token)
  O : the remaining tokens (edited and other non-q source tokens, incl. layer-1 writes there) read by layer 2: cut by freezing layer-1 attention
      output at all non-q source tokens AND restoring the edited tokens' layer-0 output to base before layer 2 reads them (resid2 at those tokens := base)
Every subset of {Q, R, O} is CUT (frozen to base) -> transfer slope on the Bayes change (s_cut).  Reported:
  SUFFICIENCY of a route = transfer with ONLY that route open; NECESSITY = drop when only that route is cut;
  REDUNDANCY = sum of single-route sufficiencies - transfer with all open (> 0: overlapping/parallel carriers; < 0: synergy);
  "restoration" in the D5 sense: with route Q cut, does opening R alone restore the effect (= sufficiency of R) -- a distinct route, not a
  downstream re-injection on the same path.
Usage: DT_TAG=_dev python3 scripts/d_routes.py SPEC... -> logs/d_routes<TAG>.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
PR = RP.relational_pairs(700, np.random.default_rng(int(os.environ.get("DT_STREAM", 87_200_000)))); n = len(PR["q"]); i = np.arange(n); q_, qp, h = PR["q"], PR["qp"], PR["h"]
pred = RP.cen(PR["PBd"]) - RP.cen(PR["PB"])
up = np.zeros((n, 43), bool); up[i, qp * 6 + 2 * h] = True; up[i, qp * 6 + 2 * h + 1] = True
mQ = np.zeros((n, 43), bool); mQ[i[:, None], q_[:, None] * 6 + np.arange(6)[None]] = True
mR = np.zeros((n, 43), bool); mR[i, 36 + q_] = True
mO = np.zeros((n, 43), bool); mO[:, :36] = True; mO &= ~mQ
out = {}
for sp in sys.argv[1:]:
    M = xc.load(sp); rB, rD = {}, {}; L0 = np.asarray(F.forward(M, jnp.asarray(PR["Xb"]), rec=rB))[i, q_]; F.forward(M, jnp.asarray(PR["Xd"]), rec=rD)
    z2b = jnp.asarray(rB["resid2"])
    def transfer(cut):
        hp = {(0, hd): (jnp.asarray(up), rD["heads0"][:, hd]) for hd in range(4)}; mp = {0: (jnp.asarray(up), rD["mlp_out0"])}
        m1 = np.zeros((n, 43), bool)
        if "Q" in cut: m1 |= mQ
        if "R" in cut: m1 |= mR
        if "O" in cut: m1 |= mO
        if m1.any():
            for hd in range(4): hp[(1, hd)] = (jnp.asarray(m1), rB["heads1"][:, hd])
        rp = None
        if "O" in cut:
            mo = jnp.asarray(up)[..., None]; rp = {2: lambda x: jnp.where(mo, z2b, x)}
        L = np.asarray(F.forward(M, jnp.asarray(PR["Xb"]), head_patch=hp, mlp_patch=mp, resid_patch=rp))[i, q_]
        return RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)), pred)["slope"]
    s = {"".join(c) or "none": transfer(set(c)) for k in range(4) for c in itertools.combinations("QRO", k)}
    tot = s["none"]; key = lambda cut: "".join(c for c in "QRO" if c in cut) or "none"
    suff = {r: s[key(set("QRO") - {r})] / tot for r in "QRO"}; nec = {r: 1 - s[key({r})] / tot for r in "QRO"}
    r = dict(transfer_all_open=tot, subsets={k: v / tot for k, v in s.items()}, sufficiency=suff, necessity=nec, redundancy=sum(suff.values()) - 1.0,
             restoration_R_after_Q_cut=suff["R"], restoration_O_after_Q_cut=s[key({"Q", "R"})] / tot)
    out[sp] = r; print(sp, {k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "d_routes" + os.environ.get("DT_TAG", "") + ".json"), "w"), indent=1)
