"""XIII-C Workstream D1/D2/D3: the layer-2 attention-row mediator re-tested with REBUILT controls (mixture / hybrid units).
Pairs: purely relational pairs (relpairs.relational_pairs; values at q identical) on stream DR_STREAM.
Mixture heads: relpairs.mixture_heads (the single XIII-B rule).  Intervention = replace the mixture heads' readout-token (36+q) attention rows
by rows from some source, everything else from the base run; observed change = centred log-prob change at q.
Patch classes (D3 taxonomy):
  TRUE       natural component-level donor: rows of the valid relational donor of THIS pair                         [component natural donor]
  NOOP       same-state replacement: rows of the base itself; must reproduce the unmodified forward (max |Δlogit| reported)  [genuine no-op]
  SHUF       position-matched natural rows of the SAME heads from ANOTHER pair's donor with the same q (wrong relational content);
             matched by construction for head, dimensionality, token positions, attention mass and baseline contribution;
             displacement magnitude reported                                                                       [contribution-matched decoy]
  SHUFD      DISPLACEMENT-matched decoy: base rows + (donor - base) row displacement of ANOTHER pair with the same q (clipped at 0, renormalised):
             same head, positions and displacement-magnitude distribution as TRUE, wrong relational content          [matched decoy]
  SHUFB      as SHUF but rows from another BASE with the same q (natural, same distribution, unrelated episode)
  PMEAN      position-matched mean: mean over episodes with the same q of donor rows (off-distribution average, flagged) [synthetic]
  OTHER      rows of the NON-mixture heads from the true donor (XIII-B decoy class; contribution typically NOT matched -> reported)
  WHOLE      whole natural activation replacement: resid2 at ALL tokens from the donor (upper reference)                  [whole natural]
Scores (SPECIFICITY is judged by R2 = variance of the Bayes change explained, not by slope: an unrelated large perturbation can have slope > 0):
  fidelity slope / R2 of the patched change on the BAYES change of the pair, share of the model's OWN total change (projection),
mean |Δ| (magnitude), and the mean row-displacement L1 distance from the base rows.
Also D7: routes (route_Q, route_R) on DR_NSTREAMS independent streams -> mean, sd, bootstrap CI over episodes on the first stream.
Usage: DR_TAG=_dev python3 scripts/d_rows.py SPEC... -> logs/d_rows<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP, metrics as MT
np.seterr(all="ignore")
ST = int(os.environ.get("DR_STREAM", 86_600_000)); NS = int(os.environ.get("DR_NSTREAMS", 4))
PR = RP.relational_pairs(800, np.random.default_rng(ST)); n = len(PR["q"]); i = np.arange(n); q = PR["q"]; dB = RP.cen(PR["PBd"]) - RP.cen(PR["PB"])
rs = np.random.default_rng(ST + 1); perm_d = np.zeros(n, int); perm_b = np.zeros(n, int)
for j in range(n):
    c = np.where((q == q[j]) & (i != j))[0]; perm_d[j] = rs.choice(c); perm_b[j] = rs.choice(c)
streams = [RP.relational_pairs(500, np.random.default_rng(ST + 100 + s)) for s in range(NS)]
nat = NV.Bm.sample_fast(NV.ENV, 1500, np.random.default_rng(ST + 2))["X"]; qn = np.random.default_rng(ST + 3).integers(0, 6, 1500)
out = {}
def run_rows(M, heads, rows_by_head, Xb):
    """forward of Xb with readout-token rows of `heads` replaced (rows_by_head[h]: (n,43) attention row at 36+q)."""
    lay = M["layers"][2]; rec = {}; F.forward(M, jnp.asarray(Xb), rec=rec); x2 = jnp.asarray(rec["resid2"])
    z = F.N.ln(x2, lay["g1"], lay["b1"]); qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1); nn = jnp.arange(len(Xb)); t = jnp.asarray(36 + q)
    for h in heads: A = A.at[nn, h, t].set(jnp.asarray(rows_by_head[h]))
    xa = x2 + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"])
    hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); xo = xa + hid @ lay["W2"] + lay["c2"]
    return np.asarray(F.N.ln(xo, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[i, q]
for sp in sys.argv[1:]:
    M = xc.load(sp); mass, share = RP.head_stats(M, nat, qn, F, jnp); MH, MHm = RP.mixture_heads(mass, share)
    rB, rD = {}, {}; L0 = np.asarray(F.forward(M, jnp.asarray(PR["Xb"]), rec=rB))[i, q]; LD = np.asarray(F.forward(M, jnp.asarray(PR["Xd"]), rec=rD))[i, q]
    AB = np.asarray(rB["pattern2"])[i, :, 36 + q]; AD = np.asarray(rD["pattern2"])[i, :, 36 + q]; own = RP.cen(RP.sm(LD)) - RP.cen(RP.sm(L0))
    r = dict(mixture_heads=MH, mass=mass.tolist(), share=list(map(float, share)), arms={})
    def score(L, rows=None, heads=None):
        d = RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)); f = RP.fid(d, dB)
        s = dict(slope=f["slope"], R2=f["R2"], own_share=float((d * own).sum() / max((own * own).sum(), 1e-12)), mean_abs=float(np.abs(d).mean()))
        if rows is not None: s["row_disp_L1"] = float(np.mean([np.abs(rows[h] - AB[:, h]).sum(-1).mean() for h in heads]))
        return s
    r["own_total"] = dict(**RP.fid(own, dB))
    if MH:
        pm = np.zeros_like(AD)
        for qq in range(6): pm[q == qq] = AD[q == qq].mean(0)
        arms = dict(TRUE={h: AD[:, h] for h in MH}, NOOP={h: AB[:, h] for h in MH}, SHUF={h: AD[perm_d, h] for h in MH}, PMEAN={h: pm[:, h] for h in MH})
        dsp = np.clip(AB + (AD - AB)[perm_d], 0, None); dsp = dsp / dsp.sum(-1, keepdims=True); arms["SHUFD"] = {h: dsp[:, h] for h in MH}
        rBB = {}; F.forward(M, jnp.asarray(PR["Xb"][perm_b]), rec=rBB); ABB = np.asarray(rBB["pattern2"])[i, :, 36 + q]; arms["SHUFB"] = {h: ABB[:, h] for h in MH}
        for nm, rows in arms.items(): Lp = run_rows(M, MH, rows, PR["Xb"]); r["arms"][nm] = score(Lp, rows, MH)
        r["arms"]["NOOP"]["max_abs_dlogit_vs_unmodified"] = float(np.abs(run_rows(M, MH, arms["NOOP"], PR["Xb"]) - L0).max())
        oth = [h for h in range(4) if h not in MH]
        if oth:
            rows = {h: AD[:, h] for h in oth}; r["arms"]["OTHER"] = score(run_rows(M, oth, rows, PR["Xb"]), rows, oth)
            r["other_heads_baseline_share"] = [float(share[h]) for h in oth]; r["mixture_heads_baseline_share"] = [float(share[h]) for h in MH]
    rd2 = np.asarray(rD["resid2"]); LW = np.asarray(F.forward(M, jnp.asarray(PR["Xb"]), resid_patch={2: lambda x: jnp.asarray(rd2)}))[i, q]; r["arms"]["WHOLE"] = score(LW)
    rq = [MT.routes(M, s) for s in streams]; rQ = np.array([x["route_Q"] for x in rq]); rR = np.array([x["route_R"] for x in rq])
    r["routes_streams"] = dict(route_Q=rQ.tolist(), route_R=rR.tolist(), route_Q_mean=float(rQ.mean()), route_Q_sd=float(rQ.std(ddof=1)), route_R_mean=float(rR.mean()), route_R_sd=float(rR.std(ddof=1)))
    out[sp] = r; json.dump(out, open(os.path.join(HERE, "..", "logs", "d_rows" + os.environ.get("DR_TAG", "") + ".json"), "w"), indent=1)
    print(sp, "MH", MH, {k: {a: round(b, 3) for a, b in v.items()} for k, v in r["arms"].items()}, "routeQ", np.round(rQ, 3), flush=True)
