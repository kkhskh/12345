"""XIII-D Workstream E4: an identifiable causal source of the early class's NON-INVARIANCE error?
Bayes is exactly invariant to MOVE edits (one agreement of a group moved from question p1 to p2, both != q; n_g kept) -- XIII-C b5: MOVE is
the costliest invariance violation.  Hypothesis H_pos: the readout-token attention that pools other-question information weights the other
questions UNEQUALLY (position-specific weights), so moving an agreement between questions changes the decision.
Measurements:
  ASYM_l   for layer l in {1, 2}: mean over heads/episodes of the coefficient of variation across the 5 other questions of the readout token's
           attention mass on each question's 6 source tokens.
  MOVE     mean |Δ centred log p| of the model on 1200 MOVE pairs (stream E4_STREAM); KL_nat = KL(Bayes || model) on 3000 natural episodes.
Interventions (forward-time, on attention rows of the readout tokens; each renormalised):
  SYM_l    replace, for each group g and other question p, the row weight on (p, g's two tokens) by its mean over the 5 other questions (keeps the
           mass on q, on the readout/episode tokens and on each group; removes only position specificity) -- in layer 1, layer 2, or both
  PERM_l   control: permute the 5 other-question blocks of the row by a random permutation per episode (keeps the row's non-uniformity and
           mass distribution; breaks its alignment with question identity)
Prediction of H_pos: SYM lowers MOVE by a large fraction and does not raise KL_nat; PERM does not lower MOVE.
Usage: E4_TAG=_dev python3 scripts/e4_noninv.py SPEC...  -> logs/e4_noninv<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
ST = int(os.environ.get("E4_STREAM", 90_600_000)); rng = np.random.default_rng(ST)
def move_pairs(n=1200):
    Xa, Xb, qq = [], [], []
    while len(qq) < n:
        X = NV.Bm.sample_fast(NV.ENV, 1, rng)["X"][0]; q = rng.integers(6); a = X.reshape(6, 3, 2); g = rng.integers(3); oth = [p for p in range(6) if p != q]
        ag = [p for p in oth if a[p, g, 0] == a[p, g, 1]]; dg = [p for p in oth if a[p, g, 0] != a[p, g, 1]]
        if not ag or not dg: continue
        p1, p2 = ag[rng.integers(len(ag))], dg[rng.integers(len(dg))]; x = X.copy(); x[p1, 2 * g + 1] = (x[p1, 2 * g] + 1 + rng.integers(2)) % 3; x[p2, 2 * g + 1] = x[p2, 2 * g]
        Xa.append(X); Xb.append(x); qq.append(q)
    return np.array(Xa), np.array(Xb), np.array(qq)
Xa, Xb, qm = move_pairs(); chk = float(np.abs(GL.posterior_closed(RP.ENV1, Xb, qm) - GL.posterior_closed(RP.ENV1, Xa, qm)).max())
Xn = NV.Bm.sample_fast(NV.ENV, 3000, rng)["X"]; qn = rng.integers(0, 6, 3000); PBn = GL.posterior_closed(RP.ENV1, Xn, qn)
PERMS = np.array([rng.permutation(5) for _ in range(5000)])
def fwd(M, X, q, mode=None, layers=()):
    """manual forward; for l in layers modify the readout-token (36 + q) attention rows of every head."""
    x = F.N.embed(dict(M["emb"]), jnp.asarray(X)); n = len(X); i = np.arange(n)
    oth = np.array([[p for p in range(6) if p != qq] for qq in q])                       # (n,5)
    for l in range(3):
        lay = M["layers"][l]; z = F.N.ln(x, lay["g1"], lay["b1"])
        qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1)
        if l in layers:
            An = np.array(A); row = An[i, :, 36 + q]                                        # (n,4,43)
            blk = np.stack([row[i[:, None], :, (oth[:, j][:, None] * 6 + np.arange(6)[None])].transpose(0, 2, 1) for j in range(5)], 2)   # (n,4,5,6)
            if mode == "SYM": nb = np.repeat(blk.reshape(n, 4, 5, 3, 2).mean(2, keepdims=True), 5, 2).reshape(n, 4, 5, 6)
            else: pm = PERMS[:n]; nb = blk[i[:, None, None], np.arange(4)[None, :, None], pm[:, None, :]]
            for j in range(5): row[i[:, None], :, (oth[:, j][:, None] * 6 + np.arange(6)[None])] = nb[:, :, j].transpose(0, 2, 1)
            An[i, :, 36 + q] = row; A = jnp.asarray(An)
        x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"])
        x = x + jax.nn.gelu(F.N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
    return np.asarray(F.N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"])[i, q]
def batched(M, X, q, **kw): return np.concatenate([fwd(M, X[a:a + 300], q[a:a + 300], **kw) for a in range(0, len(X), 300)])
def asym(M, l):
    rec = {}; F.forward(M, jnp.asarray(Xn[:1000]), rec=rec); A = np.asarray(rec[f"pattern{l}"]); i = np.arange(1000); q = qn[:1000]
    oth = np.array([[p for p in range(6) if p != qq] for qq in q]); row = A[i, :, 36 + q]
    mass = np.stack([row[i[:, None], :, (oth[:, j][:, None] * 6 + np.arange(6)[None])].sum(1) for j in range(5)], -1)    # (n,4,5)
    return float((mass.std(-1) / (mass.mean(-1) + 1e-9)).mean())
out = dict(move_bayes_invariance_maxdiff=chk, models={})
for sp in sys.argv[1:]:
    M = xc.load(sp); r = dict(ASYM1=asym(M, 1), ASYM2=asym(M, 2))
    for nm, kw in (("BASE", {}), ("SYM1", dict(mode="SYM", layers=(1,))), ("SYM2", dict(mode="SYM", layers=(2,))), ("SYM12", dict(mode="SYM", layers=(1, 2))),
                   ("PERM1", dict(mode="PERM", layers=(1,))), ("PERM12", dict(mode="PERM", layers=(1, 2)))):
        La, Lb = batched(M, Xa, qm, **kw), batched(M, Xb, qm, **kw); P = RP.sm(batched(M, Xn, qn, **kw))
        r[nm] = dict(MOVE=float(np.abs(RP.cen(RP.sm(Lb)) - RP.cen(RP.sm(La))).mean()), KL_nat=float((PBn * (np.log(PBn + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()),
                     acc=float((P.argmax(-1) == PBn.argmax(-1)).mean()))
    for nm in ("SYM1", "SYM2", "SYM12", "PERM1", "PERM12"): r[nm]["MOVE_reduction"] = 1 - r[nm]["MOVE"] / r["BASE"]["MOVE"]
    out["models"][sp] = r; print(sp, {k: ({a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else round(v, 3)) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "e4_noninv" + os.environ.get("E4_TAG", "") + ".json"), "w"), indent=1)
