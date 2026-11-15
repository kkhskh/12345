"""XIII-C Workstream B5: where does a model's deviation from exact Bayes come from, beyond any parametric decision family?
Sufficient statistics of the C = 1 exact decision at q: for each group g, its agreement count n_g at the other five questions and its two values
at q.  Decomposition of D = KL(Bayes-free model P || Bayes) on natural episodes (stream B5_STREAM):
  1. INVARIANCE VIOLATIONS (non-sufficient information).  Sufficiency-preserving edits leave Bayes exactly unchanged:
       VPERM  : relabel one group's two values at one other question q' by a value permutation (self-agreement kept)
       MOVE   : move one agreement of a group from q' to q'' (both != q; n_g kept)
       XVAL   : change one value of a non-self-agreeing pair at q' to the third value (still non-agreeing)
     the model's mean KL(P(x) || P(x')) and mean |Δ centred log p| measure dependence on information Bayes ignores.
  2. CELL MODEL (best function of the sufficient statistics).  Episodes are canonicalised (group order and candidate relabelling) by the
     sufficient statistics; within each cell the model's canonical log-probs are averaged on a TRAINING half; on the held-out half
       KL(P || cell)      = within-cell variation (dependence on non-sufficient information + unseen cells),
       KL(cell || Bayes)  = deviation that IS a function of the sufficient statistics (structured miscomputation).
  3. the parametric families of fair_fits.py are a subset of 2 (reported there).
Usage: B5_TAG=_dev python3 scripts/b5_residual.py SPEC... -> logs/b5_residual<TAG>.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
rng = np.random.default_rng(int(os.environ.get("B5_STREAM", 86_800_000))); N_ = 60000
X = NV.Bm.sample_fast(NV.ENV, N_, rng)["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_); PB = GL.posterior_closed(RP.ENV1, X, q)
a6 = X.reshape(N_, 6, 3, 2); eq = a6[..., 0] == a6[..., 1]; cnt = eq.sum(1) - eq[ii, q]; vq = a6[ii, q]          # cnt (N,3); vq (N,3,2)
PERMS = list(itertools.permutations(range(3)))
def canon(j):
    best = None
    for pc in PERMS:                                                  # candidate relabelling y -> pc[y]
        g_keys = [(int(cnt[j, g]), tuple(sorted((pc[vq[j, g, 0]], pc[vq[j, g, 1]])))) for g in range(3)]
        key = tuple(sorted(g_keys))
        if best is None or key < best[0]: best = (key, pc)
    return best
keys = []; maps = np.zeros((N_, 3), int)
for j in range(N_):
    k, pc = canon(j); keys.append(k); maps[j] = pc                        # canonical candidate pc[y] <- original y
kid = {k: n for n, k in enumerate(sorted(set(keys)))}; K = np.array([kid[k] for k in keys])
def to_canon(L):                                                          # L (N,3) indexed by original y -> canonical order
    C = np.zeros_like(L); C[ii[:, None], maps] = L; return C
half = N_ // 2; tr = ii < half; te = ~tr
def kl(P, Q): return float((P * (np.log(P + 1e-300) - np.log(Q + 1e-300))).sum(-1).mean())
# sufficiency-preserving edit sets
def edits(kind, n=1500, seed=5):
    r = np.random.default_rng(seed); Xa, Xb, qq = [], [], []
    for j in r.permutation(N_):
        if len(qq) >= n: break
        x = X[j].copy(); a = x.reshape(6, 3, 2); others = [p for p in range(6) if p != q[j]]; g = r.integers(3)
        if kind == "VPERM":
            p = others[r.integers(5)]; pm = np.array(PERMS[1 + r.integers(5)]); x[p, 2 * g:2 * g + 2] = pm[x[p, 2 * g:2 * g + 2]]
        elif kind == "MOVE":
            ag = [p for p in others if a[p, g, 0] == a[p, g, 1]]; dg = [p for p in others if a[p, g, 0] != a[p, g, 1]]
            if not ag or not dg: continue
            p1, p2 = ag[r.integers(len(ag))], dg[r.integers(len(dg))]
            x[p1, 2 * g + 1] = (x[p1, 2 * g] + 1 + r.integers(2)) % 3; x[p2, 2 * g + 1] = x[p2, 2 * g]
        elif kind == "XVAL":
            dg = [p for p in others if a[p, g, 0] != a[p, g, 1]]
            if not dg: continue
            p = dg[r.integers(len(dg))]; s = r.integers(2); x[p, 2 * g + s] = 3 - a[p, g, 0] - a[p, g, 1]
        Xa.append(X[j]); Xb.append(x); qq.append(q[j])
    Xa, Xb, qq = np.array(Xa), np.array(Xb), np.array(qq)
    chk = float(np.abs(GL.posterior_closed(RP.ENV1, Xb, qq) - GL.posterior_closed(RP.ENV1, Xa, qq)).max()); return Xa, Xb, qq, chk
ED = {k: edits(k) for k in ("VPERM", "MOVE", "XVAL")}
out = dict(n=N_, n_cells=len(kid), edits_bayes_invariance_maxdiff={k: v[3] for k, v in ED.items()}, models={})
print("cells", len(kid), out["edits_bayes_invariance_maxdiff"], flush=True)
for sp in sys.argv[1:]:
    M = xc.load(sp); fj = jax.jit(lambda X_: F.forward(M, X_)); L = np.asarray(F.batched(fj, X))[ii, q]; P = RP.sm(L)
    lc = to_canon(np.log(P + 1e-300)); S = np.zeros((len(kid), 3)); C = np.zeros(len(kid))
    np.add.at(S, K[tr], lc[tr]); np.add.at(C, K[tr], 1)
    seen = C[K] > 0; cell = S[K] / np.maximum(C[K], 1)[:, None]; cellP = np.exp(cell - np.log(np.exp(cell).sum(-1, keepdims=True)))
    Pc = to_canon(P); PBc = to_canon(PB); m = te & seen
    r = dict(KL_model_bayes=kl(Pc[te], PBc[te]), KL_model_cell=kl(Pc[m], cellP[m]), KL_cell_bayes=kl(cellP[m], PBc[m]), frac_test_seen=float(seen[te].mean()))
    r["frac_structured"] = r["KL_cell_bayes"] / max(r["KL_model_bayes"], 1e-12); r["frac_within_cell"] = r["KL_model_cell"] / max(r["KL_model_bayes"], 1e-12)
    for k, (Xa, Xb, qq, _) in ED.items():
        j = np.arange(len(qq)); Pa = RP.sm(np.asarray(F.batched(fj, Xa))[j, qq]); Pb = RP.sm(np.asarray(F.batched(fj, Xb))[j, qq])
        r[f"inv_{k}"] = dict(KL=kl(Pa, Pb), mean_abs=float(np.abs(RP.cen(Pb) - RP.cen(Pa)).mean()))
    out["models"][sp] = r; print(sp, {k: (round(v, 5) if isinstance(v, float) else {a: round(b, 5) for a, b in v.items()}) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "b5_residual" + os.environ.get("B5_TAG", "") + ".json"), "w"), indent=1)
