"""Workstream A: independent audit of Investigation X Theorem X1 (numpy; exact checks).  -> logs/audit_x1.json
Setting: T4 (G=3, C=1), U_g = W_g L_g in R^3_{>0} (per-candidate), e in {0,1}^3 with S = sum e >= 1 (on support).
  I1 z1 = (S, A = sum_g U_g, B = sum_g e_g U_g)              p ∝ S A - B
  I2 z2 = (m, U), m_g = sum_{h != g} e_h                    p ∝ sum_g m_g U_g
  I3 z3 = (T_gh = U_g e_h, g != h)                          p ∝ sum_{g != h} T_gh
Interventions on a base given a source:  VS: set S := S_src, keep e_base, U_base   (high-level INCONSISTENT: S != sum e_base in general)
                                         VE: set e := e_src, keep U_base            (consistent)
Target posterior: VS -> ∝ sum_g U_b,g (S_s - e_b,g);  VE -> ∝ sum_g U_b,g (S_s - e_s,g).
Three realizability classes for an intervention on representation z:
  R_any   : some function F(z_b, z_s) whose output under the implementation's readout equals the target (up to normalisation)
  R_lin   : F linear in (z_b, z_s)
  R_proj  : F(z_b, z_s) = z_b + P (z_s - z_b), P an orthogonal projection (the class searched by DAS / linear interchange)
Checks:
  C0  the Investigation X I2-VS proof sketch: 'm_b = m_s while S_b differs' -- impossible since S = (sum m)/2 (exhaustive over e)
  C1  I1-VS in R_proj (coordinate S)                       exact
  C2  I2-VE in R_proj (m coordinates)                      exact
  C3  I2-VS in R_lin (m' = m_b + ((1.m_s - 1.m_b)/2) 1)     exact  -> I2-VS is NOT impossible in general
  C4  I2-VS not in R_proj restricted to m-coordinates (proof in MATHEMATICAL_AUDIT_AND_RESULTS.md); numerical confirmation: dense search
      over all projections of R^3 (rank 0..3) with per-pair optimal positive scale, minimum residual > 0
  C5  I1-VE not in R_any: two on-support bases with identical z1 (groups 1,2 swapped, both non-eligible) and a source with different targets
  C6  I3-VE, I3-VS not in R_any: two bases with identical z3 differing only in the unique eligible group's U; targets differ
  C7  natural-input witnesses for C5/C6 drawn from the generator (genlib), verifying identical z and different targets on real episodes
"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, genlib as GL
np.seterr(all="ignore"); out = {}; rng = np.random.default_rng(1101)
E = [np.array(e, float) for e in itertools.product([0, 1], repeat=3) if sum(e) >= 1]
m_of = lambda e: e.sum() - e
nrm = lambda p: p / p.sum()
def tgt(U, eb, es, kind): w = (es.sum() - eb) if kind == "VS" else (es.sum() - es); return nrm((w[:, None] * U).sum(0))
# C0
c0 = [(tuple(a), tuple(b)) for a in E for b in E if np.allclose(m_of(a), m_of(b)) and a.sum() != b.sum()]
out["C0_pairs_equal_m_different_S"] = len(c0); out["C0_identity_S_equals_half_sum_m"] = all(np.isclose(e.sum(), m_of(e).sum() / 2) for e in E)
# random U for exact checks
def randU(): return rng.random((3, 3)) + 0.05
err = {"C1": 0.0, "C2": 0.0, "C3": 0.0}
for _ in range(2000):
    Ub, Us = randU(), randU(); eb, es = E[rng.integers(len(E))], E[rng.integers(len(E))]
    # C1: I1, swap S coordinate
    S1 = es.sum(); p = nrm(S1 * Ub.sum(0) - (eb[:, None] * Ub).sum(0)); err["C1"] = max(err["C1"], float(np.abs(p - tgt(Ub, eb, es, "VS")).max()))
    # C2: I2, swap m coordinates
    p = nrm((m_of(es)[:, None] * Ub).sum(0)); err["C2"] = max(err["C2"], float(np.abs(p - tgt(Ub, eb, es, "VE")).max()))
    # C3: I2, linear non-projection map for VS
    mb, ms = m_of(eb), m_of(es); mp = mb + ((ms.sum() - mb.sum()) / 2) * np.ones(3)
    p = nrm((mp[:, None] * Ub).sum(0)); err["C3"] = max(err["C3"], float(np.abs(p - tgt(Ub, eb, es, "VS")).max()))
out["C1_I1_VS_projection_max_err"], out["C2_I2_VE_projection_max_err"], out["C3_I2_VS_linear_map_max_err"] = err["C1"], err["C2"], err["C3"]
# C4: all projections P on R^3 acting on m (U kept), proportionality with per-pair optimal scale; generic U so need m' ∝ target weight vector
pairs = [(a, b) for a in E for b in E]
def resid(P):
    r = 0.0
    for eb, es in pairs:
        mp = m_of(eb) + P @ (m_of(es) - m_of(eb)); t = es.sum() - eb                       # need mp = c t, c > 0
        c = max(mp @ t / (t @ t), 0.0); r = max(r, float(np.linalg.norm(mp - c * t)))
    return r
best = {0: resid(np.zeros((3, 3))), 3: resid(np.eye(3))}
for k in (1, 2):
    b = np.inf
    for _ in range(40000):
        Q, _ = np.linalg.qr(rng.standard_normal((3, k))); b = min(b, resid(Q @ Q.T))
    best[k] = b
best["sym_111"] = resid(np.ones((3, 3)) / 3)
out["C4_min_residual_over_projections_by_rank"] = {str(k): v for k, v in best.items()}
# C5: I1-VE information impossibility (constructed): base1 U, base2 = groups 1,2 swapped; e_b = (1,0,0); source e_s = (0,1,0)
U = randU(); U2 = U[[0, 2, 1]]; eb = np.array([1., 0, 0]); es = np.array([0., 1, 0])
z1 = lambda U, e: np.concatenate([[e.sum()], U.sum(0), (e[:, None] * U).sum(0)])
out["C5_I1_same_z1"] = bool(np.allclose(z1(U, eb), z1(U2, eb))); out["C5_targets_differ_L1"] = float(np.abs(tgt(U, eb, es, "VE") - tgt(U2, eb, es, "VE")).sum())
# C6: I3 information impossibility: e_b = (1,0,0) (unique eligible g*=0): z3 does not contain U_0
def z3(U, e): return np.concatenate([U[g] * e[h] for g in range(3) for h in range(3) if g != h])
Ua = randU(); Ub_ = Ua.copy(); Ub_[0] = randU()[0]
out["C6_I3_same_z3"] = bool(np.allclose(z3(Ua, eb), z3(Ub_, eb)))
out["C6_VE_targets_differ_L1"] = float(np.abs(tgt(Ua, eb, es, "VE") - tgt(Ub_, eb, es, "VE")).sum())
es2 = np.array([1., 1, 0]); out["C6_VS_targets_differ_L1(S_src=2)"] = float(np.abs(tgt(Ua, eb, es2, "VS") - tgt(Ub_, eb, es2, "VS")).sum())
# C7: natural witnesses from generator episodes
env = GL.Env(G=3, C=1); d = GL.sample(env, 200000, np.random.default_rng(7)); X = d["X"]; q = np.random.default_rng(8).integers(0, 6, len(X)); i = np.arange(len(X))
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[i, q]); Ug = (W[..., None] * rel[i, q] / noi)
cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]; e = ((cnt == 5) & (a_[..., 0] == a_[..., 1])).astype(float)
# C5 natural: base episode with e=(1,0,0) and its group-1/2-swapped twin is itself a natural episode (permuting source slots of groups 1,2)
sel = np.where((e[:, 0] == 1) & (e[:, 1] == 0) & (e[:, 2] == 0))[0][:500]
Xs = X[sel].reshape(len(sel), 6, 3, 2)[:, :, [0, 2, 1]].reshape(len(sel), 6, 6)
rel2, _, _ = GL._group_lik(env, Xs); relm2 = rel2.mean(-1); lr2 = np.log(relm2 / noi); ii = np.arange(len(sel))
W2 = np.exp(lr2.sum(1) - lr2[ii, q[sel]]); U2n = W2[..., None] * rel2[ii, q[sel]] / noi
same = np.allclose(np.stack([z1(Ug[s], e[s]) for s in sel]), np.stack([z1(U2n[j], e[sel][j]) for j in ii]), rtol=1e-9)
dif = np.array([np.abs(tgt(Ug[s], e[s], es, "VE") - tgt(U2n[j], e[s], es, "VE")).sum() for j, s in enumerate(sel)])
out["C7_natural_I1_witness"] = dict(n=int(len(sel)), identical_z1=bool(same), frac_targets_differ=float((dif > 1e-6).mean()), median_L1=float(np.median(dif)))
json.dump(out, open(os.path.join(HERE, "..", "logs", "audit_x1.json"), "w"), indent=1); print(json.dumps(out, indent=1))
