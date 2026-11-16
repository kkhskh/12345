"""XIV [num]: independent checks of CROSS_FAMILY_MATHEMATICAL_STRUCTURE.md.
(1) T4 Proposition 1: single-source marginals uniform under every role (exact computation from the role laws) and empirical check on
    200k native T4 episodes (per role, chi-square).  (2) T4 consequence (b): the exact posterior for question q is invariant (max |diff|)
    under value relabelling within a pair at a non-target question that preserves within-pair equality.  (3) GATE consequence (b): Bayes is
    exactly invariant under (x0, x1) -> (x0+k, x1-k) on all 3^6 inputs.  (4) GATE Proposition 6: power means of the two pair likelihoods
    with the Bayes (point-mass) weights coincide for rho in {0.01, 0.5, 1, 2}.  Writes logs/math_verify.json."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, native as NV, genlib as GL, relpairs as RP, gatelib as GTL
from scipy.stats import chisquare
out = {}; K, a = 3, 0.8
out["T4_reliable_marginal"] = [(1 / K) * a + (1 - 1 / K) * (1 - a) / (K - 1)] * K
rng = np.random.default_rng(97_500_000); d = NV.Bm.sample_fast(NV.ENV, 50000, rng); X = d["X"]
ch = [float(chisquare(np.bincount(X[:, :, s].ravel(), minlength=3)).pvalue) for s in range(6)]
out["T4_single_source_uniform_chisq_p"] = ch
q = rng.integers(0, 6, 20000); Xs = X[:20000].copy(); P0 = GL.posterior_closed(RP.ENV1, Xs, q)
Xe = Xs.copy(); qq = (q + 1 + rng.integers(0, 5, len(q))) % 6; pr = rng.integers(0, 3, len(q)); perm = np.array([list(p) for p in itertools.permutations(range(3))])[rng.integers(0, 6, len(q))]
for i in range(len(q)):
    for s in (2 * pr[i], 2 * pr[i] + 1): Xe[i, qq[i], s] = perm[i][Xs[i, qq[i], s]]        # same permutation on both sources of the pair: equality preserved
P1 = GL.posterior_closed(RP.ENV1, Xe, q); out["T4_nontarget_relabel_max_abs_diff"] = float(np.abs(P1 - P0).max())
Xb = Xs.copy()
for i in range(len(q)): Xb[i, qq[i], 2 * pr[i]] = (Xs[i, qq[i], 2 * pr[i]] + 1) % 3                       # control: equality-breaking/making edit
Pb = GL.posterior_closed(RP.ENV1, Xb, q); fin = np.isfinite(Pb).all(-1); out["T4_nontarget_equality_edit_mean_abs_diff_control"] = float(np.abs(Pb[fin] - P0[fin]).mean()); out["T4_equality_edit_impossible_fraction"] = float(1 - fin.mean())
Xall = np.array(list(itertools.product(range(3), repeat=6)))[:, None, :]
B = GTL.rule("bayes", Xall); md = 0.0
for k in (1, 2):
    Xk = Xall.copy(); Xk[:, :, 0] = (Xall[:, :, 0] + k) % 3; Xk[:, :, 1] = (Xall[:, :, 1] - k) % 3; md = max(md, float(np.abs(GTL.rule("bayes", Xk) - B).max()))
out["GATE_shift_invariance_max_abs_diff"] = md
L23, L45 = GTL.lik(Xall, (2, 3)), GTL.lik(Xall, (4, 5)); w = GTL.gate(Xall)[..., None].astype(float); dev = 0.0
for rho in (0.01, 0.5, 1.0, 2.0):
    M = (w * L23 ** rho + (1 - w) * L45 ** rho) ** (1 / rho); M /= M.sum(-1, keepdims=True); dev = max(dev, float(np.abs(M - B).max()))
out["GATE_powermean_pointmass_max_abs_diff"] = dev
json.dump(out, open(os.path.join(HERE, "..", "logs", "math_verify.json"), "w"), indent=1); print(json.dumps(out, indent=1))
