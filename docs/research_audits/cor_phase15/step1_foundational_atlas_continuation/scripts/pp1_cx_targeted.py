"""CX2 targeted: the same low-density-corrupted model, scored on an intervention family whose endpoints are CONCENTRATED
in the low-density region (density ratio >> 1).  Edit family: natural T4 episodes with the 'make a second group perfect'
edit -- for each episode, set every disagreeing pair of the NOISE group equal (all questions), then score margin changes
at q for the base->edit pair.  Reports the fraction of endpoints in the corrupted region and sign agreement."""
import json, os, sys
HERE = os.path.dirname(__file__); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src"))
import numpy as np, gen_family as G
f4 = G.Fam(); K, Q = 3, 6
d = G.sample(f4, 3000, np.random.default_rng(6_160_000)); X0, cfg = d["X"], d["cfg"]; cf = f4.configs
noi = np.array([3 - cf[c][0] - cf[c][1] for c in cfg])
Xe = X0.copy(); n = np.arange(len(X0))
q = 0
for k in range(1, Q):                                      # make the noise group agree in every question except q
    a, b = 2 * noi, 2 * noi + 1; Xe[n, k, b] = Xe[n, k, a]
def lp(X):
    p = G.posterior(f4, X); lx = p["logM"].sum(1); m = lx.max(-1); m = np.where(np.isfinite(m), m, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p["post_y"]), np.log(np.exp(lx - m[:, None]).mean(-1)) + m
l0, px0 = lp(X0); le, pxe = lp(Xe)
Dn = G.sample(f4, 3000, np.random.default_rng(6_150_000)); _, pxn = lp(Dn["X"]); thr = np.percentile(pxn, 2)
rng = np.random.default_rng(7)
def corrupt(l, px):
    l = np.nan_to_num(l, nan=-50.0, neginf=-50.0).copy()
    for i in np.where(px < thr)[0]: l[i] = l[i][:, rng.permutation(K)]
    return l
c0, ce = corrupt(l0, px0), corrupt(le, pxe)
order = np.argsort(-np.nan_to_num(l0[:, q], nan=-1e9), -1); c1, c2 = order[:, 0], order[:, 1]
dm = (le[n, q, c1] - le[n, q, c2]) - (l0[n, q, c1] - l0[n, q, c2]); dmc = (ce[n, q, c1] - ce[n, q, c2]) - (c0[n, q, c1] - c0[n, q, c2])
sel = np.isfinite(dm) & (np.abs(dm) > 0.5)
res = dict(frac_edit_endpoints_low_density=float((pxe < thr)[sel].mean()), frac_natural_low_density=float((pxn < thr).mean()),
           n_scored=int(sel.sum()), sign_agreement_corrupted_model=float((np.sign(dmc[sel]) == np.sign(dm[sel])).mean()))
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp1_cx_targeted.json"), "w"), indent=1); print(res)
