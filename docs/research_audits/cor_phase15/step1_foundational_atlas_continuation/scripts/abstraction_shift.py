"""Abstraction test (mandate §6/§7): does the model factor through RELATIVE agreement counts?
T1 generator (p_c = 0): the Bayes answer at q is invariant to adding one agreement to EVERY group (uniform shift) in another
question k, and changes when one agreement is added to a single group (single shift).  Edit: choose k != q in which all three
groups disagree internally; uniform = set b := a for all three groups in k; single = only for the group with the lowest count.
Score: mean KL(model(x) || model(x')) at q for uniform vs single, and the same for Bayes (uniform must be 0).
Models: T1 Transformers (s0-2 main, s10-13 replication) and T4 models evaluated on T1 data.  Stream 7_900_000."""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, pp2_sig as S, gen_family as G
f1 = G.Fam(pc=0.0); K, Q = 3, 6
d = G.sample(f1, 20000, np.random.default_rng(7_900_000)); X = d["X"]
Ag = np.stack([(X[:, :, a] == X[:, :, b]) for a, b in G.GROUPS], -1)          # N,Q,3
q = 0; cand = (~Ag[:, 1:].any(-1))                                              # questions 1..5 where all groups disagree
ok = cand.any(1); X = X[ok]; k = 1 + cand[ok].argmax(1); n = len(X); idx = np.arange(n)
Xu, Xs = X.copy(), X.copy(); cnt = Ag[ok].sum(1); low = cnt.argmin(-1)
for g, (a, b) in enumerate(G.GROUPS):
    Xu[idx, k, b] = Xu[idx, k, a]
    sel = low == g; Xs[idx[sel], k[sel], b] = Xs[idx[sel], k[sel], a]
def kl(P, R): return float((P * (np.log(P + 1e-12) - np.log(R + 1e-12))).sum(-1).mean())
pb = lambda Z: G.posterior(f1, Z)["post_y"][:, q]
res = dict(n=n, bayes=dict(uniform=kl(pb(X), pb(Xu)), single=kl(pb(X), pb(Xs))))
M = {f"T1_s{s}": ("tf", f"checkpoints/main/T1_M_K3_s{s}.npz") for s in [0, 1, 2]}
M.update({f"T1_s{s}": ("tf", f"checkpoints/replication/T1_M_K3_s{s}.npz") for s in [10, 11, 12, 13]})
M.update({f"T4tf_s{s}": ("tf", f"checkpoints/replication/T4_M_K3_s{s}_fromT1.npz") for s in [10, 12]}); M["T4ds_s0"] = ("ds", "checkpoints/main/T4_M_K3_s0_deepsets.npz")
for nm, (kd, pth) in M.items():
    full = os.path.join(LR, pth)
    if not os.path.exists(full): continue
    ap = S.load(kd, full); sm = lambda Z: np.exp(ap(Z)[0][:, q] - np.log(np.exp(ap(Z)[0][:, q]).sum(-1, keepdims=True)))
    P0, Pu, Ps = sm(X), sm(Xu), sm(Xs)
    res[nm] = dict(uniform=kl(P0, Pu), single=kl(P0, Ps), invariance_ratio=kl(P0, Pu) / max(kl(P0, Ps), 1e-9))
    print(nm, res[nm], flush=True)
print(res["bayes"], n); json.dump(res, open(os.path.join(HERE, "..", "logs", "abstraction_shift.json"), "w"), indent=1)
