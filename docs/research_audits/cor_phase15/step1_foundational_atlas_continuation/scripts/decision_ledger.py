"""Candidate-decision ledger (mandate §13) on the frozen PP1 edit population (S1 edits; stream 6_100_000, NB=800 for cost).
Pairs (c1, c2) = base top-2 candidates at q.  For Bayes and three models (TF s10 class A', TF s12 class C, DeepSets s0):
  reversal      : edits where the Bayes argmax at q changes; does the model's argmax change to the same candidate?
  conf_vs_pref  : edits that change Bayes confidence (|d max-prob| > 0.1) without changing the Bayes argmax; fraction in
                  which the model's argmax nevertheless changes (confidence-preference leakage)
  support_rule  : 'the edited group's own-q supported candidate gains' (candidate-specific support heuristic): its sign
                  agreement with Bayes dm for edits where the edited group agrees in q, split by the edited group's leave-q
                  count (boundary 4<->5 vs interior)
  misleading    : on p_c = 0 episodes (no colluder, so a perfectly agreeing group is honest), fraction of questions in which
                  the model's argmax differs from true Bayes when some group agrees in all 6 questions
Writes logs/decision_ledger.json."""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, gen_family as G, pp2_sig as S
f4, f1 = G.Fam(pc=1.0, rho=1.0), G.Fam(pc=0.0); K, Q = 3, 6; NB = 800
d = G.sample(f4, NB, np.random.default_rng(6_100_000)); X0 = d["X"]; bi, qq, kk, gg, Xe = [], [], [], [], []
for q in range(Q):
    for k in range(Q):
        if k == q: continue
        for g, (a, b) in enumerate(G.GROUPS):
            X = X0.copy(); ag = X[:, k, a] == X[:, k, b]; X[:, k, b] = np.where(ag, (X[:, k, a] + 1) % K, X[:, k, a])
            Xe.append(X); bi.append(np.arange(NB)); qq.append(np.full(NB, q)); kk.append(np.full(NB, k)); gg.append(np.full(NB, g))
Xe, bi, qq, kk, gg = map(np.concatenate, (Xe, bi, qq, kk, gg)); n = len(Xe); ar = np.arange(n)
with np.errstate(divide="ignore", invalid="ignore"):
    P0, Pe = G.posterior(f4, X0)["post_y"], G.posterior(f4, Xe)["post_y"]
S1 = np.isfinite(Pe).all((1, 2)); lb0, lbe = np.log(P0), np.log(Pe)
order = np.argsort(-np.nan_to_num(lb0, nan=-9), -1); c1, c2 = order[bi, qq, 0], order[bi, qq, 1]
dm = (lbe[ar, qq, c1] - lbe[ar, qq, c2]) - (lb0[bi, qq, c1] - lb0[bi, qq, c2])
b_arg0 = np.nan_to_num(P0, nan=0)[bi, qq].argmax(-1); b_arge = np.nan_to_num(Pe, nan=0)[ar, qq].argmax(-1)
rev = S1 & (b_arg0 != b_arge); conf = S1 & (b_arg0 == b_arge) & (np.abs(np.nan_to_num(Pe, nan=0)[ar, qq].max(-1) - P0[bi, qq].max(-1)) > 0.1)
A0 = np.stack([(X0[:, :, a] == X0[:, :, b]) for a, b in G.GROUPS], -1); Ae = np.stack([(Xe[:, :, a] == Xe[:, :, b]) for a, b in G.GROUPS], -1)
ga = A0[bi, qq, gg]; sup = X0[bi, qq, 2 * gg]                                   # the edited group's candidate at q (when it agrees)
cb = A0[bi].sum(1)[ar, gg] - A0[bi, qq, gg]; ca = Ae.sum(1)[ar, gg] - Ae[ar, qq, gg]
gain = np.where(ca > cb, 1, -1)                                                # count goes up or down
rule = np.where(sup == c1, gain, np.where(sup == c2, -gain, 0))               # 'support' heuristic: more agreement helps own candidate
sel = S1 & ga & (rule != 0) & (np.abs(dm) > 0.5); bnd = ((cb == 4) & (ca == 5)) | ((cb == 5) & (ca == 4))
res = dict(n_S1=int(S1.sum()), n_reversals=int(rev.sum()), n_conf_only=int(conf.sum()),
           support_rule_bayes_sign_agreement=dict(all=float((np.sign(dm) == rule)[sel].mean()), boundary=float((np.sign(dm) == rule)[sel & bnd].mean()),
                                                  interior=float((np.sign(dm) == rule)[sel & ~bnd].mean()), n=int(sel.sum())))
D1 = G.sample(f1, 3000, np.random.default_rng(9_400_000)); X1 = D1["X"]; tb1 = G.posterior(f1, X1)["post_y"].argmax(-1)
perf = np.stack([(X1[:, :, a] == X1[:, :, b]).all(1) for a, b in G.GROUPS], -1).any(-1)
M = {"tf_s10": ("tf", "checkpoints/replication/T4_M_K3_s10_fromT1.npz"), "tf_s12": ("tf", "checkpoints/replication/T4_M_K3_s12_fromT1.npz"), "ds_s0": ("ds", "checkpoints/main/T4_M_K3_s0_deepsets.npz")}
for nm, (k_, pth) in M.items():
    ap = S.load(k_, os.path.join(LR, pth)); a0 = ap(X0)[0].argmax(-1)[bi, qq]
    ae = np.concatenate([ap(Xe[i:i + 20000])[0] for i in range(0, n, 20000)]).argmax(-1)[ar, qq]
    a1 = ap(X1)[0].argmax(-1)
    res[nm] = dict(reversal_follow=float((ae == b_arge)[rev].mean()), reversal_model_flips=float((ae != a0)[rev].mean()),
                   conf_only_leakage=float((ae != a0)[conf].mean()),
                   misleading_pc0_perfect_group_disagree_true=float((a1 != tb1)[perf].mean()), pc0_other_disagree_true=float((a1 != tb1)[~perf].mean()))
    print(nm, res[nm], flush=True)
print({k: v for k, v in res.items() if not isinstance(v, dict) or k.startswith("support")})
json.dump(res, open(os.path.join(HERE, "..", "logs", "decision_ledger.json"), "w"), indent=1)
