"""M7: the single episode-level posterior P(R|X) is an exact reusable summary for every question.
M8: accuracies of simple (non-Bayes) solvers, to state which simpler models solve each task.
M9: closed-form agreement likelihood ratios and the T4 non-monotonicity."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import bayes as B

out, res = [], {}
def P(s): out.append(s); print(s)

def agree(X):           # (N,Q,6) -> (N,Q,3)
    return np.stack([X[:, :, a] == X[:, :, b] for a, b in B.GROUPS], -1).astype(int)

def vote(X, srcs, K):   # count of reports per candidate among srcs: (N,Q,K)
    return np.stack([(X[:, :, srcs] == k).sum(-1) for k in range(K)], -1)

def acc_of(scores, Y, rng):  # argmax with random tie-break
    s = scores + 1e-6 * rng.random(scores.shape)
    return float((s.argmax(-1) == Y).mean())

for t in ["T0", "T1", "T4", "T5"]:
    env = B.Env(t, K=3, Q=6); rng = np.random.default_rng(11)
    d = B.sample_fast(env, 20000, rng); X, Y = d["X"], d["Y"]
    post = B.episode_posterior(env, X)
    r = {}
    if env.R_values[0] is not None:
        # M7: P(Y_q|X) = sum_R P(R|X) P(Y_q | X_q, R)
        Rs = env.R_values
        cond = np.zeros(X.shape[:2] + (len(Rs), env.K))
        for i, R in enumerate(Rs):
            for q in range(env.Q):
                L = B.q_lik(env, X[:, q], R); cond[:, q, i] = L / np.maximum(L.sum(-1, keepdims=True), 1e-300)
        valid = post["post_R"][:, None, :, None] * cond
        recon = valid.sum(2)
        r["M7_max_err_episode_posterior_reuse"] = float(np.abs(recon - post["post_y"]).max())
    A = agree(X); n = A.sum(1)                                   # coherence counts (N,3)
    allv = vote(X, list(range(6)), env.K)
    r["majority_all6"] = acc_of(allv.astype(float), Y, rng)
    # most-coherent-group vote: use the votes of the group with the largest total agreement count
    gbest = (n + 1e-6 * rng.random(n.shape)).argmax(-1)
    gv = np.stack([vote(X, list(B.GROUPS[g]), env.K) for g in range(3)], 1)   # (N,3,Q,K)
    r["most_coherent_group_vote"] = acc_of(gv[np.arange(len(X)), gbest] + 0.01 * allv, Y, rng)
    # least-coherent-but-not-perfect heuristics for T4: exclude groups that agree in all Q questions
    w = np.where(n == env.Q, 0.0, n.astype(float))
    gb2 = (w + 1e-6 * rng.random(n.shape)).argmax(-1)
    r["most_coherent_excluding_perfect"] = acc_of(gv[np.arange(len(X)), gb2] + 0.01 * allv, Y, rng)
    # coherence-weighted vote: each group's vote weighted by its count
    r["coherence_weighted_vote"] = acc_of(np.einsum("ng,ngqk->nqk", n.astype(float), gv) + 0.01 * allv, Y, rng)
    r["bayes_full"] = float((post["post_y"].argmax(-1) == Y).mean())
    r["bayes_own_question"] = float((post["own_y"].argmax(-1) == Y).mean())
    res[t] = r
    P(f"{t}: " + " | ".join(f"{k} {v:.4g}" for k, v in r.items()))

# M9 closed-form likelihood ratios (alpha=.8, K=3)
a, K = 0.8, 3
e = (1 - a) / (K - 1)
pa_rel = a * a + (K - 1) * e * e; pa_noise = 1 / K
P(f"M9: P(pair agrees | reliable) = {pa_rel:.4f}; | noise = {pa_noise:.4f}; | colluding = 1")
P(f"    LR(agree) = {pa_rel / pa_noise:.4f}; LR(disagree) = {(1 - pa_rel) / (1 - pa_noise):.4f}")
res["M9"] = dict(p_agree_rel=pa_rel, p_agree_noise=pa_noise, LR_agree=pa_rel / pa_noise, LR_dis=(1 - pa_rel) / (1 - pa_noise))
# T4 non-monotone: posterior of group g being reliable as a function of its own count with others fixed
env = B.Env("T4", K=3, Q=6)
rows = []
for ng in range(7):
    for nh in [6]:
        for nk in [2]:
            # log P(counts | R) up to constants; R=(rel, coll)
            def lp(R):
                rel, col = R; cnt = {0: ng, 1: nh, 2: nk}
                if cnt[col] != 6: return -np.inf
                third = [g for g in range(3) if g not in R][0]
                return cnt[rel] * np.log(pa_rel) + (6 - cnt[rel]) * np.log(1 - pa_rel) + cnt[third] * np.log(pa_noise) + (6 - cnt[third]) * np.log(1 - pa_noise)
            l = np.array([lp(R) for R in env.R_values]); pr = np.exp(l - l.max()); pr /= pr.sum()
            prel0 = sum(pr[i] for i, R in enumerate(env.R_values) if R[0] == 0)
            rows.append((ng, float(prel0)))
P("    T4, group1 count 6, group2 count 2: P(group 0 reliable) vs group-0 count: " + ", ".join(f"{a}:{b:.3f}" for a, b in rows))
res["M9_T4_curve"] = rows
open(os.path.join(os.path.dirname(__file__), "..", "logs", "math_checks2.txt"), "w").write("\n".join(out) + "\n")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "logs", "math_checks2.json"), "w"), indent=1)
