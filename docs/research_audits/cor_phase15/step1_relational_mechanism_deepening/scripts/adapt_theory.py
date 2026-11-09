"""Adaptive-source theory (Workstream F), exact Bayes.  Writes logs/adapt/theory.json.
 (1) ADAPT (change-point reliability; src/tasks3.py): Bayes vs the static T1 reference; count insufficiency (episodes with
     equal per-group agreement COUNTS but different posteriors); the posterior lag after a change: P(true current reliable
     group | X_{-q}) as a function of the distance from q to the change point.
 (2) TRIGGERED COPIER: groups (reliable g, copier h, noise).  Per question: if the reliable pair DISAGREES, the copier pair
     copies one common uniform value w.p. rho_t; otherwise it reports independently uniform.  Compared with a STATIC
     colluder with the same marginal agreement rate (copy w.p. rho_s every question, rho_s chosen to match):
     (a) exact per-question likelihoods; (b) identifiability of the trigger: posterior P(triggered | X) under a 50/50
     mixture of the two mechanisms; (c) decision cost of assuming the wrong mechanism (Bayes acc, plug-in acc).
 (3) CAMOUFLAGED COPIER: copier agreement rate equal to the honest rate a_rel (atlas I2) and triggered on nothing ->
     exactly observationally equivalent to an honest pair (proved in ADAPTIVE_SOURCE_THEORY.md); checked numerically."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, tasks3 as T3, gen_family as G
K, Q = 3, 6; out = {}; alpha = 0.8; e = (1 - alpha) / (K - 1); ys = np.arange(K)
# (1)
rng = np.random.default_rng(9_700_000); d = T3.sample_adapt(20000, rng); X, Y = d["X"], d["Y"]
b = T3.bayes_adapt(X); s = T3.bayes_adapt_static(X); cfg = np.array([T3.ADAPT_CFGS[c] for c in d["cfg"]])
ch = cfg[:, 2] < 6
out["acc"] = dict(bayes=float((b["post_y"].argmax(-1) == Y).mean()), static=float((s.argmax(-1) == Y).mean()),
                  bayes_changed=float((b["post_y"].argmax(-1) == Y)[ch].mean()), static_changed=float((s.argmax(-1) == Y)[ch].mean()),
                  bayes_unchanged=float((b["post_y"].argmax(-1) == Y)[~ch].mean()), static_unchanged=float((s.argmax(-1) == Y)[~ch].mean()))
A = np.stack([(X[:, :, a] == X[:, :, bb]) for a, bb in G.GROUPS], -1); cnt = A.sum(1)
from collections import defaultdict
grp = defaultdict(list)
for i in range(len(X)): grp[tuple(cnt[i]) + tuple(X[i, 0])].append(i)
sp = [np.abs(b["post_y"][v, 0] - b["post_y"][v, 0].mean(0)).max() for v in grp.values() if len(v) > 1]
out["count_insufficiency"] = dict(max_spread=float(max(sp)), median_spread=float(np.median(sp)), n_classes=len(sp))
# posterior lag: P(group active at q is called reliable at q), by signed distance q - c
lag = defaultdict(list)
J = T3._adapt_joint(X); lM = np.log(J.sum(-1)); lp = np.log(T3.ADAPT_PRIOR)
for q in range(Q):
    loo = lM[:, [k for k in range(Q) if k != q]].sum(1) + lp; w = np.exp(loo - loo.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    act = np.array([[g1 if q < c else g2 for (g1, g2, c) in T3.ADAPT_CFGS]])      # active group under each cfg at q
    for i in np.where(ch)[0]:
        g_true = cfg[i, 0] if q < cfg[i, 2] else cfg[i, 1]; lag[int(q - cfg[i, 2])].append(w[i, act[0] == g_true].sum())
out["posterior_active_group_by_distance_q_minus_c"] = {k: float(np.mean(v)) for k, v in sorted(lag.items())}
print(out, flush=True)
# (2) triggered vs static copier.  Configurations: (g, h) ordered, mechanism m in {trig, stat}.
a_rel = alpha ** 2 + (1 - alpha) ** 2 / (K - 1); rho_t = 0.9
# copier agreement rate under trigger: P(rel disagrees) * (rho_t + (1-rho_t)/K) + P(rel agrees) * 1/K
a_trig = (1 - a_rel) * (rho_t + (1 - rho_t) / K) + a_rel / K; rho_s = (a_trig - 1 / K) / (1 - 1 / K)
def q_lik(xr, xc, y_vals, mech):
    """P(x_rel pair, x_cop pair | y) for each y: (..., K)"""
    pa = np.where(xr[..., 0, None] == y_vals, alpha, e); pb = np.where(xr[..., 1, None] == y_vals, alpha, e); prel = pa * pb
    same_c = (xc[..., 0] == xc[..., 1]); rel_dis = xr[..., 0] != xr[..., 1]
    rho = np.where(rel_dis, rho_t, 0.0) if mech == "trig" else np.full(xr.shape[:-1], rho_s)
    pc = rho * same_c / K + (1 - rho) / K ** 2
    return prel * pc[..., None]
def sample_mech(n, mech, rng):
    gh = [(g, h) for g in range(3) for h in range(3) if g != h]; ci = rng.integers(0, 6, n); Y = rng.integers(0, K, (n, Q)); X = rng.integers(0, K, (n, Q, 6))
    for i, (g, h) in enumerate(gh):
        s_ = np.where(ci == i)[0]
        for src in (2 * g, 2 * g + 1):
            ok = rng.random((len(s_), Q)) < alpha; X[s_[:, None], np.arange(Q)[None], src] = np.where(ok, Y[s_], (Y[s_] + rng.integers(1, K, (len(s_), Q))) % K)
        dis = X[s_, :, 2 * g] != X[s_, :, 2 * g + 1]; rho = np.where(dis, rho_t, 0.0) if mech == "trig" else np.full(dis.shape, rho_s)
        cop = rng.random(dis.shape) < rho; c = rng.integers(0, K, dis.shape)
        X[s_, :, 2 * h] = np.where(cop, c, X[s_, :, 2 * h]); X[s_, :, 2 * h + 1] = np.where(cop, c, X[s_, :, 2 * h + 1])
    return X, Y, ci
gh = [(g, h) for g in range(3) for h in range(3) if g != h]
def posterior(X, mechs):
    Js = []
    for mech in mechs:
        for (g, h) in gh:
            xr = np.stack([X[:, :, 2 * g], X[:, :, 2 * g + 1]], -1); xc = np.stack([X[:, :, 2 * h], X[:, :, 2 * h + 1]], -1)
            Js.append(q_lik(xr, xc, ys, mech) / K ** 2 / K)      # noise pair 1/K^2, prior on y 1/K
    J = np.stack(Js, 2); lM = np.log(J.sum(-1)); C_ = J.shape[2]
    loo = np.stack([lM[:, [k for k in range(Q) if k != q]].sum(1) for q in range(Q)], 1)
    w = np.exp(loo - loo.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True); py = (w[..., None] * J).sum(2); py /= py.sum(-1, keepdims=True)
    tot = lM.sum(1); pc = np.exp(tot - tot.max(-1, keepdims=True)); pc /= pc.sum(-1, keepdims=True); return py, pc
R2 = dict(a_rel=a_rel, a_copier_triggered=a_trig, rho_static_matched=rho_s)
for true in ("trig", "stat"):
    X2, Y2, _ = sample_mech(8000, true, np.random.default_rng(9_710_000)); py_mix, pc_mix = posterior(X2, ("trig", "stat"))
    py_t, _ = posterior(X2, ("trig",)); py_s, _ = posterior(X2, ("stat",))
    R2[f"true_{true}"] = dict(P_triggered_mean=float(pc_mix[:, :6].sum(1).mean()), acc_mixture_bayes=float((py_mix.argmax(-1) == Y2).mean()),
                              acc_assume_trig=float((py_t.argmax(-1) == Y2).mean()), acc_assume_static=float((py_s.argmax(-1) == Y2).mean()))
out["triggered_copier"] = R2; print(R2, flush=True)
# (3) camouflaged copier: static copier with agreement rate a_rel -> per-question law of the pair equals the honest pair's marginal
rho_c = (a_rel - 1 / K) / (1 - 1 / K)
xa, xb = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
honest = (np.where(xa[..., None] == ys, alpha, e) * np.where(xb[..., None] == ys, alpha, e)).mean(-1)
cam = rho_c * (xa == xb) / K + (1 - rho_c) / K ** 2
out["camouflaged_copier"] = dict(rho=rho_c, max_abs_diff_pair_law=float(np.abs(honest - cam).max()))
print(out["camouflaged_copier"]); json.dump(out, open(os.path.join(HERE, "..", "logs", "adapt", "theory.json"), "w"), indent=1)
