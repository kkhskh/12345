"""§1-§3 mathematical validation (run before any training). Writes logs/math_checks.{txt,json}.

M1  E1 4-source example of the mandate, reproduced (analytic code in bayes.e1_example) and independently by
    brute-force enumeration over (R, Y) with a separately written likelihood.
M2  E0 additivity: exact log odds == sum of per-source log-likelihood ratios.
M3  Six-source, multi-question version: two episodes with IDENTICAL per-question global vote histograms but
    different exact posteriors (explicit construction + random search).
M4  Information requirements for T0-T5 (exact Bayes accuracy of restricted statistics on 20k episodes):
      full episode | own question only | per-question global histograms (all questions) |
      own question + group within-pair agreement counts of the other questions | oracle R known
M5  Sufficiency theorem check: under uniform Y priors, P(R | X_{-q}) is a function of the per-group within-pair
    agreement counts of the other questions (T1 and T4). Verified by grouping episodes by counts.
M6  Sampler equivalence: vectorised sampler vs reference loop sampler (moment comparison).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bayes as B  # noqa: E402

LOG, OUT = [], {}


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def brute_force_example(prior=(0.025, 0.025, 0.95), acc=0.88):
    """Independent implementation: enumerate R in {0: A reliable, 1: B reliable} and Y; generic per-source likelihood."""
    K = 3
    def lik_source(x, y, reliable):
        if not reliable:
            return 1.0 / K
        return acc if x == y else (1.0 - acc) / (K - 1)
    res = []
    for reports in [((0, 0), (1, 1)), ((0, 1), (0, 1))]:
        w = np.zeros(K)
        for R in (0, 1):
            for y in range(K):
                p = 0.5 * prior[y]
                for gi, grp in enumerate(reports):
                    for x in grp:
                        p *= lik_source(x, y, gi == R)
                w[y] += p
        res.append(w / w.sum())
    return res


def counts(X):
    """per question & group: 1 if the two sources of the group agree. X (N,Q,6) -> (N,Q,3)"""
    return np.stack([(X[:, :, a] == X[:, :, b]).astype(int) for a, b in B.GROUPS], -1)


def main():
    rng = np.random.default_rng(0)
    # M1
    a1, b1 = B.e1_example()
    a2, b2 = brute_force_example()
    log("M1 E1 example (prior .025,.025,.95; reliable acc .88):")
    log(f"   A=(0,0),B=(1,1): {np.round(a1, 5)}  independent: {np.round(a2, 5)}")
    log(f"   A=(0,1),B=(0,1): {np.round(b1, 5)}  independent: {np.round(b2, 5)}")
    assert np.allclose(a1, a2) and np.allclose(b1, b2)
    assert np.allclose(np.round(a1, 5), [0.42523, 0.42523, 0.14954]) and np.allclose(np.round(b1, 5), [0.21782, 0.21782, 0.56436])
    OUT["M1"] = dict(config1=a1.tolist(), config2=b1.tolist())
    # M2
    e = B.e0_additivity_check(rng)
    log(f"M2 E0 additivity: max |exact log odds - additive sum| = {e:.2e}")
    OUT["M2"] = e

    # M3 six-source matched-histogram construction (T1, K=3, Q=6)
    env = B.Env("T1", Q=6)
    # episode A: group 0 agrees on every question (consistent); episode B: same per-question histograms, but the
    # agreeing pair is split across groups (each group disagrees internally)
    qa = []; qb = []
    for q in range(6):
        y = q % 3; o1, o2 = (y + 1) % 3, (y + 2) % 3
        qa.append([y, y, o1, o2, o1, o2])          # histogram: y:2, o1:2, o2:2 ; group0 agrees
        qb.append([y, o1, y, o2, o1, o2])          # same histogram; no group agrees
    XA = np.array([qa]); XB = np.array([qb])
    pa, pb = B.episode_posterior(env, XA), B.episode_posterior(env, XB)
    log("M3 six-source, Q=6: identical per-question global histograms (2,2,2) in every question")
    log(f"   episode A (group 0 coherent): P(R|X) = {np.round(pa['post_R'][0], 4)}; question 0 posterior {np.round(pa['post_y'][0, 0], 4)}")
    log(f"   episode B (no coherent group): P(R|X) = {np.round(pb['post_R'][0], 4)}; question 0 posterior {np.round(pb['post_y'][0, 0], 4)}")
    OUT["M3"] = dict(A=dict(post_R=pa["post_R"][0].tolist(), post_y0=pa["post_y"][0, 0].tolist()),
                     B=dict(post_R=pb["post_R"][0].tolist(), post_y0=pb["post_y"][0, 0].tolist()))
    # random search: pairs with same per-question histograms and max posterior difference
    d = B.sample_fast(env, 20000, np.random.default_rng(3))
    H = np.stack([np.stack([(d["X"][:, q, :] == k).sum(1) for k in range(3)], -1) for q in range(6)], 1)  # N,Q,K
    key = [tuple(h.ravel()) for h in H]
    p = B.episode_posterior(env, d["X"])
    from collections import defaultdict
    grp = defaultdict(list)
    for i, k in enumerate(key):
        grp[k].append(i)
    best = 0.0; arg = None
    for k, idx in grp.items():
        if len(idx) > 1:
            P0 = p["post_y"][idx, 0]
            dd = np.abs(P0[:, None, :] - P0[None, :, :]).sum(-1).max()
            if dd > best:
                best = dd; arg = idx
    log(f"   random search over 20k episodes: max L1 difference of question-0 posteriors at identical histograms = {best:.4f}")
    OUT["M3_random_max_L1"] = float(best)

    # M4 information requirements
    log("M4 exact Bayes accuracy of restricted statistics (20k episodes each, K=3, Q=6):")
    OUT["M4"] = {}
    for task in ["T0", "T1", "T4", "T5"]:
        env = B.Env(task, Q=6)
        d = B.sample_fast(env, 20000, np.random.default_rng(10 + ["T0", "T1", "T4", "T5"].index(task)))
        p = B.episode_posterior(env, d["X"])
        full = float((p["post_y"].argmax(-1) == d["Y"]).mean())
        own = float((p["own_y"].argmax(-1) == d["Y"]).mean())
        # histogram-only Bayes (exact): P(H|y,R) by enumerating all K^6 report vectors
        allx = np.array(list(itertools.product(range(3), repeat=6)))
        hist = np.stack([(allx == k).sum(1) for k in range(3)], 1)
        hkeys = {tuple(h): i for i, h in enumerate(np.unique(hist, axis=0))}
        hid = np.array([hkeys[tuple(h)] for h in hist])
        nH = len(hkeys)
        Rs = env.R_values
        PH = np.zeros((len(Rs), 3, nH))
        for r, R in enumerate(Rs):
            for y in range(3):
                lik = B.q_lik(env, allx, R)[:, y]
                np.add.at(PH[r, y], hid, lik)
        Hq = np.stack([np.stack([(d["X"][:, q, :] == k).sum(1) for k in range(3)], -1) for q in range(6)], 1)
        hidq = np.array([[hkeys[tuple(h)] for h in Hq[:, q]] for q in range(6)]).T          # N,Q
        Lq = PH[:, :, hidq]                                   # nR, K, N, Q
        Mq = Lq.mean(1)                                       # nR, N, Q  (uniform prior on y)
        logM = np.log(np.maximum(Mq, 1e-300))
        tot = logM.sum(2)
        hist_acc = []
        for q in range(6):
            lm = tot - logM[:, :, q]
            w = np.exp(lm - lm.max(0)); w /= w.sum(0)
            post = np.einsum("rn,rkn->nk", w, Lq[:, :, :, q])
            hist_acc.append((post.argmax(1) == d["Y"][:, q]).mean())
        hist_acc = float(np.mean(hist_acc))
        # oracle R known
        if Rs[0] is not None:
            orc = []
            for q in range(6):
                L = np.stack([B.q_lik(env, d["X"][:, q, :], Rs[r]) for r in range(len(Rs))], 0)  # nR,N,K
                orc.append((L[d["ridx"], np.arange(len(d["ridx"]))].argmax(1) == d["Y"][:, q]).mean())
            oracle = float(np.mean(orc))
        else:
            oracle = full
        OUT["M4"][task] = dict(full=full, own_question_only=own, histograms_all_questions=hist_acc, oracle_R=oracle)
        log(f"   {task}: full {full:.4f} | own question only {own:.4f} | per-question histograms (all q) {hist_acc:.4f} | oracle R {oracle:.4f}")

    # M5 sufficiency of group agreement counts for P(R | X_{-q})
    log("M5 sufficiency: P(R | X_{-q}) as a function of the other questions' per-group agreement counts")
    OUT["M5"] = {}
    for task in ["T1", "T4"]:
        env = B.Env(task, Q=6)
        d = B.sample_fast(env, 20000, np.random.default_rng(30))
        p = B.episode_posterior(env, d["X"])
        C = counts(d["X"])                                     # N,Q,3
        q = 0
        cnt = C[:, 1:, :].sum(1)                               # counts over questions != 0
        ks = [tuple(c) for c in cnt]
        from collections import defaultdict
        g = defaultdict(list)
        for i, k in enumerate(ks):
            g[k].append(i)
        spread = max(np.abs(p["pi_minus"][idx, q] - p["pi_minus"][idx, q].mean(0)).max() for idx in g.values())
        OUT["M5"][task] = dict(n_count_cells=len(g), max_within_cell_deviation=float(spread))
        log(f"   {task}: {len(g)} distinct count vectors; max deviation of P(R|X_-q) within a count cell = {spread:.2e}")

    # M6 sampler equivalence
    log("M6 sampler equivalence (vectorised vs reference loop), K=3 Q=6, 20k episodes each:")
    OUT["M6"] = {}
    for task in ["T0", "T1", "T4", "T5"]:
        env = B.Env(task, Q=6)
        a = B.sample_fast(env, 20000, np.random.default_rng(40)); b = B.sample_episodes(env, 20000, np.random.default_rng(41))
        def stats(dd):
            C = counts(dd["X"]).mean((0, 1))
            accs = [(dd["X"][:, :, s] == dd["Y"]).mean() for s in range(6)]
            return np.concatenate([C, accs])
        diff = float(np.abs(stats(a) - stats(b)).max())
        OUT["M6"][task] = diff
        log(f"   {task}: max |moment difference| = {diff:.4f} (Monte-Carlo scale ~0.005)")

    (ROOT / "logs" / "math_checks.txt").write_text("\n".join(LOG) + "\n")
    (ROOT / "logs" / "math_checks.json").write_text(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main()
