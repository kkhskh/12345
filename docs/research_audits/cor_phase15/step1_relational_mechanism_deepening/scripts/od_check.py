"""O/D weighted-voting coincidence (mandate §11): exact check of the reviewer-proposed condition OD1' for independent
symmetric sources.  For S sources with log-odds weights w_s, K candidates, every report vector x in [K]^S:
  Bayes score of c = sum_{s: x_s = c} w_s ;  D_c = #{s : x_s != c}.
  NSR (no strict reversal): for all x and all c, c': D_c < D_c'  =>  score_c > score_c'   (equivalently Bayes never strictly
  prefers a candidate with strictly larger discrepancy).  ARGMIN: the Bayes argmax lies in the arg-min-D set.
  COND (OD1'): w(P) < w(R) for all disjoint P, R with |P| < |R| and |P| + |R| <= S.
Also the band sufficient condition w_max/w_min < (m+1)/m, m = floor((S-1)/2).
Random weight vectors (accuracies uniform in (1/K, 0.99)) and near-boundary vectors; reports the confusion between COND and
NSR / ARGMIN, and counterexamples.  Writes logs/od_check.json."""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(__file__)
def weights(acc, K): return np.log(acc * (K - 1) / (1 - acc))
def nsr_argmin(w, K):
    S = len(w); Xs = np.array(list(itertools.product(range(K), repeat=S)))
    sc = np.stack([(w[None] * (Xs == c)).sum(1) for c in range(K)], 1); D = np.stack([(Xs != c).sum(1) for c in range(K)], 1)
    nsr = True
    for c in range(K):
        for c2 in range(K):
            if c == c2: continue
            if ((D[:, c] < D[:, c2]) & (sc[:, c] <= sc[:, c2] + 1e-12)).any(): nsr = False
    am = sc.argmax(1); argmin = bool((D[np.arange(len(Xs)), am] == D.min(1)).all())
    return nsr, argmin
def cond(w):
    S = len(w); idx = range(S)
    for p in range(S + 1):
        for r in range(p + 1, S + 1 - p):
            for P in itertools.combinations(idx, p):
                rest = [i for i in idx if i not in P]
                for R in itertools.combinations(rest, r):
                    if w[list(P)].sum() >= w[list(R)].sum() - 1e-12: return False
    return True
out = {}
for S, K in ((4, 3), (5, 3), (6, 3), (5, 4)):
    rng = np.random.default_rng(10_000 + S * 10 + K); conf = {}; ex = []
    m = (S - 1) // 2
    for t in range(300):
        if t % 2 == 0: acc = rng.uniform(1 / K + 0.01, 0.99, S)
        else:      # near the band boundary
            base = rng.uniform(0.6, 0.9); acc = np.clip(base + rng.normal(0, 0.06, S), 1 / K + 0.01, 0.995)
        w = weights(acc, K); c_ = cond(w); n_, a_ = nsr_argmin(w, K); band = w.max() / w.min() < (m + 1) / m
        key = f"cond={c_},nsr={n_},argmin={a_},band={band}"; conf[key] = conf.get(key, 0) + 1
        if c_ != n_ and len(ex) < 5: ex.append(dict(acc=[round(float(a), 4) for a in acc], cond=c_, nsr=n_, argmin=a_))
    out[f"S{S}_K{K}"] = dict(confusion=conf, counterexamples=ex); print(S, K, conf, ex[:2], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "od_check.json"), "w"), indent=1)
