"""Bayes references for the holdout run (C6): exact posterior with the prior restricted to the 24 TRAINING (sigma, tau)
pairs (the training distribution's Bayes), evaluated on training-pair and held-out-pair episodes (stream 9_950_000,
as eval_trans.py).  On held-out pairs the restricted-prior Bayes is misspecified by construction: it shows what a
learner that perfectly fits the training prior must do there.  Writes logs/trans/restricted_bayes.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, tasks3 as T3, tasks_ext as TX
K, Q = 3, 6; idx = [T3.ALL_PAIRS.index(pq) for pq in T3.TRAIN_PAIRS]; out = {}
for nm, PS in (("train_pairs", T3.TRAIN_PAIRS), ("holdout_pairs", T3.HOLDOUT_PAIRS)):
    d = T3.sample_trans_pairs(6000, np.random.default_rng(9_950_000), PS); X, Y, typ = d["X"], d["Y"], d["typ"]
    J = TX.trans_joint(X)[:, :, idx]; lM = np.log(J.sum(-1))
    loo = np.stack([lM[:, [k for k in range(Q) if k != q]].sum(1) for q in range(Q)], 1)
    w = np.exp(loo - loo.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True); py = (w[..., None] * J).sum(2); py /= py.sum(-1, keepdims=True)
    out[nm] = {t: float((py.argmax(-1) == Y)[typ == i].mean()) for i, t in enumerate(["AB", "BC", "C_only"])}
print(out); json.dump(out, open(os.path.join(HERE, "..", "logs", "trans", "restricted_bayes.json"), "w"), indent=1)
