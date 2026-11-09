"""ADAPT evaluation (Workstream F3, Q3-E).  Usage: eval_adapt.py CKPT NAME.  Theory stream 9_700_000 (20,000 episodes).
 (1) accuracy on changed / unchanged episodes vs exact Bayes and the static T1 reference (Q3-E1);
 (2) Q3-E2: count classes (equal per-group agreement counts and equal own reports at q) whose exact argmax is not constant
     across the class: fraction of such questions where the model's argmax equals exact Bayes (static Bayes is constant
     within a class by construction);
 (3) history intervention: permute the ORDER of the other five questions (each question's six reports move together;
     counts unchanged, so the static reference is unchanged); on questions whose unique exact-Bayes argmax changes, fraction
     where the model follows the new exact answer, and the model's change rate."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, tasks3 as T3, train3 as TR, ckpt as C, gen_family as G
from collections import defaultdict
ck, name = sys.argv[1], sys.argv[2]; p = C.load(ck)
d = T3.sample_adapt(20000, np.random.default_rng(9_700_000)); X, Y = d["X"], d["Y"]
cfg = np.array([T3.ADAPT_CFGS[c] for c in d["cfg"]]); ch = cfg[:, 2] < 6
P = TR.answer_probs(p, X); b = T3.bayes_adapt(X)["post_y"]; s = T3.bayes_adapt_static(X)
acc = lambda Pm, m: float((Pm.argmax(-1) == Y)[m].mean())
res = dict(ckpt=ck, model_changed=acc(P, ch), bayes_changed=acc(b, ch), static_changed=acc(s, ch),
           model_unchanged=acc(P, ~ch), bayes_unchanged=acc(b, ~ch), static_unchanged=acc(s, ~ch), model_all=acc(P, np.ones(len(X), bool)), bayes_all=acc(b, np.ones(len(X), bool)))
res["Q3E1"] = bool(res["model_changed"] - res["static_changed"] >= 0.015 and abs(res["model_unchanged"] - res["bayes_unchanged"]) <= 0.02)
A = np.stack([(X[:, :, a] == X[:, :, bb]) for a, bb in G.GROUPS], -1); cnt = A.sum(1); hits = []
for q in range(6):
    grp = defaultdict(list)
    for i in range(len(X)): grp[tuple(cnt[i]) + tuple(X[i, q])].append(i)
    for v in grp.values():
        if len(v) < 2: continue
        am = b[v, q].argmax(-1)
        if len(set(am)) > 1: hits += list((P[v, q].argmax(-1) == am))
res["Q3E2_n"] = len(hits); res["Q3E2_model_follows_exact"] = float(np.mean(hits)); res["Q3E2"] = bool(res["Q3E2_model_follows_exact"] >= 0.6)
rng = np.random.default_rng(9_700_100); fol, chg = [], []
for q in range(6):
    oth = [k for k in range(6) if k != q]; Xp = X.copy()
    perm = np.stack([rng.permutation(oth) for _ in range(len(X))]); Xp[:, oth] = X[np.arange(len(X))[:, None], perm]
    bp = T3.bayes_adapt(Xp)["post_y"]; Pp = TR.answer_probs(p, Xp)
    def uq(Pm): ss = np.sort(Pm, -1); return ss[..., -1] - ss[..., -2] > 1e-6
    sel = (b[:, q].argmax(-1) != bp[:, q].argmax(-1)) & uq(b[:, q]) & uq(bp[:, q])
    fol += list((Pp[:, q].argmax(-1) == bp[:, q].argmax(-1))[sel]); chg += list((Pp[:, q].argmax(-1) != P[:, q].argmax(-1))[sel])
res["history_intervention"] = dict(n=len(fol), model_follows_new_exact=float(np.mean(fol)), model_change_rate=float(np.mean(chg)))
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(HERE, "..", "logs", "adapt", f"eval_{name}.json"), "w"), indent=1)
