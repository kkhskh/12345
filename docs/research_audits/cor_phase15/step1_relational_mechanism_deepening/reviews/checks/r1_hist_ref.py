import sys, json
D="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"
sys.path.insert(0, D+"/src")
import numpy as np, tasks3 as T3
d = T3.sample_adapt(20000, np.random.default_rng(9_700_000)); X, Y = d["X"], d["Y"]
b = T3.bayes_adapt(X)["post_y"]; s = T3.bayes_adapt_static(X)
rng = np.random.default_rng(9_700_100); fs, fo, n = [], [], 0
for q in range(6):
    oth = [k for k in range(6) if k != q]; Xp = X.copy()
    perm = np.stack([rng.permutation(oth) for _ in range(len(X))]); Xp[:, oth] = X[np.arange(len(X))[:, None], perm]
    bp = T3.bayes_adapt(Xp)["post_y"]; sp = T3.bayes_adapt_static(Xp)
    def uq(Pm): ss = np.sort(Pm, -1); return ss[..., -1] - ss[..., -2] > 1e-6
    sel = (b[:, q].argmax(-1) != bp[:, q].argmax(-1)) & uq(b[:, q]) & uq(bp[:, q])
    fs += list((sp[:, q].argmax(-1) == bp[:, q].argmax(-1))[sel]); fo += list((b[:, q].argmax(-1) == bp[:, q].argmax(-1))[sel])
    print(q, sel.sum(), "static changes:", int((sp[:,q].argmax(-1)!=s[:,q].argmax(-1))[sel].sum()))
print(json.dumps(dict(n=len(fs), static_follows_new_exact=float(np.mean(fs)), old_exact_follows_new=float(np.mean(fo)))))
