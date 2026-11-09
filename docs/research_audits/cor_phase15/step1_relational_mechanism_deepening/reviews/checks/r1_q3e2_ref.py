import os, sys, json
D="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"
sys.path.insert(0, D+"/src")
import numpy as np, tasks3 as T3, gen_family as G
from collections import defaultdict, Counter
d = T3.sample_adapt(20000, np.random.default_rng(9_700_000)); X, Y = d["X"], d["Y"]
b = T3.bayes_adapt(X)["post_y"]; s = T3.bayes_adapt_static(X)
A = np.stack([(X[:, :, a] == X[:, :, bb]) for a, bb in G.GROUPS], -1); cnt = A.sum(1)
nonconst=[]
hs, ho, sizes, nclass, hy_b, hy_s = [], [], [], 0, [], []
for q in range(6):
    grp = defaultdict(list)
    for i in range(len(X)): grp[tuple(cnt[i]) + tuple(X[i, q])].append(i)
    for v in grp.values():
        if len(v) < 2: continue
        am = b[v, q].argmax(-1)
        if len(set(am)) > 1:
            nclass += 1; sizes.append(len(v))
            sa = s[v, q].argmax(-1)
            if len(set(sa))>1:
                nonconst.append(float(np.ptp(s[v,q],0).max()))
            hs += list(sa == am)
            maj = Counter(am).most_common(1)[0][0]; ho += list(am == maj)
            hy_b += list(am == Y[v, q]); hy_s += list(sa == Y[v, q])
print("nonconst classes",len(nonconst),"max spread",max(nonconst) if nonconst else 0)
print(json.dumps(dict(n=len(hs), n_classes=nclass, median_class_size=float(np.median(sizes)),
  static_follows_exact=float(np.mean(hs)), best_count_rule_follows_exact=float(np.mean(ho)),
  bayes_acc_in_set=float(np.mean(hy_b)), static_acc_in_set=float(np.mean(hy_s))), indent=1))
