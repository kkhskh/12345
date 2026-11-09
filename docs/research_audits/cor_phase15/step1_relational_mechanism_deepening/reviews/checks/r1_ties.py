import sys
sys.path.insert(0, "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening/src")
import numpy as np, tasks3 as T3, gen_family as G
from collections import defaultdict
d = T3.sample_adapt(20000, np.random.default_rng(9_700_000)); X = d["X"]
b = T3.bayes_adapt(X)["post_y"]
A = np.stack([(X[:, :, a] == X[:, :, bb]) for a, bb in G.GROUPS], -1); cnt = A.sum(1)
tie=0; tot=0; cls_tie_only=0; ncls=0
for q in range(6):
    grp = defaultdict(list)
    for i in range(len(X)): grp[tuple(cnt[i]) + tuple(X[i, q])].append(i)
    for v in grp.values():
        if len(v) < 2: continue
        am = b[v, q].argmax(-1)
        if len(set(am)) > 1:
            ncls+=1; ss=np.sort(b[v,q],-1); t=(ss[:,-1]-ss[:,-2])<1e-9; tie+=t.sum(); tot+=len(v)
            # does class remain mixed using only untied members
            if len(set(am[~t]))<=1: cls_tie_only+=1
print("items",tot,"exact-tied items",tie,"classes",ncls,"classes not mixed once ties removed",cls_tie_only)
