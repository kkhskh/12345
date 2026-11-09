import sys
sys.path.insert(0, "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening/src")
import numpy as np, tasks3 as T3
d = T3.sample_adapt(20000, np.random.default_rng(9_700_000)); X = d["X"]
s = T3.bayes_adapt_static(X); rng = np.random.default_rng(9_700_100)
q=0; oth=[1,2,3,4,5]; Xp=X.copy(); perm=np.stack([rng.permutation(oth) for _ in range(len(X))]); Xp[:,oth]=X[np.arange(len(X))[:,None],perm]
sp=T3.bayes_adapt_static(Xp); print("max |static - static_perm| at q0:", np.abs(s[:,0]-sp[:,0]).max())
ss=np.sort(s[:,0],-1); print("frac static ties at q0:", np.mean(ss[:,-1]-ss[:,-2]<1e-9))
