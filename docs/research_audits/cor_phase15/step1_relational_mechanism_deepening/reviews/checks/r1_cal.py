import sys, itertools
sys.path.insert(0, "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening/src")
import numpy as np, tasks3 as T3
d = T3.sample_adapt(40000, np.random.default_rng(123)); X, Y = d["X"], d["Y"]
r = T3.bayes_adapt(X); P = r["post_y"]
print("acc", (P.argmax(-1)==Y).mean(), "mean P(true y)", np.take_along_axis(P, Y[...,None], -1).mean(), "mean max", P.max(-1).mean())
# brute force for 3 episodes, question 2: P(y|X) via full joint over cfg
K=3; a=0.8; e=(1-a)/2
for i in range(3):
    q=2; num=np.zeros(K)
    for ci,(g1,g2,c) in enumerate(T3.ADAPT_CFGS):
        pr=T3.ADAPT_PRIOR[ci]; lik=1.0
        for k in range(6):
            g = g1 if k < c else g2
            def lk(y):
                v=1/K
                for gg in range(3):
                    s0,s1=X[i,k,2*gg],X[i,k,2*gg+1]
                    v*= ((a if s0==y else e)*(a if s1==y else e)) if gg==g else 1/K**2
                return v
            if k==q: continue
            lik*=sum(lk(y) for y in range(K))
        for y in range(K): num[y]+=pr*lik*lk(y) if False else 0
        # recompute own term
        g = g1 if q < c else g2
        for y in range(K):
            v=1/K
            for gg in range(3):
                s0,s1=X[i,q,2*gg],X[i,q,2*gg+1]
                v*= ((a if s0==y else e)*(a if s1==y else e)) if gg==g else 1/K**2
            num[y]+=pr*lik*v
    print(np.round(num/num.sum(),6), np.round(P[i,q],6))
print("cfg posterior true-cfg mean", r["post_cfg"][np.arange(len(X)), d["cfg"]].mean())
