import itertools, numpy as np, sys, os
sys.path.insert(0, '/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening/src')
import tasks3 as T3, tasks_ext as TX
P = [tuple(p) for p in itertools.permutations(range(3))]
def comp(t,s): return tuple(t[s[i]] for i in range(3))
# T3: one AB pair (a->b) and one BC pair (b'->c), beta=1
res=set()
for a in range(3):
  for b in range(3):
    for b2 in range(3):
      for c in range(3):
        S=[s for s in P if s[a]==b]; T=[t for t in P if t[b2]==c]
        res.add((b==b2, len({comp(t,s) for s in S for t in T})))
print("T3 one-pair-each: (chained, #composite candidates):", sorted(res))
# certainty with one pair each: AB (0->0), BC (0->0), scored C-only x_C=0
S=[s for s in P if s[0]==0]; T=[t for t in P if t[0]==0]
print("preimages of 0 under composites:", {tuple(i for i in range(3) if comp(t,s)[i]==0) for s in S for t in T})
# T1 factorisation check: posterior over 36 pairs is an outer product (product prior)
rng=np.random.default_rng(1); d=T3.sample_trans_pairs(2000,rng); J=TX.trans_joint(d["X"]); lM=np.log(J.sum(-1))
loo=lM[:,1:].sum(1); w=np.exp(loo-loo.max(-1,keepdims=True)); w/=w.sum(-1,keepdims=True); W=w.reshape(-1,6,6)
print("max |W - outer(marg)|:", np.abs(W-W.sum(2)[:,:,None]*W.sum(1)[:,None,:]).max())
# Restricted (training) prior: does the T1 'consequence' hold? remove C reports in BC questions
idx=[T3.ALL_PAIRS.index(p) for p in T3.TRAIN_PAIRS]
d=T3.sample_trans_pairs(6000,np.random.default_rng(9_950_000),T3.TRAIN_PAIRS); X,Y,typ=d["X"],d["Y"],d["typ"]
for nm,Xx in (("full",X),("no C in BC",None)):
  if Xx is None:
    Xx=X.copy(); bc=(X[:,:,0]==T3.MASK)&(X[:,:,2]!=T3.MASK); Xx[:,:,4]=np.where(bc,T3.MASK,X[:,:,4]); Xx[:,:,5]=np.where(bc,T3.MASK,X[:,:,5])
  for pri,ii in (("product",list(range(36))),("train-restricted",idx)):
    J=TX.trans_joint(Xx)[:,:,ii]; lM=np.log(J.sum(-1))
    loo=np.stack([lM[:,[k for k in range(6) if k!=q]].sum(1) for q in range(6)],1)
    w=np.exp(loo-loo.max(-1,keepdims=True)); w/=w.sum(-1,keepdims=True); py=(w[...,None]*J).sum(2)
    acc=(py.argmax(-1)==Y)
    print(nm,pri,{t:round(float(acc[typ==i].mean()),3) for i,t in enumerate(["AB","BC","C"])})
