import sys; sys.path.insert(0,'/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening/src')
import tasks3 as T3, tasks_ext as TX, numpy as np
AP=T3.ALL_PAIRS; MASK=TX.MASK; Q=6
def rb(X,pairs):
    idx=[AP.index(pq) for pq in pairs]; J=TX.trans_joint(X)[:,:,idx]; lM=np.log(J.sum(-1))
    loo=np.stack([lM[:,[k for k in range(Q) if k!=q]].sum(1) for q in range(Q)],1)
    w=np.exp(loo-loo.max(-1,keepdims=True)); w/=w.sum(-1,keepdims=True); py=(w[...,None]*J).sum(2); return py/py.sum(-1,keepdims=True)
for nm,PS in (("train",T3.TRAIN_PAIRS),("holdout",T3.HOLDOUT_PAIRS)):
    d=T3.sample_trans_pairs(6000,np.random.default_rng(9_950_000),PS); X,Y,typ=d["X"],d["Y"],d["typ"]; c=typ==2
    Xo=X.copy(); Xo[typ!=2]=MASK   # remove every other question entirely
    print(nm,"prior24, no other-question evidence, C-only acc:",(rb(Xo,T3.TRAIN_PAIRS).argmax(-1)==Y)[c].mean(),
          " prior36:",(rb(Xo,AP).argmax(-1)==Y)[c].mean())
