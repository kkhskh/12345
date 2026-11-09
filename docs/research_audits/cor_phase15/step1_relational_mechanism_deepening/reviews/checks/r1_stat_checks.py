import sys, os, numpy as np, json
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening/src"; sys.path.insert(0,B)
import tasks3 as T3, tasks_ext as TX
K,Q=3,6; MASK=TX.MASK; P=T3.PERMS; AP=T3.ALL_PAIRS; TRP=T3.TRAIN_PAIRS; HOP=T3.HOLDOUT_PAIRS
comp=lambda a,b:T3.composite_index([a],[b])[0]
print("PERMS",[tuple(p) for p in P])
# 1. composite structure under 24-pair prior
from collections import Counter
for s in range(6):
    print("sigma",s,"train taus",[t for (a,t) in TRP if a==s],"comps",sorted(Counter(comp(s,t) for (a,t) in TRP if a==s).items()))
def MI(pairs,f,g):
    n=len(pairs); J=Counter((f(p),g(p)) for p in pairs); A=Counter(f(p) for p in pairs); Bc=Counter(g(p) for p in pairs)
    return sum(c/n*np.log2((c/n)/(A[a]/n*Bc[b]/n)) for (a,b),c in J.items())
for nm,ps in (("all",AP),("train",TRP),("holdout",HOP)):
    print(nm,"H(comp)",MI(ps,lambda p:comp(*p),lambda p:comp(*p)),"I(comp;sigma)",MI(ps,lambda p:comp(*p),lambda p:p[0]),"I(comp;tau)",MI(ps,lambda p:comp(*p),lambda p:p[1]),
          "max P(comp|sigma) avg",np.mean([max(Counter(comp(*p) for p in ps if p[0]==s).values())/sum(1 for p in ps if p[0]==s) for s in range(6)]),
          "max P(comp|tau) avg",np.mean([max(Counter(comp(*p) for p in ps if p[1]==t).values())/sum(1 for p in ps if p[1]==t) for t in range(6)]))
def restricted_bayes(X,pairs):
    idx=[AP.index(pq) for pq in pairs]; J=TX.trans_joint(X)[:,:,idx]; lM=np.log(J.sum(-1))
    loo=np.stack([lM[:,[k for k in range(Q) if k!=q]].sum(1) for q in range(Q)],1)
    w=np.exp(loo-loo.max(-1,keepdims=True)); w/=w.sum(-1,keepdims=True); py=(w[...,None]*J).sum(2); return py/py.sum(-1,keepdims=True), lM
out={}
for nm,PS in (("train_eps",TRP),("holdout_eps",HOP)):
    d=T3.sample_trans_pairs(6000,np.random.default_rng(9_950_000),PS); X,Y,typ=d["X"],d["Y"],d["typ"]; c=typ==2
    mC=np.where(((X[:,:,0]==MASK)&(X[:,:,2]!=MASK))[...,None]&(np.arange(6)>=4)[None,None],MASK,X)
    mA=np.where((X[:,:,4]==MASK)[...,None]&(np.arange(6)<2)[None,None],MASK,X)
    r={}
    for pn,pp in (("prior24",TRP),("prior36",AP),("prior12",HOP)):
        for xn,XX in (("intact",X),("mask_tau",mC),("mask_sigma",mA)):
            py,_=restricted_bayes(XX,pp); r[f"{pn}_{xn}_Conly"]=float((py.argmax(-1)==Y)[c].mean())
    # restricted MAP ceilings (full 6-question posterior) for probes
    for pn,pp in (("prior24",TRP),("prior36",AP),("prior12",HOP)):
        idx=[AP.index(pq) for pq in pp]; J=TX.trans_joint(X)[:,:,idx]; lM=np.log(J.sum(-1)).sum(1); po=np.exp(lM-lM.max(-1,keepdims=True)); po/=po.sum(-1,keepdims=True)
        Ws=np.zeros((6000,6)); Wt=np.zeros((6000,6)); Wc=np.zeros((6000,6))
        for j,(a,b) in enumerate(pp): Ws[:,a]+=po[:,j]; Wt[:,b]+=po[:,j]; Wc[:,comp(a,b)]+=po[:,j]
        cm=T3.composite_index(d["sigma"],d["tau"])
        r[f"MAPceil_{pn}"]=dict(sigma=float((Ws.argmax(-1)==d["sigma"]).mean()),tau=float((Wt.argmax(-1)==d["tau"]).mean()),composite=float((Wc.argmax(-1)==cm).mean()))
        r[f"MAPceil_{pn}_test2000"]=dict(sigma=float((Ws.argmax(-1)==d["sigma"])[4000:].mean()),tau=float((Wt.argmax(-1)==d["tau"])[4000:].mean()),composite=float((Wc.argmax(-1)==cm)[4000:].mean()))
    out[nm]=r
print(json.dumps(out,indent=1))
