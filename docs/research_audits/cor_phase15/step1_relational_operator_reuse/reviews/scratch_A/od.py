import sys, os, itertools, json
PKG="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_operator_reuse"
sys.path.insert(0, PKG+"/src")
import numpy as np, xvitask as T
K=3; XA=T.xall(6); IDX=np.array([243,81,27,9,3,1])
def flips(a):
    # all on-support flips with probability: return list of (src_idx, dst_idx, prob) per input
    g = XA[:, :a].sum(1)%K==0; part = XA[:, :a-1].sum(1)%K; op=(-part)%K
    out=[]
    for i,x in enumerate(XA):
        if g[i]:
            for r in (1,2):
                y=x.copy(); y[a-1]=(op[i]+r)%K; out.append((i,(y*IDX).sum(),0.5))
        else:
            y=x.copy(); y[a-1]=op[i]; out.append((i,(y*IDX).sum(),1.0))
    return out
def uniq(P,tol=1e-9): s=np.sort(P,-1); return s[...,-1]-s[...,-2]>tol
def FG(Pm, PB, px, a):
    # Pm: rule probs (ties as distributions). F_G with argmax; for ties use expected following
    num=den=0.0
    for i,j,w in flips(a):
        if not(uniq(PB[i]) and uniq(PB[j]) and PB[i].argmax()!=PB[j].argmax()): continue
        wt=px[i]*w; den+=wt; nb=PB[j].argmax()
        # prob model decision changes and equals new bayes argmax (independent tie-breaking draws)
        pchg = Pm[j][nb]*(1-Pm[i][nb])
        num+=wt*pchg
    return num/den
def tieP(score):  # argmin of score, uniform over ties
    m = score==score.min(1,keepdims=True); return m/m.sum(1,keepdims=True)
res={}
for c in ["2:0.9:0.9:0.3333333333333333","2:0.9:0.9:0.05","3:0.9:0.9:0.3333333333333333","3:0.9:0.9:0.05","2:0.9:0.55:0.05"]:
    a,b1,b0,nu=c.split(":"); a=int(a); b1,b0,nu=float(b1),float(b0),float(nu)
    px,PB=T.joint(a,b1,b0,nu)
    Dall=np.stack([(XA!=k).sum(1) for k in range(K)],1); Dev=np.stack([(XA[:,3:]!=k).sum(1) for k in range(K)],1)
    Pall=tieP(Dall); Pev=tieP(Dev)
    acc=lambda P: float((px[:,None]*PB*P).sum())
    # fraction of (input,flip) on which argmin D_all changes
    ch=np.mean([np.abs(Pall[i]-Pall[j]).max()>0 for i,j,w in flips(a)])
    # order-1 rule with heavy weight on last gate source + evidence vote
    sc = -(2.0*(XA[:,[a-1]]==np.arange(K)) + (XA[:,3:,None]==np.arange(K)).sum(1))
    P1=tieP(sc)
    # best accuracy over order-1 weighted votes on evidence only (grid)
    best=0
    for w in itertools.product(np.linspace(0,3,13),repeat=3):
        s=-sum(w[t]*(XA[:,3+t,None]==np.arange(K)) for t in range(3)); best=max(best,acc(tieP(s)))
    R=np.load(PKG+"/logs/refs_"+c.replace(":","_")+".npz")
    res[c]=dict(acc_ev=acc(Pev),acc_all=acc(Pall),acc_order1NLLopt_argmax=float((px*PB[np.arange(729),R["order1"].argmax(1)]).sum()),
      best_weighted_vote_acc=best, frac_flip_changes_minDall=float(ch), FG_minD_all=FG(Pall,PB,px,a), FG_minD_ev=FG(Pev,PB,px,a),
      FG_order1_lastgate=FG(P1,PB,px,a), FG_bayes=FG(np.eye(3)[PB.argmax(1)],PB,px,a))
    print(c,{k:round(v,4) for k,v in res[c].items()})
json.dump(res,open("/tmp/revA17/od.json","w"),indent=1)
