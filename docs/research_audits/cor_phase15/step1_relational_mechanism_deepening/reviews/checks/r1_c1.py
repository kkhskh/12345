import sys, itertools, numpy as np
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_computation_atlas"
sys.path.insert(0,B+"/src"); sys.path.insert(0,B+"/scripts")
import gen_family as G
# reuse helpers from bayes_boundaries without running it: copy
def role_ll(f, n, M):
    ap = G.agree_probs(f)
    def ll(a):
        with np.errstate(divide="ignore"):
            return n*np.log(a)+(M-n)*np.log1p(-a) if a<1 else np.where(n==M,0.0,-np.inf)
    return {r: ll(ap[r]) for r in ["rel","noise","coll"]}
def post(f,counts,M):
    lls=[role_ll(f,counts[...,g],M) for g in range(3)]; lp=[]
    for (g,h),pr in zip(f.configs,f.prior):
        s=np.log(pr)
        for k in range(3):
            r="rel" if k==g else ("coll" if k==h else "noise"); s=s+lls[k][r]
        lp.append(s)
    lp=np.stack(lp,-1); m=lp.max(-1,keepdims=True); w=np.exp(lp-m); return w/w.sum(-1,keepdims=True)
def prole(f,p,g,role):
    idx=[i for i,(a,b) in enumerate(f.configs) if (role=="rel" and a==g) or (role=="coll" and b==g) or (role=="noise" and a!=g and b!=g)]
    return p[...,idx].sum(-1)
# 1. Theorem U for all roles incl monotone extremes
viol={"rel":0,"coll":0,"noise":0}; mono={"top":0,"bot":0}; rows=0
for rho,pc,alpha,K,Q,kappa in itertools.product([1.0,0.9,0.7,0.5],[0.05,0.5,1.0],[0.6,0.8,0.95],[2,3,5],[3,6],[0.0,0.3,0.6]):
    f=G.Fam(K=K,Q=Q,alpha=alpha,rho=rho,pc=pc,kappa=kappa); ap=G.agree_probs(f); M=Q-1; n=np.arange(M+1)
    oth=np.array(list(itertools.product(range(M+1),repeat=2)))
    C=np.zeros((len(oth),M+1,3)); C[:,:,0]=n; C[:,:,1]=oth[:,None,0]; C[:,:,2]=oth[:,None,1]
    with np.errstate(invalid="ignore"): p=post(f,C,M)
    order=sorted(ap,key=ap.get)
    for role in ["rel","coll","noise"]:
        pr=prole(f,p,0,role)
        for row in pr:
            if not np.isfinite(row).all(): continue
            rows+=1; d=np.diff(row); s=np.sign(np.where(abs(d)<1e-12,0,d)); s=s[s!=0]
            if len(s)>1 and np.any((s[:-1]<0)&(s[1:]>0)): viol[role]+=1
            if role==order[-1] and np.any(d< -1e-12): mono["top"]+=1
            if role==order[0] and np.any(d> 1e-12): mono["bot"]+=1
print("U all roles: rows",rows,"qc viol",viol,"monotone-extreme viol",mono)
# 2. drop magnitudes in phase cells K3 Q6 a.8
for rho,pc in [(1,0.05),(0.8,0.2),(0.7,0.2),(0.5,1.0),(0.5,0.5),(0.8,0.05),(0.9,1.0)]:
    f=G.Fam(rho=rho,pc=pc); M=5; n=np.arange(6)
    oth=np.array(list(itertools.product(range(6),repeat=2)))
    C=np.zeros((36,6,3)); C[:,:,0]=n; C[:,:,1]=oth[:,None,0]; C[:,:,2]=oth[:,None,1]
    with np.errstate(invalid="ignore"): pr=prole(f,post(f,C,M),0,"rel")
    d=G.sample(f,3000,np.random.default_rng(5)); A=np.stack([(d["X"][:,:,a]==d["X"][:,:,b]) for a,b in G.GROUPS],-1)
    loo=(A.sum(1)[:,None,:]-A).reshape(-1,3); key=loo[:,1]*6+loo[:,2]; w=np.bincount(key,minlength=36)/len(key)
    drop=np.nan_to_num(pr[:,4]-pr[:,5]); 
    print(f"rho{rho} pc{pc}: frac drop {(w*(drop>1e-12)).sum():.3f} frac drop>0.05 {(w*(drop>0.05)).sum():.3f} wmean drop|drop {(w*drop*(drop>1e-12)).sum()/max((w*(drop>1e-12)).sum(),1e-12):.4f}  nan-rows wt {(w*~np.isfinite(pr).all(1)).sum():.3f}")
