import sys, itertools, numpy as np
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_computation_atlas"
sys.path.insert(0,B+"/src")
import gen_family as G
rng=np.random.default_rng(0)
# (a) D1 specialisation test: edit in group g (question k) raises g's LOO count; g votes c1 at q (both sources report c1).
for nm,f in {"T4":G.Fam(),"rho0.8pc0.5":G.Fam(rho=0.8,pc=0.5),"rho0.7pc0.2k0.3":G.Fam(rho=0.7,pc=0.2,kappa=0.3)}.items():
    d=G.sample(f,1500,np.random.default_rng(1)); X0=d["X"]; tot=0; mis=0; mis2=0
    p0=G.posterior(f,X0)
    for q,k,g in itertools.product(range(6),range(6),range(3)):
        if q==k: continue
        a,b=G.GROUPS[g]
        sel=(X0[:,k,a]!=X0[:,k,b])&(X0[:,q,a]==X0[:,q,b])        # create agreement in k; g votes v=X[q,a] at q
        if not sel.any(): continue
        X1=X0[sel].copy(); X1[:,k,b]=X1[:,k,a]; p1=G.posterior(f,X1)
        v=X0[sel,q,a]; other=(v+1)%f.K
        with np.errstate(all="ignore"):
            dm=(np.log(p1["post_y"][np.arange(len(v)),q,v])-np.log(p1["post_y"][np.arange(len(v)),q,other]))-(np.log(p0["post_y"][sel,q][np.arange(len(v)),v])-np.log(p0["post_y"][sel,q][np.arange(len(v)),other]))
            pi0=p0["pi_minus"][sel,q]; pi1=p1["pi_minus"][:,q]
            def role(pi,r):
                idx=[i for i,(gg,h) in enumerate(f.configs) if (r=="rel" and gg==g) or (r=="coll" and h==g)]
                return pi[:,idx].sum(-1)
            rr=role(pi1,"rel")/role(pi0,"rel"); rc=role(pi1,"coll")/role(pi0,"coll")
            rn=(1-role(pi1,"rel"))/(1-role(pi0,"rel")); pred=rr>rn    # if no coll mass: helps iff rel rises
            ok=np.isfinite(dm)&(np.abs(dm)>1e-9)&np.isfinite(rr)
        tot+=ok.sum(); mis+=((dm>0)!=pred)[ok].sum()
    print(nm,"D1 specialisation: n",tot,"sign mismatches",mis)
# (b) (U) is not sufficient when two groups depend on y: two reliable pairs (structure outside family) -- marginal per question
K=3; al=0.8; e=(1-al)/(K-1)
p=lambda x,y: al if x==y else e
