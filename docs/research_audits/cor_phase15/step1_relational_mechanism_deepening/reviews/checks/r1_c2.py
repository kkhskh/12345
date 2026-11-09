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
            pred=np.where(np.isfinite(rc), rr>rc, rr>1)    # if no coll mass: helps iff rel rises
            ok=np.isfinite(dm)&(np.abs(dm)>1e-9)&np.isfinite(rr)
        tot+=ok.sum(); mis+=((dm>0)!=pred)[ok].sum()
    print(nm,"D1 specialisation: n",tot,"sign mismatches",mis)
# (b) (U) is not sufficient when two groups depend on y: two reliable pairs (structure outside family) -- marginal per question
K=3; al=0.8; e=(1-al)/(K-1)
p=lambda x,y: al if x==y else e
def m_two_rel(x):  # x=(a,b,c,d)
    return sum(p(x[0],y)*p(x[1],y)*p(x[2],y)*p(x[3],y) for y in range(K))/K
print("two-rel-groups, both agree, same value vs different value:", m_two_rel((0,0,0,0)), m_two_rel((0,0,1,1)))
# (c) (U) necessity: an example violating (U) where counts stay sufficient: noise biased q, coll common value ~ q, rel pair under uniform Y
# check whether role log-lik ratio depends on value: here ratio coll/noise for agreement on value v = rho*q_v+(1-rho)q_v^2 over q_v^2 -> depends on v; counterexample holds.
# (d) I4 sampling uncertainty: paired bootstrap over episodes
for nm,f in {"T1":G.Fam(pc=0.0),"T4":G.Fam(pc=1.0,rho=1.0)}.items():
    fp=G.Fam(pc=f.pc,rho=f.rho,perm=True); d=G.sample(fp,1500,np.random.default_rng(8)); Xp=d["X"]
    pu=G.posterior_unknown(f,Xp)["post_y"]; inv=np.argsort(d["perms"],-1); Xk=np.take_along_axis(Xp,inv[:,None,:],-1); pk=G.posterior(f,Xk)["post_y"]
    ck=(pk.argmax(-1)==d["Y"]).mean(1); cu=(pu.argmax(-1)==d["Y"]).mean(1)
    ek=pk.max(-1).mean(1); eu=pu.max(-1).mean(1)   # Rao-Blackwellised expected accuracy
    bs=rng.integers(0,1500,(2000,1500))
    diff=(ck-cu)[bs].mean(1); diffe=(ek-eu)[bs].mean(1)
    print(nm,"known",ck.mean().round(4),"unknown",cu.mean().round(4),"diff",(ck-cu).mean().round(4),"95%CI",np.percentile(diff,[2.5,97.5]).round(4),
          "| expected-acc diff",(ek-eu).mean().round(4),np.percentile(diffe,[2.5,97.5]).round(4), "SE acc known", ck[bs].mean(1).std().round(4))
