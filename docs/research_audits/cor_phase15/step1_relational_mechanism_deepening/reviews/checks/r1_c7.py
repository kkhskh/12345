import sys, numpy as np
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_computation_atlas"
sys.path.insert(0,B+"/src")
import gen_family as G
f=G.Fam(K=3,rho=0.7,kappa=0.4,pc=1.0)
xa,xb=np.meshgrid(np.arange(3),np.arange(3),indexing="ij")
for r in ["rel","coll","noise"]:
    L=G.pair_lik(f,xa,xb,r)  # 3,3,K
    m=L.mean(-1)
    print(r,"norm per y",L.sum((0,1)).round(12),"marginal agree-vals",np.diag(m).round(5),"offdiag",m[~np.eye(3,dtype=bool)].round(5))
# sampler vs likelihood: config (0,1): group0 rel, group1 coll
d=G.sample(f,200000,np.random.default_rng(0),cfg_idx=np.full(200000,f.configs.index((0,1))))
X,Y=d["X"][:,0],d["Y"][:,0]
for r,(a,b) in [("rel",(0,1)),("coll",(2,3))]:
    emp=np.zeros((3,3)); np.add.at(emp,(X[Y==0,a],X[Y==0,b]),1); emp/=emp.sum()
    print(r,"max |emp - lik| given y=0:",np.abs(emp-G.pair_lik(f,xa,xb,r)[...,0]).max().round(4))
rs=(G.agree_probs(G.Fam())["rel"]-1/3)/(2/3); print("rho* at kappa=0, alpha=.8, K=3:",rs)
f2=G.Fam(pc=1.0,rho=rs,kappa=0.0); X=G.sample(f2,3000,np.random.default_rng(4))["X"]; J=G.joint(f2,X).sum(-1); cf=f2.configs; sw=[cf.index((h,g)) for g,h in cf]
print("kappa=0 exchange max |dloglik|",np.abs(np.log(J).sum(1)-np.log(J[:,:,sw]).sum(1)).max())
