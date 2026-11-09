import sys, numpy as np
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_computation_atlas"
sys.path.insert(0,B+"/src")
import gen_family as G
for seed in [8,9]:
  for nm,f in {"T1":G.Fam(pc=0.0),"T4":G.Fam(pc=1.0)}.items():
    fp=G.Fam(pc=f.pc,rho=f.rho,perm=True); d=G.sample(fp,1500,np.random.default_rng(seed)); Xp=d["X"]
    pu=G.posterior_unknown(f,Xp)["post_y"]; inv=np.argsort(d["perms"],-1); Xk=np.take_along_axis(Xp,inv[:,None,:],-1); pk=G.posterior(f,Xk)["post_y"]
    # perm check: X[..., j] = Xorig[..., perms[j]] ; inverse undo
    Y=d["Y"]
    print(seed,nm,"known acc %.4f E %.4f | unknown acc %.4f E %.4f | logscore known %.4f unknown %.4f"%((pk.argmax(-1)==Y).mean(),pk.max(-1).mean(),(pu.argmax(-1)==Y).mean(),pu.max(-1).mean(),
          np.log(np.take_along_axis(pk,Y[...,None],-1)).mean(),np.log(np.take_along_axis(pu,Y[...,None],-1)).mean()))
# check undo of permutation is right: known-pairing posterior on unpermuted sampler output
f=G.Fam(pc=0.0); d=G.sample(f,1500,np.random.default_rng(8)); pk=G.posterior(f,d["X"])["post_y"]
print("no-perm T1 known acc %.4f E %.4f"%((pk.argmax(-1)==d["Y"]).mean(),pk.max(-1).mean()))
