import sys, numpy as np
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_computation_atlas"
sys.path.insert(0,B+"/src")
import gen_family as G
shifts={"rho0.9":(G.Fam(pc=1.0,rho=0.9),6_200_000),"rho0.7":(G.Fam(pc=1.0,rho=0.7),6_200_001),"pc0":(G.Fam(pc=0.0),6_200_002)}
for sn,(f,seed) in shifts.items():
    d=G.sample(f,6000,np.random.default_rng(seed)); X,Y=d["X"],d["Y"]
    tp=G.posterior(f,X)["post_y"]; tb=tp.argmax(-1)
    A=np.stack([(X[:,:,a]==X[:,:,b]) for a,b in G.GROUPS],-1).sum(1)
    onsup=(A==6).any(-1)   # some group could be a rho=1 colluder
    ref=None
    for rho in [0.99,0.999,0.9999,0.99999]:
        pp=G.posterior(G.Fam(pc=1.0,rho=rho),X)["post_y"]; pb=pp.argmax(-1)
        if rho==0.999: ref=pb; refp=pp
        dis=pb!=tb
        print(sn,f"plugin rho={rho}: acc {(pb==Y).mean():.4f} n_disagree_vs_true {dis.sum()} frac of disagreements in off-support episodes {(~onsup[:,None].repeat(6,1))[dis].mean():.3f}",
              "argmax change vs rho0.999:", (pb!=ref).sum() if ref is not None else "-")
    dis=ref!=tb
    srt=np.sort(tp,-1); mt=np.log(srt[...,-1])-np.log(srt[...,-2]); sp=np.sort(refp,-1); mp=np.log(sp[...,-1])-np.log(sp[...,-2])
    # margin of true Bayes between its top and the plugin's choice
    lt=np.log(tp); gap_true=np.take_along_axis(lt,tb[...,None],-1)[...,0]-np.take_along_axis(lt,ref[...,None],-1)[...,0]
    lp=np.log(refp); gap_plug=np.take_along_axis(lp,ref[...,None],-1)[...,0]-np.take_along_axis(lp,tb[...,None],-1)[...,0]
    print(sn,"on disagreements: median log-gap true(top vs plugin choice)",np.median(gap_true[dis]).round(3),"frac<0.2",(gap_true[dis]<0.2).mean().round(3),
          "| median log-gap plugin",np.median(gap_plug[dis]).round(3),"frac<0.2",(gap_plug[dis]<0.2).mean().round(3), "| episodes w/ disagreement",len(np.unique(np.where(dis)[0])))
