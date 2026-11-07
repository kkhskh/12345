import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'src')
import mediator as MD, generators as G, ckpt
for task, arch, hook in [("REL-XNOR","TF1","resid_mid0"),("REL-GATE","TF1","resid_mid0"),("REL-XNOR","TF2","resid_mid1"),("REL-XNOR","TF2","resid_mid0"),("REL-GATE","TF2","resid_mid1"),("REL-MIS","TF1","resid_mid0")]:
    P=ckpt.load(f"checkpoints/historical/{MD.CODE[task]}_{arch}_s0.npz")
    site=MD.Site(P,task,hook)
    A,C=G.matched_pairs(task,1000,np.random.default_rng(5001)); Ae,Ce=G.matched_pairs(task,600,np.random.default_rng(5002))
    ev=G.sample(task,3000,np.random.default_rng(5004))
    U,k,_=MD.s_rel(site,A,C)
    gc=MD.GridCoords(site,U)
    res={kind: MD.surrogate_test(site,U,gc,kind,ev,Ae,Ce) for kind in ["exact","additive","linear","aggregate"]}
    print(task,arch,hook,"k",k,"IIA",MD.iia(site,Ae,Ce,U)[0],"addR2",round(gc.additive_r2(),3),{k_:(round(v['acc'],3),round(v['pair_flip'],3)) for k_,v in res.items()})
