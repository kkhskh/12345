import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'src')
import mediator as MD, generators as G, ckpt, model as M, jax.numpy as jnp
for task, arch, seed in [("REL-XNOR","TF1",0),("REL-XNOR","TF1",1),("REL-MIS","TF1",0),("REL-GATE","TF1",0)]:
    P=ckpt.load(f"checkpoints/historical/{MD.CODE[task]}_{arch}_s{seed}.npz")
    site=MD.Site(P,task,"resid_mid0")
    A,C=G.matched_pairs(task,1000,np.random.default_rng(5001))
    U,k,_=MD.s_rel(site,A,C)
    gc=MD.GridCoords(site,U)
    g=gc.g; Aa,Bb=np.meshgrid(g,g,indexing='ij')
    for p,Z in gc.C.items():
        R=Z-(Z.mean(1,keepdims=True)+Z.mean(0,keepdims=True)-Z.mean((0,1),keepdims=True))
        for j in range(Z.shape[-1]):
            r=R[...,j].ravel()
            if r.var()<1e-12: print(task,seed,p,"dim",j,"interaction ~0"); continue
            feats={"ab":(Aa*Bb).ravel(),"sign(ab)":np.sign(Aa*Bb).ravel(),"tanh(2ab)":np.tanh(2*Aa*Bb).ravel(),"sign(ab)min(|a|,|b|)":(np.sign(Aa*Bb)*np.minimum(abs(Aa),abs(Bb))).ravel()}
            r2={n:round(1-np.var(r-np.polyval(np.polyfit(f,r,1),f))/np.var(r),3) for n,f in feats.items()}
            print(task,arch,seed,"p",p,"dim",j,"interaction share",round(R[...,j].var()/Z[...,j].var(),3),r2)
    # pattern-freeze: replace layer-0 attention pattern with that of a fixed reference input -> additive?
    ref=np.asarray(M.forward(P,MD.CODE[task],MD.to_inp(task,dict(a=np.zeros(1),b=np.zeros(1),p=np.ones(1))),None))
    store={}
    def cap(n,arr):
        if n=="pattern0": store['p']=np.asarray(arr)
        return arr
    M.forward(P,MD.CODE[task],MD.to_inp(task,dict(a=np.array([0.0]),b=np.array([0.0]),p=np.array([1.0]))),cap)
    pat=store['p']
    def frozen(d):
        st={}
        def hk(n,arr):
            if n=="pattern0": return jnp.broadcast_to(jnp.asarray(pat),arr.shape)
            if n=="resid_mid0": st['x']=np.asarray(arr)
            return arr
        M.forward(P,MD.CODE[task],MD.to_inp(task,d),hk); return st['x'][:,-1,:]
    pvals=[-1.0,1.0] if task=="REL-GATE" else [0.0]
    for p in pvals:
        h=frozen(dict(a=Aa.ravel(),b=Bb.ravel(),p=np.full(Aa.size,p)))
        Zf=(h@U).reshape(61,61,-1)
        Rf=Zf-(Zf.mean(1,keepdims=True)+Zf.mean(0,keepdims=True)-Zf.mean((0,1),keepdims=True))
        print("   pattern-frozen: interaction share per dim", [float(round(Rf[...,j].var()/max(Zf[...,j].var(),1e-12),6)) for j in range(Zf.shape[-1])])
