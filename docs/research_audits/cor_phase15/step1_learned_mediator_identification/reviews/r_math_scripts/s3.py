import sys, itertools, numpy as np
sys.path.insert(0,"/tmp/rv/w/src")
import mediator_sign_test as T
TW=T.additive_twin(); RF,embRF=T.relation_free()
gg=np.concatenate([-np.linspace(3,.1,15),np.linspace(.1,3,15)])
for name,f in (("C'",lambda a,b:T.mg(T.C,a,b)),("TWIN",lambda a,b:T.mg(TW,a,b)),("RF",lambda a,b:T.mg(RF,a,b,emb=embRF))):
    M=np.array([[f(a,b) for b in gg] for a in gg])
    for tol in (0,1e-9,1e-6,1e-3):
        v=0
        for i,j in itertools.combinations(range(30),2):
            d=M[i]-M[j]; s=np.sign(d[np.abs(d)>tol]); v+=(s>0).any() and (s<0).any()
        # also columns
        vc=0
        for i,j in itertools.combinations(range(30),2):
            d=M[:,i]-M[:,j]; s=np.sign(d[np.abs(d)>tol]); vc+=(s>0).any() and (s<0).any()
        print(name,tol,v,vc)
    # twin: interaction share
    Ms=M.mean(1,keepdims=True)+M.mean(0,keepdims=True)-M.mean()
    print(name,"interaction share",((M-Ms)**2).sum()/((M-M.mean())**2).sum())
# S1 robustness: twin with shifted readout - is S1 definition-dependent? C' gate q+c
print("---magnitudes")
for name,mm in (("TWIN",TW),("C'",T.C)):
    M=np.array([[T.mg(mm,a,b) for b in gg] for a in gg]); sd=M.std(); w=0;n=0
    for i,j in itertools.combinations(range(30),2):
        d=M[:,i]-M[:,j]
        if (d>0.05*sd).any() and (d<-0.05*sd).any(): n+=1; w=max(w,min(d.max(),-d.min())/sd)
    print(name,"col reversals >0.05SD",n,"max SD",w)
