import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, sys.argv[1])
import learned_analysis as LA
rng=np.random.default_rng(23)
def eqsum_test(P, lim, n=4000):
    S=rng.uniform(-lim,lim,n); a1=rng.uniform(-lim,lim,n); b1=S-a1; a2=rng.uniform(-lim,lim,n); b2=S-a2
    ok=(np.abs(b1)<=lim)&(np.abs(b2)<=lim); S,a1,b1,a2,b2=S[ok],a1[ok],b1[ok],a2[ok],b2[ok]
    p=np.zeros(len(S))
    m1=LA.fwd_margin(P,'T2',LA.inp_bin(a1,b1,p)); m2=LA.fwd_margin(P,'T2',LA.inp_bin(a2,b2,p))
    agree_vs_conf=(np.sign(a1*b1)!=np.sign(a2*b2))
    flips=np.mean(np.sign(m1)!=np.sign(m2)); rel=np.mean((m1-m2)**2)/np.var(np.concatenate([m1,m2]))
    # D_sum in package's style with bins over [-2lim,2lim] of width ~0.1
    A,B=np.meshgrid(np.linspace(-lim,lim,121),np.linspace(-lim,lim,121)); A=A.ravel(); B=B.ravel()
    m=LA.fwd_margin(P,'T2',LA.inp_bin(A,B,np.zeros(len(A))))
    nb=int(40*lim/1.0); ds,_,_=LA.dsum(m,A+B,nb=nb,lo=-2*lim,hi=2*lim)
    i=np.argmax(np.abs(m1-m2))
    return rel, flips, ds, (a1[i],b1[i],a2[i],b2[i],m1[i],m2[i])
for arch in ['TF1','TF2','TF2AO']:
    for seed in range(5):
        P=LA.load('T2',arch,seed)
        out=[]
        for lim in [3,6,10,30]:
            rel,fl,ds,ex=eqsum_test(P,lim)
            out.append(f'lim{lim}: eqsum relMSE={rel:.1e} flips={fl:.4f} Dsum={ds:.1e}')
        print(arch,seed,' | '.join(out), '| worst@30', np.round(ex,3))
