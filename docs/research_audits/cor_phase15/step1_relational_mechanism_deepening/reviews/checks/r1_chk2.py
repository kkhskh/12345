import numpy as np, itertools
K=3; al=0.8; e=(1-al)/(K-1); a_rel=al**2+(K-1)*e**2; rc=(a_rel-1/K)/(1-1/K)
print("a_rel",a_rel,"rho_c",rc)
# (i) Markov copier: copy indicator stationary rc, persistence lam; P(agree_q, agree_q+1)
g={1:1.0,0:1/K}
for lam in (0.0,0.5,0.8):
    p11=lam+(1-lam)*rc; p01=(1-lam)*rc
    T={1:{1:p11,0:1-p11},0:{1:p01,0:1-p01}}; pi={1:rc,0:1-rc}
    pp=sum(pi[c]*T[c][c2]*g[c]*g[c2] for c in (0,1) for c2 in (0,1))
    print(f"Markov lam={lam}: P(copy_q)={rc:.3f} const; P(agree_q & agree_q+1)={pp:.4f} vs honest {a_rel**2:.4f}")
# (ii) copier copying w.p. r' only when the reliable pair AGREES; marginal copy prob = rc (constant per question)
rp=rc/a_rel; print("r' =",rp)
# per-question joint law of (rel pair agree, copier pair agree)
trig = {(1,1):a_rel*(rp+(1-rp)/K),(1,0):a_rel*(1-rp)*(1-1/K),(0,1):(1-a_rel)/K,(0,0):(1-a_rel)*(1-1/K)}
stat = {(i,j):(a_rel if i else 1-a_rel)*(a_rel if j else 1-a_rel) for i in (0,1) for j in (0,1)}
print("copier marginal agree (trig-on-agree):",trig[(1,1)]+trig[(0,1)]," honest:",a_rel)
print("joint trig-on-agree:",{k:round(v,4) for k,v in trig.items()}); print("joint static/honest-indep:",{k:round(v,4) for k,v in stat.items()})
# (iii) OD2, K=2: weights (10,4,4,4): COND fails (w1>w2+w3) but NSR holds for K=2
w=np.array([10,4,4,4.]); S=4; ok=True
for x in itertools.product(range(2),repeat=S):
    x=np.array(x); sc=[w[x==c].sum() for c in (0,1)]; D=[(x!=c).sum() for c in (0,1)]
    for c in (0,1):
        if D[c]<D[1-c] and not sc[c]>sc[1-c]: ok=False
print("K=2 NSR holds:",ok,"; COND fails since w1=10 >= w2+w3=8")
# same weights, K=3: NSR should fail (x = (0,1,1,2))
x=np.array([0,1,1,2]); print("K=3 x=",x,"scores",[w[x==c].sum() for c in range(3)],"D",[(x!=c).sum() for c in range(3)])
