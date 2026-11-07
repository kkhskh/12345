import sys, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1]+'/../../step1_causal_bridge_deep_investigation/src')
import program_b as pb, cbmath
rng=np.random.default_rng(5)
for fl in [False, True]:
  for kw in [dict(), dict(lam=10.0, kB=4.0, wU=1.0), dict(lam=1.0,kB=0.5,wU=3.0,rho=0.5,kappa=0.2)]:
    m,e=pb.qk3(final_ln=fl,**kw); mo,eo=pb.qk3(final_ln=fl,qk_off=True,**kw)
    bad=0; n=0; maxerr=0; ex=None
    for i in range(4000):
        zA=rng.normal(0,2,3)*rng.uniform(0.1,5); zB=rng.normal(0,2,3)*rng.uniform(0.1,5)
        k,j=rng.choice(3,2,replace=False)
        t=pb.run(m,e,zA,zB); to=pb.run(mo,eo,zA,zB)
        dm=(t.logits[k]-t.logits[j])-(to.logits[k]-to.logits[j])
        aB,aBo=t.pattern[1][0][2,1],to.pattern[1][0][2,1]
        wB=float(t.VO[1][0][1]@(m.WU[:,k]-m.WU[:,j]))
        wA=float(t.VO[1][0][0]@(m.WU[:,k]-m.WU[:,j])); wR=float(t.VO[1][0][2]@(m.WU[:,k]-m.WU[:,j]))
        al=t.pattern[1][0][2]; wbar=0.0 if (abs(wA)<1e-12 and abs(wR)<1e-12) else (al[0]*wA+al[2]*wR)/(1-al[1])
        pred=(aB-aBo)*(wB-wbar)
        if not fl: maxerr=max(maxerr,abs(pred-dm))
        if abs(dm)>1e-9:
            n+=1
            if np.sign(pred)!=np.sign(dm):
                bad+=1
                if ex is None: ex=(zA.round(4).tolist(),zB.round(4).tolist(),int(k),int(j),float(dm),float(pred))
    print('final_ln',fl,kw,'sign errors',bad,'/',n,'max |pred-dm| (no LN)',maxerr,'example',ex)
# exact reproducible counterexample
m,e=pb.qk3(final_ln=True); mo,eo=pb.qk3(final_ln=True,qk_off=True)
zA=np.array([1.8839,-2.7601,-1.4939]); zB=np.array([7.3496,-11.1922,6.7408]); k,j=0,2
t=pb.run(m,e,zA,zB); to=pb.run(mo,eo,zA,zB)
wB=float(t.VO[1][0][1]@(m.WU[:,k]-m.WU[:,j]))
print('CE: alphaB full',t.pattern[1][0][2,1],'uniform',to.pattern[1][0][2,1],'w_B',wB,
      'margin full',t.logits[k]-t.logits[j],'uniform',to.logits[k]-to.logits[j])
m2,e2=pb.qk3(); mo2,eo2=pb.qk3(qk_off=True)
t2=pb.run(m2,e2,zA,zB); to2=pb.run(mo2,eo2,zA,zB)
print('same input, no final LN: margin full',t2.logits[k]-t2.logits[j],'uniform',to2.logits[k]-to2.logits[j])
