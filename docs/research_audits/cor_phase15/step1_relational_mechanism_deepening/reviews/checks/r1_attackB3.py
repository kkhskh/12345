# 3-candidate generalization of the QK gate with vector evidence (candidate-score vectors)
import sys, numpy as np
sys.path.insert(0, "/tmp/pkgcopy_r1/src")
from cbmath import *
d=16; Bs=mean_zero_basis(d)
FA=Bs[0:3]; FB=Bs[3:6]; FQ=Bs[6:9]; FU=Bs[9:12]; anS,anR,mkA=Bs[12],Bs[13],Bs[14]
dh=3
def build(lam=4.0, qk_off=False, wQ=2.0, wU=0.6, kB=1.5, L1=12.0, prior=np.zeros(3)):
    WQ1=np.zeros((1,d,dh)); WQ1[0]=L1*np.outer(anR,[1,0,0]); WK1=np.zeros((1,d,dh)); WK1[0]=np.outer(mkA,[1,0,0])
    WV1=np.zeros((1,d,dh)); WV1[0]=FA.T            # value = A's 3-d evidence
    WO1=np.zeros((1,dh,d)); WO1[0]=wQ*FQ+wU*FU      # copy to query dirs and direct vote
    l1=Layer(WQ=WQ1,WK=WK1,WV=WV1,WO=WO1,bO=np.zeros(d),Win=np.zeros((d,1)),bin=np.zeros(1),Wout=np.zeros((1,d)),bout=np.zeros(d),mlp_on=False)
    WQ2=np.zeros((1,d,dh)); WQ2[0]=(0 if qk_off else lam)*FQ.T
    WK2=np.zeros((1,d,dh)); WK2[0]=FB.T
    WV2=np.zeros((1,d,dh)); WV2[0]=FB.T
    WO2=np.zeros((1,dh,d)); WO2[0]=kB*FU
    l2=Layer(WQ=WQ2,WK=WK2,WV=WV2,WO=WO2,bO=np.zeros(d),Win=np.zeros((d,1)),bin=np.zeros(1),Wout=np.zeros((1,d)),bout=np.zeros(d),mlp_on=False)
    return Model(layers=[l1,l2],WU=FU.T.copy(),bU=prior.copy())
def emb(a,b,kappa=1.5,mu=3.0,rho=3.0):
    return np.stack([np.array(a)@FA+kappa*anS+mu*mkA, np.array(b)@FB+kappa*anS, rho*anR])
def run(mm,a,b):
    tr=forward(mm,emb(a,b),2); return tr.logits, tr.pattern[1][0][2][1]
cases={"top differs, dot>0 (both oppose c3)":((1,0,-1),(0,1,-1)),
       "same top c1, dot<0":((1,0.8,-1.8),(1,-1.8,0.8)),
       "same top c1, dot>0":((1,0,-1),(1,-1,0))}
for prior in (np.zeros(3), np.array([0,0.15,0])):
  g=build(prior=prior); f=build(qk_off=True,prior=prior)
  print("prior",prior)
  for nm,(a,b) in cases.items():
    lg,w=run(g,a,b); lf,wf=run(f,a,b)
    print(f"  {nm}: a={a} b={b} dot={np.dot(a,b):+.2f} | gate w_B={w:.3f} logits={np.round(lg,3)} argmax c{np.argmax(lg)+1} | fixed w_B={wf:.3f} logits={np.round(lf,3)} argmax c{np.argmax(lf)+1}")
