import sys, numpy as np
sys.path.insert(0,'/home/claude/12345/docs/research_audits/cor_phase15/step1_mechanism_identification_continuation/src')
from program_b import qk3, run
m,e=qk3(final_ln=True); mo,eo=qk3(final_ln=True,qk_off=True)
rng=np.random.default_rng(5); bad=0; n=0; ex=None
for _ in range(4000):
    zA=rng.normal(0,2,3); zB=rng.normal(0,2,3)
    k=int(zA.argmax()); j=int(np.argsort(zA)[-2])
    t=run(m,e,zA,zB); to=run(mo,eo,zA,zB)
    dm=(t.logits[k]-t.logits[j])-(to.logits[k]-to.logits[j])
    vm=float(t.VO[1][0][1] @ (m.WU[:,k]-m.WU[:,j]))
    pred=np.sign(t.pattern[1][0][2,1]-to.pattern[1][0][2,1])*np.sign(vm)
    if abs(dm)>1e-9:
        n+=1
        if pred!=np.sign(dm): bad+=1; ex=ex or (zA.round(3).tolist(),zB.round(3).tolist(),float(dm),float(pred))
print("B-TM sign prediction under final LN (zero bias):",bad,"failures /",n,"; example",ex)
