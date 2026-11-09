import sys; sys.path.insert(0,'/tmp/r1')
import cprime_fixed as cf, numpy as np
r=0.05; ok=True; okm=True
for a in np.linspace(0.2,3.0,57):
    for bx in [(a,a+0.05,-(a+0.05),-r*a),(r*a,a+0.05,-a-0.05,-a)]:
        pos,neg,und=cf.certify(bx,min_w=5e-3,max_boxes=40000)
        if pos or und: ok=False; print("Q+- fail",bx)
    for bx in [(-(a+0.05),-a,r*a,a+0.05),(-(a+0.05),-r*a,a,a+0.05)]:
        pos,neg,und=cf.certify(bx,min_w=5e-3,max_boxes=40000)
        if pos or und: okm=False; print("Q-+ fail",bx); break
print("Q+- full cover certified:",ok," Q-+ mirror certified:",okm)
