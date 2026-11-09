import sys; sys.path.insert(0,'/tmp/r1')
import cprime_fixed as cf, numpy as np
quads = {"Q++": (0, 3, 0, 3), "Q--": (-3, 0, -3, 0), "Q+-": (0, 3, -3, 0), "Q-+": (-3, 0, 0, 3)}
for n,box in quads.items():
    pos,neg,und=cf.certify(box,min_w=2e-3)
    # min |F| margin over certified boxes
    mins=[]
    for bx in pos: mins.append(cf.interval_m(cf.I(bx[0],bx[1]),cf.I(bx[2],bx[3])).lo)
    for bx in neg: mins.append(-cf.interval_m(cf.I(bx[0],bx[1]),cf.I(bx[2],bx[3])).hi)
    print(n, "pos",cf.area(pos)/9,"neg",cf.area(neg)/9,"und",cf.area(und)/9,"min cert margin",min(mins))
for n,sa,sb,want,tau in [("Q++",1,1,"pos",0.09),("Q--",-1,-1,"neg",0.0),("Q+-",1,-1,"neg",0.02),("Q-+",-1,1,"neg",0.02)]:
    a0,a1=sorted([sa*tau,sa*3.0]); b0,b1=sorted([sb*tau,sb*3.0])
    pos,neg,und=cf.certify((a0,a1,b0,b1),min_w=5e-3,max_boxes=60000)
    bad=(neg if want=="pos" else pos)+und
    print(n,tau,"certified" if not bad else "FAILS",len(bad))
