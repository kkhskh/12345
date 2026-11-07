import sys, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1]+'/../../step1_causal_bridge_deep_investigation/src')
import cbmath
from cprime import closed_form
from relation_free_equivalent import relation_free
mR,emb=relation_free()
tw=lambda a,b: cbmath.margin(cbmath.forward(mR,emb(a,b),3))
rng=np.random.default_rng(17)
def outside(a,b): return not ((a>0 and b>0 and min(a,b)<0.09) or (a*b<0 and min(abs(a),abs(b))<0.02))
for R in [3,5,10]:
    n=bad=0; ex=[]
    for a,b in rng.uniform(-R,R,(20000,2)):
        if not outside(a,b): continue
        n+=1; mc=closed_form(a,b)['m']; mt=tw(a,b)
        if np.sign(mc)!=np.sign(mt):
            bad+=1
            if len(ex)<3: ex.append((round(a,4),round(b,4),mc,mt))
    print(f'|a|,|b|<={R}: outside dead bands n={n} mismatches={bad}', ex)
# dense near dead band edges inside [-3,3]
n=bad=0; ex=[]
for _ in range(40000):
    s=rng.integers(4); big=rng.uniform(0.02,3); 
    if s==0: a,b=rng.uniform(0.09,0.3),rng.uniform(0.09,3)
    elif s==1: a,b=rng.uniform(0.09,3),rng.uniform(0.09,0.3)
    elif s==2: a,b=big,-rng.uniform(0.02,0.06)
    else: a,b=-rng.uniform(0.02,0.06),big
    if rng.random()<0.5 and s<2: a,b=-a,-b  # Q-- too
    n+=1; mc=closed_form(a,b)['m']; mt=tw(a,b)
    if np.sign(mc)!=np.sign(mt):
        bad+=1; ex.append((a,b,mc,mt))
print('edge sampling in [-3,3]: n',n,'mismatch',bad, ex[:3])
for a,b in [(10,-0.03),(50,-0.035),(-0.025,10),(3.5,-0.02)]:
    print((a,b),'C prime m',closed_form(a,b)['m'],'twin',tw(a,b))
# twin's own dead band: where is twin wrong vs the target rule within its band?
for a,b in [(0.5,-0.01),(1,-0.0036),(1,-0.01),(0.05,0.05),(0.01,0.01),(2,-0.015)]:
    print('band pt',(a,b),'C prime',closed_form(a,b)['m'],'twin',tw(a,b))
