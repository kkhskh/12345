# continuous ReLU reader min(u,v) = relu(u) - relu(-u) - relu(u-v) on shared-LN read coords: implements T on all of {ab != 0}
import sys, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1]+'/../../step1_causal_bridge_deep_investigation/src')
import mlp_norm as M, cbmath
d=6; Bs=cbmath.mean_zero_basis(d); e1,e2,e3,e4,e5=Bs[:5]
sig=lambda t: t/np.sqrt((t*t+4.0)/d+1e-5)
rng=np.random.default_rng(3); bad=0; n=0
pts=np.concatenate([rng.uniform(-3,3,(100000,2)), rng.uniform(-1e-6,1e-6,(50000,2)), np.column_stack([rng.uniform(0,3,50000), rng.uniform(-1e-8,1e-8,50000)])])
for a,b in pts:
    if a*b==0: continue
    r=sig(a)*e1+sig(b)*e2+3.0*e5; z=M.ln(r[None])[0]; u,v=z@e1,z@e2
    m=M.relu(u)-M.relu(-u)-M.relu(u-v)
    T=np.sign(a) if a*b>0 else -1
    n+=1; bad+= np.sign(m)!=T
print('min-reader: points',n,'rule violations',bad)
