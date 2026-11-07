import sys, numpy as np, dataclasses
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1]+'/../../step1_causal_bridge_deep_investigation/src')
import cbmath
from cprime import closed_form, P
mb,D,emb=cbmath.rereading(block1_mlp=False)
mbl=lambda a,b: cbmath.margin(cbmath.forward(mb,emb(a,b),3))
pb=dataclasses.replace(P(), lam1=0.0)  # lam1=0 -> q writes nothing -> uniform routing == blind
print('blind via closed form lam1=0 vs forward at (2,2):', closed_form(2,2,pb)['m'], mbl(2,2))
for t in [(1,1),(2,2),(3,3),(1.5,0.5)]: print('m_blind',t, mbl(*t))
best=-1e9; arg=None
for a in np.concatenate([np.linspace(0.01,10,300), [20,50,100,1e3,1e4]]):
    for b in -np.concatenate([np.logspace(-8,0,200), np.linspace(1,10,50)]):
        v=closed_form(a,b,pb)['m']
        if v>best: best=v; arg=(a,b)
print('sup m_blind over a>0>b (grid):', best, arg)
best=-1e9
for b in np.concatenate([np.linspace(0.01,10,300), [1e3,1e4]]):
    for a in -np.logspace(-8,1,300):
        v=closed_form(a,b,pb)['m']; best=max(best,v)
print('sup over a<0<b:', best)
