import sys, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1]+'/../../step1_causal_bridge_deep_investigation/src')
import cbmath
from relation_free_equivalent import relation_free
mR,emb=relation_free()
tw=lambda a,b: cbmath.margin(cbmath.forward(mR,emb(a,b),3))
for r in np.arange(3,12,0.25):
    print(round(r,2), 'twin(r,r)=%.4f'%tw(r,r), 'twin(r,3)=%.4f'%tw(r,3.0), 'twin(r,0.5)=%.4f'%tw(r,0.5))
