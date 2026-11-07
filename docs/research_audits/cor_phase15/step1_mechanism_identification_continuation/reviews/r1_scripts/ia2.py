import sys, math, numpy as np, mpmath as mp
sys.path.insert(0,'/home/claude/12345/docs/research_audits/cor_phase15/step1_mechanism_identification_continuation/src')
from cprime import *
from cprime import _dn, _up
mp.mp.dps=50
k2m2=13.0; d=10; e=1e-5
fx=lambda t: t / math.sqrt((t * t + k2m2) / d + e)
def fxm(t):
    t=mp.mpf(t); return t/mp.sqrt((t*t+k2m2)/d+mp.mpf(e))
rng=np.random.default_rng(0); bad=[]
for t in rng.uniform(0,3,200000):
    lo=_dn(fx(t)); hi=_up(fx(t)); ex=fxm(t)
    if not (mp.mpf(lo)<=ex<=mp.mpf(hi)): bad.append(t)
print(len(bad), "of 200000 degenerate endpoints NOT enclosed; e.g.", repr(bad[:3]))
t=bad[0]; print(repr(_dn(fx(t))), repr(_up(fx(t))), mp.nstr(fxm(t),25))
