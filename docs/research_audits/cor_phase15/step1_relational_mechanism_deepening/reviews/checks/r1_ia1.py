import sys, math, numpy as np, mpmath as mp
sys.path.insert(0,'/home/claude/12345/docs/research_audits/cor_phase15/step1_mechanism_identification_continuation/src')
from cprime import *
mp.mp.dps=50
k2m2=13.0; d=10; e=1e-5
fx=lambda t: t / math.sqrt((t * t + k2m2) / d + e)
def fxm(t):
    t=mp.mpf(t); return t/mp.sqrt((t*t+k2m2)/d+mp.mpf(e))
rng=np.random.default_rng(0); worst=0
for t in np.concatenate([rng.uniform(-3,3,200000), rng.uniform(-1e-3,1e-3,10000)]):
    if t==0: continue
    v=fx(t); ex=fxm(t)
    ulp=abs(np.spacing(v))
    err=float(abs(mp.mpf(v)-ex)/ulp); worst=max(worst,err)
print("fx worst error in ulps:", worst)
# heven error vs pad
def hm(t):
    t=mp.mpf(t); K_=mp.sqrt(2/mp.pi); return t*mp.tanh(K_*(t+mp.mpf(C)*t**3))
worstpad=0
for t in rng.uniform(-8,8,100000):
    v=heven(t); ex=hm(t); pad=1e-15*(1+abs(v))
    worstpad=max(worstpad, float(abs(mp.mpf(v)-ex)/pad))
print("heven err / pad worst:", worstpad)
# K itself: math.sqrt(2/pi) in float vs exact; C=0.044715 decimal not exact - model uses same float so ok
print("K err", float(mp.mpf(K)-mp.sqrt(2/mp.pi)))
