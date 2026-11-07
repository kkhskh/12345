import sys, math, dataclasses, numpy as np
sys.path.insert(0, sys.argv[1])
from cprime import P, closed_form, forward_margin, certify, interval_m, I
from cprime_regions import suff
rng = np.random.default_rng(7)
# 1. closed form vs forward at large |a|,|b| and random extreme params
worst = 0; worst_rel=0; info=None
for t in range(300):
    p = dataclasses.replace(P(), lam=float(10**rng.uniform(-2,2)), mu=float(10**rng.uniform(-2,1.5)),
        kappa=float(10**rng.uniform(-3,1.5)), rho=float(10**rng.uniform(-3,1.5)), lam1=float(10**rng.uniform(-2,1.5)),
        c1=float(10**rng.uniform(-2,1.5)), wv=float(10**rng.uniform(-3,2)), prior_v=float(10**rng.uniform(-3,2)),
        eps=float(10**rng.uniform(-12,-1)))
    a, b = rng.uniform(-50,50,2)
    try:
        c = closed_form(a,b,p)['m']; f = forward_margin(a,b,p)
    except OverflowError as e:
        print('overflow', p, a, b); continue
    dd = abs(c-f); rel = dd/max(1e-300,abs(f))
    if dd > worst: worst=dd; info=(p,a,b,c,f)
    if np.sign(c)!=np.sign(f): print('SIGN MISMATCH closed vs forward', p, a, b, c, f)
print('closed vs forward worst abs diff', worst, info)
# 2. suff soundness at extreme params and |a|,|b|<=50, tiny eps
viol=[]; n=0; dec=0; ovf=0
for t in range(400):
    p = dataclasses.replace(P(), lam=float(10**rng.uniform(-2,2.5)), mu=float(10**rng.uniform(-2,1.5)),
        kappa=float(10**rng.uniform(-3,1.5)), rho=float(10**rng.uniform(-3,1.5)), lam1=float(10**rng.uniform(-2,1.5)),
        c1=float(10**rng.uniform(-2,1.5)), wv=float(10**rng.uniform(-3,2)), prior_v=float(10**rng.uniform(-4,3)),
        eps=float(10**rng.uniform(-15,-1)))
    for a,b in np.concatenate([rng.uniform(-50,50,(50,2)), rng.uniform(-1e-3,1e-3,(25,2)), 10**rng.uniform(-8,1.7,(25,2))*rng.choice([-1,1],(25,2))]):
        try:
            s = suff(a,b,p); m = closed_form(a,b,p)['m']
        except OverflowError:
            ovf+=1; continue
        n+=1
        if s!=0:
            dec+=1
            if m==0 or np.sign(m)!=s: viol.append((p,a,b,s,m))
print('suff: n',n,'decided',dec,'violations',len(viol),'overflow',ovf)
for v in viol[:5]: print('  VIOL', v)
# 3. negative/zero parameters
for name, kw in [('wv<0',dict(wv=-1.0)),('prior_v<0',dict(prior_v=-1.0)),('lam<0',dict(lam=-6.0)),('c1<0',dict(c1=-1.5)),('lam1<0',dict(lam1=-1.5)),('mu<0',dict(mu=-3.0)),('prior_v=0',dict(prior_v=0.0)),('wv=0',dict(wv=0.0))]:
    p = dataclasses.replace(P(), **kw); bad=0; tot=0; ex=None
    for a,b in rng.uniform(-3,3,(3000,2)):
        s=suff(a,b,p); m=closed_form(a,b,p)['m']
        if s!=0:
            tot+=1
            if np.sign(m)!=s: bad+=1; ex=ex or (a,b,s,m)
    print(name,'decided',tot,'violations',bad, ex)
