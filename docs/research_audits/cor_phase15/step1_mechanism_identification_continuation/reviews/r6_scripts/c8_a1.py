import sys, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1]+'/../../step1_causal_bridge_deep_investigation/src')
import mlp_norm as M, cbmath
d=6; Bs=cbmath.mean_zero_basis(d); e1,e2,e3,e4,e5=Bs[:5]
rng=np.random.default_rng(31)
for act,name in [(M.gelu,'gelu'),(M.relu,'relu')]:
  for wn in [0,5,1e3,1e6,1e9,1e12]:
    for c in [1e-8,1.5,1e8]:
      bad=zero=0; n=3000; ex=None
      for _ in range(n):
        x,y=rng.uniform(-50,50,2)*10**rng.uniform(-3,0,2)
        w=rng.normal(size=3); w=w/np.linalg.norm(w)*wn
        r=x*e1+y*e2+w[0]*e3+w[1]*e4+w[2]*e5
        z=M.ln(r[None],eps=10**rng.uniform(-12,-1))[0]
        q=M.q_of(z,e1,e2,c=c,act=act)
        if q==0: zero+=1; ex=ex or ('zero',x,y,q)
        elif np.sign(q)!=np.sign(x*y): bad+=1; ex=('wrong',x,y,q)
      print(name,'|w|',wn,'c',c,'wrong',bad,'zero',zero,ex)
z=M.ln((1.0*e1+1e-17*e2+3*e5)[None])[0]; print('x=1,y=1e-17 q=',M.q_of(z,e1,e2))
z=M.ln((1.0*e1-1e-9*e2+1e8*e5)[None])[0]; print('x=1,y=-1e-9,|w|=1e8 q=',M.q_of(z,e1,e2))
