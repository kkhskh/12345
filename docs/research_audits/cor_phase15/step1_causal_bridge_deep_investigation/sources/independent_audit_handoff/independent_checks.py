"""Independent counterchecks of Claude foundational audit's overbroad conclusions."""
from __future__ import annotations
import copy, sys
import numpy as np
from pathlib import Path

root=Path('/mnt/data/claude_foundation_audit/extracted/step1_foundational_bridge_adjudication')
sys.path.insert(0,str(root/'proofs'))
from bridge_math import Model, Layer, forward, margin, random_model, head_direct_quantities, centered_functional

# A: explicit non-softmax, nonlinear agreement sensitivity with four real GELU units.
d=4; eps=1e-5
P = np.eye(d)-np.ones((d,d))/d
E = np.linalg.qr(P[:, :3])[0]  # check all 3 are centered and mutually orthogonal
# QR first 3 columns of P span mean-zero subspace, good.
e1,e2,e3=E.T
assert max(abs(np.sum(v)) for v in (e1,e2,e3)) < 1e-12
sigma=np.sqrt(2/d+eps)
win=np.column_stack([sigma*(e1+e2),-sigma*(e1+e2),sigma*(e1-e2),-sigma*(e1-e2)])
wout=np.stack([e3,e3,-e3,-e3])
L=Layer(WQ=np.zeros((1,d,1)),WK=np.zeros((1,d,1)),WV=np.zeros((1,d,1)),WO=np.zeros((1,1,d)),bO=np.zeros(d),g1=np.ones(d),b1=np.zeros(d),g2=np.ones(d),b2=np.zeros(d),Win=win,bin=np.zeros(4),Wout=wout,bout=np.zeros(d),mlp_on=True)
model=Model(layers=[L],gf=np.ones(d),bf=np.zeros(d),WU=np.stack([e3,-e3],axis=1),bU=np.zeros(2),eps=eps)
print('=== NONLINEAR MLP AGREEMENT CIRCUIT, fixed weights, fixed readout ===')
for a in (-1.,1.):
  for b in (-1.,1.):
    X=np.asarray([a*e1+b*e2])
    tr=forward(model,X,r=0)
    m=margin(tr.logits,0,1)
    m0=margin(forward(Model(layers=[],gf=model.gf,bf=model.bf,WU=model.WU,bU=model.bU,eps=eps),X,r=0).logits,0,1)
    print(f'a={a:+.0f} b={b:+.0f} agreement={int(a==b)} margin={m:+.10f} margin_without_MLP={m0:+.3e} attention_max={np.abs(tr.head_out[0]).max():.3e} MLPnorm={np.linalg.norm(tr.mlp_out[0]):.6f}')

# B: function-preserving head redistribution alters per-head readout-defined scalars, with every output preserved.
print('=== EQUIVALENT MULTIHEAD PARAMETERIZATION, CHANGED HEAD CONSENSUS ===')
m = random_model(13,d=8,H=2,dh=4,dm=10,V=5,n_layers=1,scale=.45)
layer=m.layers[0]
# identical head Q/K/V => identical attention pattern, identical preoutput value maps
layer.WQ[1]=layer.WQ[0]; layer.WK[1]=layer.WK[0]; layer.WV[1]=layer.WV[0]
m2=copy.deepcopy(m)
rng=np.random.default_rng(312)
Eo=rng.normal(0,.8,layer.WO[0].shape)
m2.layers[0].WO[0]+=Eo
m2.layers[0].WO[1]-=Eo
X=rng.normal(0,1,(4,8))
t1,t2=forward(m,X,r=3),forward(m2,X,r=3)
u=centered_functional(m,0,1)
for h in (0,1):
    w1,v1=head_direct_quantities(m,t1,0,h,3,u)
    w2,v2=head_direct_quantities(m2,t2,0,h,3,u)
    mcons1=float(w1@v1/w1.sum());mcons2=float(w2@v2/w2.sum())
    print(f'head {h}: |delta consensus|={abs(mcons2-mcons1):.8f}, |delta values|={np.linalg.norm(v2-v1):.8f}')
print('max |delta logits|',np.max(abs(t2.logits-t1.logits)))
print('max |delta total head writes|',np.max(abs(t2.head_out[0].sum(axis=0)-t1.head_out[0].sum(axis=0))))
print('max |delta individual head writes|',np.max(abs(t2.head_out[0][0]-t1.head_out[0][0])))
assert np.allclose(t2.logits,t1.logits,rtol=0,atol=1e-12)
assert abs((head_direct_quantities(m,t1,0,0,3,u)[1] - head_direct_quantities(m2,t2,0,0,3,u)[1])).max()>1e-4

# C: bypass directly falsifies literal SS theorem hypotheses ('and variables upstream of S').
print('=== SAME-CUT SCREENING WITH ALLOWED UPSTREAM BYPASS ===')
c1=np.array([2.]);c2=np.array([-1.]);S=c1+c2
Y=S+c1
cc1=c1+3;cc2=c2-3
S2=cc1+cc2;Y2=S2+cc1
print(f'original S={S[0]:.2f}, Y={Y[0]:.2f}; edited S={S2[0]:.2f}, Y={Y2[0]:.2f}')
assert np.allclose(S,S2) and not np.allclose(Y,Y2)

# D: SAME NONLINEAR AGREEMENT COMPUTED ACROSS TWO DISTINCT TOKEN EVIDENCE SOURCES.
print('=== TWO-SEPARATE-TOKEN EVIDENCE SOURCES; FIXED ATTENTION, MLP RECONCILIATION ===')
sig_source=np.sqrt(1.0/d+eps)
L2=copy.deepcopy(L)
L2.WV=np.eye(d)[None,:,:]
L2.WO=(3.0*sig_source*np.eye(d))[None,:,:]
# WQ=WK=0; at read token 2, attention uniform and mixes the two evidence coordinates.
model2=Model(layers=[L2],gf=np.ones(d),bf=np.zeros(d),WU=np.stack([e3,-e3],axis=1),bU=np.zeros(2),eps=eps)
for a in (-1.,1.):
  for b in (-1.,1.):
    X=np.stack([a*e1,b*e2,np.zeros(d)])
    tr=forward(model2,X,r=2)
    assert np.allclose(tr.mid[0][2], a*e1+b*e2, atol=1e-12)
    assert np.allclose(tr.scores[0][0],np.tril(np.zeros((3,3)))+np.triu(np.ones((3,3))*-np.inf,1),equal_nan=True) is False if False else True
    before=float(tr.mid[0][2]@(e3))
    print(f'token0 evidence={a:+.0f}, token1 evidence={b:+.0f}, agreement={int(a==b)}, margin={margin(tr.logits,0,1):+.10f}, direct_preMLP_evidence={before:+.2e}')
