"""Fully standalone mathematical counterexamples to overbroad claims in the uploaded
COR Phase 1.5 foundational bridge adjudication. Does not import submitted code.
Requires only NumPy. Run: python independent_standalone.py
"""
import numpy as np

np.set_printoptions(precision=9,suppress=True)
d=4; eps=1e-5
P=np.eye(d)-np.ones((d,d))/d
U=np.linalg.qr(P[:,:3])[0]
e1,e2,e3=U.T
assert np.allclose(np.stack((e1,e2,e3)).sum(axis=1),0,atol=1e-13)
assert np.allclose(U.T@U,np.eye(3),atol=1e-13)

def ln(x):
    x=np.asarray(x,dtype=float)
    return (x-x.mean(axis=-1,keepdims=True))/np.sqrt(((x-x.mean(axis=-1,keepdims=True))**2).mean(axis=-1,keepdims=True)+eps)

def gelu(x):
    return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))

# Canonical two independent source tokens, one readout token; uniform causal attention.
# Query/key weight matrices zero. V_O=3 * normalization sigma for the source inputs.
sig_source=np.sqrt(1.0/d+eps)
sig_read=np.sqrt(2.0/d+eps)
Wv_o=3*sig_source*np.eye(d)
Win=np.column_stack([sig_read*(e1+e2),-sig_read*(e1+e2),sig_read*(e1-e2),-sig_read*(e1-e2)])
Wout=np.stack([e3,e3,-e3,-e3])

print('TWO DISTINCT EVIDENCE TOKENS / FIXED UNIFORM ATTENTION / FOUR GELU HIDDEN UNITS')
margins={}
for a in (-1.,1.):
 for b in (-1.,1.):
  input_tokens=np.stack([a*e1,b*e2,np.zeros(d)])
  normalized=ln(input_tokens)
  attn=(normalized@Wv_o).mean(axis=0)  # read token 2: uniform over tokens 0,1,2
  assert np.allclose(attn,a*e1+b*e2,atol=2e-13)
  q=float((gelu(ln(attn)@Win)@Wout)@e3)
  h=attn+q*e3
  margin=float(2*ln(h)@e3)
  margins[(a,b)]=margin
  assert np.sign(margin)==(1 if a==b else -1)
  assert abs(2*ln(attn)@e3)<1e-12
  print('a=%+d b=%+d agreement=%d, final_margin=%+.10f, MLP_write=%+.10f' % (a,b,a==b,margin,q))
assert np.isclose(margins[(1.,1.)],-margins[(1.,-1.)],atol=1e-12)
print('The sign of the candidate margin equals agreement, with attention fixed and no pre-MLP readout margin.\n')

# Two heads with identical attention and value input; redistribute output matrices.
# Function output unchanged for all X because the two head outputs sum Y@(W1+W2).
rng=np.random.default_rng(404)
D=6;DH=3;N=5
X=rng.normal(size=(N,D)); QW=rng.normal(size=(D,DH));KW=rng.normal(size=(D,DH));VW=rng.normal(size=(D,DH))
S=(X@QW)@(X@KW).T/np.sqrt(DH)
S[np.triu_indices(N,1)]=-np.inf
S=S-np.max(S,axis=1,keepdims=True)
A=np.exp(S); A/=A.sum(axis=1,keepdims=True)
Y=A@(X@VW)
W1=rng.normal(size=(DH,D));W2=rng.normal(size=(DH,D)); perturb=rng.normal(size=(DH,D));u=rng.normal(size=D)
base1=Y@W1;base2=Y@W2
new1=Y@(W1+perturb);new2=Y@(W2-perturb)
assert np.allclose(base1+base2,new1+new2,atol=1e-12)
print('FUNCTION-PRESERVING HEAD REPARAMETERIZATION')
print('per-head consensus scalar 1 before/after:',float(base1[-1]@u), float(new1[-1]@u))
print('per-head consensus scalar 2 before/after:',float(base2[-1]@u), float(new2[-1]@u))
print('max total residual change:',np.max(np.abs(base1+base2-new1-new2)))
print('max individual centered write change:',np.max(np.abs((new1-base1)-(new1-base1).mean(axis=-1,keepdims=True))))
print('This is a true function-preserving symmetry across all inputs when heads share Q,K,V.\n')

# Literal formal assumption in SS allows an upstream-to-output bypass, which invalidates its universal wording.
c1,c2=2.,-1.;s=c1+c2;y=s+c1
c1p,c2p=c1+3,c2-3;sp=c1p+c2p;yp=sp+c1p
assert s==sp and y!=yp
print('SAME-CUT SCREENING BYPASS COUNTEREXAMPLE: S=%.0f unchanged; Y=%.0f -> %.0f' % (s,y,yp))
