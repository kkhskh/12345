import numpy as np
d,rho,eps=6,1.0,1e-5
P=lambda a: 1.5-0.5*np.tanh(5*a)
Q=lambda b: 2.0-1.0*np.tanh(5*b)
g=np.concatenate([-np.linspace(3,0.5,60),np.linspace(0.5,3,60)])
A,B=np.meshgrid(g,g,indexing='ij')
X=P(A)/np.sqrt((P(A)**2+Q(B)**2+rho**2)/d+eps)
relu=lambda t: np.maximum(t,0)
out=relu(X-1.0)-relu(X-1.15)-relu(X-1.6)+relu(X-1.75)-0.075
print("XNOR accuracy:", ((out>0)==(A*B>0)).mean())
