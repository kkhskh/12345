import sys
P="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_operator_reuse"; sys.path.insert(0,P+"/src"); sys.path.insert(0,P+"/scripts")
import numpy as np, xvitask as T, xvi_math_core as M
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
K=3; XA=T.xall(6)
for c in ["2:0.9:0.9:0.05","3:0.9:0.9:0.05"]:
    a,b1,b0,nu=c.split(":"); a=int(a); px,PB=T.joint(a,int(0) or float(b1),float(b0),float(nu))
    for order in (2,3):
        F=M.design(order,XA); d=F.shape[1]; s=np.sort(PB,1); u=s[:,-1]-s[:,-2]>1e-9; ys=PB.argmax(1)
        rows=[]; 
        # max total slack t (bounded) s.t. score(x,y*)-score(x,y) >= t_x ; maximise number satisfied ~ check feasibility with margin 1
        A=[];b=[]
        for i in np.flatnonzero(u):
            for y in range(K):
                if y==ys[i]: continue
                r=np.zeros(d*K); r[np.arange(d)*K+y]+=F[i]; r[np.arange(d)*K+ys[i]]-=F[i]; A.append(r); b.append(-1.0)
        A=csr_matrix(np.array(A)); res=linprog(np.zeros(d*K),A_ub=A,b_ub=b,bounds=(-1e4,1e4),method="highs")
        print(c,"order",order,"Bayes decisions exactly representable:",res.status==0, "unique-Bayes inputs",int(u.sum()))
