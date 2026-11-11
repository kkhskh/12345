import os, sys, time, json
PK="/home/claude/12345/docs/research_audits/cor_phase15/step1_perfect_agreement_boundary"
sys.path.insert(0, PK+"/src")
import numpy as np, boundary as B
E=B.E; STREAM=16_000_000; NE=3000
D=E.sample(NE,STREAM); X,cfg=D["X"],D["cfg"]; q=np.random.default_rng(STREAM+1).integers(0,6,NE); idx=np.arange(NE)
frm=4; Xp=X.copy(); gs=np.full(NE,-1); ok=np.zeros(NE,bool)
for qq in range(6):
    s=q==qq; a,g_,k_,o=B.count_edit(X[s],qq,cfg[s],np.random.default_rng(STREAM+10+qq+100*frm),frm,frm+1); Xp[s],gs[s],ok[s]=a,g_,o
vote=np.array([B.group_vote(X[i:i+1],q[i],max(gs[i],0))[0] for i in idx]); orient=ok&(vote>=0)
n0=E.loo(X)[idx,q]; Hb=n0==5; Vq=np.stack([np.array([B.group_vote(X[i:i+1],q[i],g)[0] for i in idx]) for g in range(3)],1)
R1=np.zeros(NE,bool)
for i in np.where(orient)[0]:
    oth=[h for h in range(3) if h!=gs[i]]; R1[i]=any(Hb[i,h] and Vq[i,h]==vote[i] for h in oth)
v=np.maximum(vote,0); print("R1 n",R1.sum())
res={}
for m in sys.argv[1:]:
    t0=time.time(); M=B.Model(m); kind=M.kind
    A0,L0,at0,ml0=M.acts(X); A1,L1,at1,ml1=M.acts(Xp); c0=E.clogit(L0[idx,q])
    om=lambda L: B.oriented_margin(E.clogit(L[idx,q]),v)-B.oriented_margin(c0,v)
    dfull=om(L1); dA=A1-A0; T=A0.shape[1]
    def mask(posf):
        mk=np.zeros((NE,T),bool)
        for i in idx: mk[i,posf(i)]=True
        return mk
    site=mask(lambda i:E.site_positions(kind,q[i])); src=mask(lambda i:list(range(6*q[i],6*q[i]+6))); rd=mask(lambda i:[36+q[i]])
    allm=np.ones((NE,T),bool); oth_src=np.zeros((NE,T),bool); oth_src[:,:36]=True; oth_src&=~src
    out={}
    def rec(name,dm): out[name]=dict(R1_mean=float(dm[R1].mean()),R1_share=float((dm*dfull)[R1].sum()/(dfull**2)[R1].sum()),all_share=float((dm*dfull)[orient].sum()/(dfull**2)[orient].sum()))
    rec("full",dfull)
    for nm,mk in (("site",site),("qsrc",src),("readout_q",rd),("other_src",oth_src),("nonsite",~site)):
        rec(nm,om(M.run(X,dA*mk[:,:,None])))
    # half-scale linearity at site
    rec("site_x0.5",om(M.run(X,0.5*dA*site[:,:,None])))
    one=np.ones(T,np.float32); zero=np.zeros(T,np.float32)
    for nm,mk in (("site",site),("all",allm)):
        dl=dA*mk[:,:,None]
        rec(nm+"|direct(attn,mlp frozen)",om(M.run_frozen(X,dl,at0,ml0,one,one)))
        rec(nm+"|attn frozen (direct+MLP)",om(M.run_frozen(X,dl,at0,ml0,one,zero)))
        rec(nm+"|mlp frozen (direct+attn)",om(M.run_frozen(X,dl,at0,ml0,zero,one)))
    res[m]=out; print(m,f"{time.time()-t0:.0f}s"); [print("  %-30s R1 mean %+.3f  R1 share %+.2f  all share %+.2f"%(k,o["R1_mean"],o["R1_share"],o["all_share"])) for k,o in out.items()]
    sys.stdout.flush()
json.dump(res,open("/home/claude/12345/docs/research_audits/cor_phase15/step1_perfect_agreement_boundary/reviews/checks/R4_path_split.json","w"),indent=1)
