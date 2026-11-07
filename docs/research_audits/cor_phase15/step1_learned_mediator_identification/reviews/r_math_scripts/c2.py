import sys, numpy as np
sys.path.insert(0,"/tmp/rv/step1_causal_bridge_deep_investigation/src")
import cbmath
BASE,D,EMB=cbmath.rereading(); R=3
rng=np.random.default_rng(51); PTS=rng.uniform(0.3,3.0,(200,2))
fams={"b":lambda a,b:(a,-b),"a":lambda a,b:(-a,b),"asym":lambda a,b:(-0.5*a,1.7*b)}
for f,T in fams.items():
    sh=[];dq=[];vs=[]
    for a,b in PTS:
        base=cbmath.forward(BASE,EMB(a,b),R); cf=cbmath.forward(BASE,EMB(*T(a,b)),R)
        mb,mc=cbmath.margin(base),cbmath.margin(cf)
        val=base.mlp_out[0].copy(); val[2]=cf.mlp_out[0][2]
        mq=cbmath.margin(cbmath.forward(BASE,EMB(a,b),R,cbmath.replace_site(0,"mlp_out",val)))
        sh.append((mq-mb)/(mc-mb)); dq.append(np.linalg.norm(cf.mlp_out[0][2]-base.mlp_out[0][2]))
        vv=cbmath.margin(cbmath.forward(BASE,EMB(a,b),R,cbmath.replace_site(1,"V0",cf.V[1][0])))
        vs.append((vv-mb)/(mc-mb))
    sh=np.array(sh); vs=np.array(vs)
    print(f,"qB share median/min/max",np.median(sh),sh.min(),sh.max(),"|dq_B| max",max(dq), "values share min/med/max",vs.min(),np.median(vs),vs.max())
