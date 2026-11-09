import sys,numpy as np
sys.argv=['x']; exec(open('/tmp/r1/model_checks.py').read().split('out={}')[0])
apply,_=mk("checkpoints/trans_holdout_s1.npz")
for nm,PS in (("train_eps",TRP),("holdout_eps",HOP)):
    d=T3.sample_trans_pairs(6000,np.random.default_rng(9_950_000),PS); X,Y,typ=d["X"],d["Y"],d["typ"]; c=typ==2
    P=apply(X)[0]; pf=TX.bayes_trans(X)["post_y"]; pr=rbayes(X,TRP); m=P.argmax(-1)
    s=c&uniq(pf)&uniq(pr)&(pr.argmax(-1)!=pf.argmax(-1))
    print(nm,"n",s.sum(),"model=restricted",(m==pr.argmax(-1))[s].mean(),"model=full",(m==pf.argmax(-1))[s].mean(),
          "| overall unique-both: agree restr",(m==pr.argmax(-1))[c&uniq(pf)&uniq(pr)].mean(),"agree full",(m==pf.argmax(-1))[c&uniq(pf)&uniq(pr)].mean())
