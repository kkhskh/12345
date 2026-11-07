import sys, numpy as np
R="/home/claude/12345/docs/research_audits/cor_phase15/step1_learned_mediator_identification"
sys.path.insert(0,R+"/src")
import generators as G
for t in ["REL-XNOR","REL-GATE","REL-MIS","AGG-NL","AGG-LIN"]:
    print(t, G.aggregate_only_ceiling(t), G.aggregate_only_ceiling(t,seed=1,bins=300))
# analytic XNOR ceiling under mixture: P(ab>0 | z) ; Monte Carlo held-out check
rng=np.random.default_rng(5)
for t in ["REL-XNOR","REL-MIS"]:
    tr=G.sample(t,2_000_000,rng); te=G.sample(t,2_000_000,rng); bins=1200
    kb=lambda d: np.clip(((d["z"]+6)/12*bins).astype(int),0,bins-1)
    pos=np.bincount(kb(tr),weights=tr["y"]>0,minlength=bins); cnt=np.bincount(kb(tr),minlength=bins)
    rule=pos>cnt-pos
    print(t,"held-out",((rule[kb(te)])==(te["y"]>0)).mean())
# XNOR vs MIS identical ceiling? MIS: y=-sign(a) if ab>0 else sign(z). at fixed z, ab>0 => sign(a)=sign(z) so y=-sign z; else y=sign z
