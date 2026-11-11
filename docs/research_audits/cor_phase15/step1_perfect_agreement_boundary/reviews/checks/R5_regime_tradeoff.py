import json,sys,numpy as np
PK="/home/claude/12345/docs/research_audits/cor_phase15/step1_perfect_agreement_boundary"
sys.path.insert(0,PK+"/src"); import boundary as B; E=B.E
r=json.load(open("pad.json"))
for a in "TP":
    ks=[k for k in r if k[1]==a]; r1=np.array([r[k]["obs"]["F45R1"] for k in ks]); r3=np.array([r[k]["obs"]["F45R3"] for k in ks]); g=np.array([r[k]["gd_coef"][0] for k in ks])
    print(a,"corr(R1,R3)=%.2f corr(R1,gain)=%.2f corr(R3,gain)=%.2f"%(np.corrcoef(r1,r3)[0,1],np.corrcoef(r1,g)[0,1],np.corrcoef(r3,g)[0,1]))
s=json.load(open(PK+"/logs/summary_tables.json"))
for k,v in s.items(): print(k, {rg: round(v["F45"]["regimes"][rg]["B"],3) for rg in ("R1","R2","R3")}, "F34R3 %.3f"%v["F34"]["regimes"]["R3"]["B"])
g=np.load(PK+"/logs/conf/generator.npz",allow_pickle=True); X,q=g["X"],g["q"]; n=E.loo(X)[np.arange(len(X)),q]
n5=(n==5).sum(1); print("frac decisions with >=2 groups at 5/5 (leave-q):", (n5>=2).mean(), " with exactly 1:", (n5==1).mean())
# how many of those have the two 5/5 groups voting same candidate at q
G=B.G; same=np.zeros(len(X),bool)
for i in np.where(n5>=2)[0]:
    vs=[ (X[i,q[i],G.GROUPS[gg][0]] if X[i,q[i],G.GROUPS[gg][0]]==X[i,q[i],G.GROUPS[gg][1]] else -1) for gg in range(3) if n[i,gg]==5]
    vs=[v for v in vs if v>=0]; same[i]= len(vs)>=2 and len(set(vs))<len(vs)
print("frac natural decisions: >=2 perfect groups agreeing at q (R1-after type):", same.mean())
