"""POST HOC discriminability check for PP1: how much sign agreement do non-network reference rules with near-Bayes
on-distribution accuracy obtain on the same edits?  (local factorised plug-in; Bayes with logit noise of several sizes;
monotone T1 plug-in).  Same edit construction as pp1_sign.py, 500 base episodes, stream 6,100,000."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import gen_family as G, local_bayes as LB
f4, f1 = G.Fam(pc=1.0, rho=1.0), G.Fam(pc=0.0); K, Q = 3, 6
d = G.sample(f4, 500, np.random.default_rng(6_100_000)); X0, Y0 = d["X"], d["Y"]
bi, qq, gg, Xe = [], [], [], []
for q in range(Q):
    for k in range(Q):
        if k == q: continue
        for g, (a, b) in enumerate(G.GROUPS):
            X = X0.copy(); ag = X[:, k, a] == X[:, k, b]; X[:, k, b] = np.where(ag, (X[:, k, a] + 1) % K, X[:, k, a])
            Xe.append(X); bi.append(np.arange(500)); qq.append(np.full(500, q)); gg.append(np.full(500, g))
Xe = np.concatenate(Xe); bi = np.concatenate(bi); qq = np.concatenate(qq); gg = np.concatenate(gg); n = len(Xe)
A0 = np.stack([(X0[:, :, a] == X0[:, :, b]) for a, b in G.GROUPS], -1).astype(int); Ae = np.stack([(Xe[:, :, a] == Xe[:, :, b]) for a, b in G.GROUPS], -1).astype(int)
cb = A0[bi].sum(1)[np.arange(n), gg] - A0[bi, qq, gg]; ca = Ae.sum(1)[np.arange(n), gg] - Ae[np.arange(n), qq, gg]
boundary = ((cb == 4) & (ca == 5)) | ((cb == 5) & (ca == 4))
with np.errstate(divide="ignore", invalid="ignore"):
    lb0, lbe = np.log(G.posterior(f4, X0)["post_y"]), np.log(G.posterior(f4, Xe)["post_y"])
order = np.argsort(-np.nan_to_num(lb0, nan=-1e9), -1); c1 = order[bi, qq, 0]; c2 = order[bi, qq, 1]
def dm(l0, le): return (le[np.arange(n), qq, c1] - le[np.arange(n), qq, c2]) - (l0[bi, qq, c1] - l0[bi, qq, c2])
dm4 = dm(lb0, lbe); sel = np.isfinite(dm4) & (np.abs(dm4) > 0.5); selb = sel & boundary
res = {}
def score(name, l0, le, acc):
    x = dm(l0, le); res[name] = dict(ondist_acc=acc, sign_all=float((np.sign(x[sel]) == np.sign(dm4[sel])).mean()), sign_boundary=float((np.sign(x[selb]) == np.sign(dm4[selb])).mean()))
loc0, loce = np.log(LB.local_post(G.Fam(pc=1.0, rho=0.999), X0)), np.log(LB.local_post(G.Fam(pc=1.0, rho=0.999), Xe))
score("local_plugin", loc0, loce, float((loc0.argmax(-1) == Y0).mean()))
p1_0, p1_e = np.log(G.posterior(f1, X0)["post_y"]), np.log(G.posterior(f1, Xe)["post_y"])
score("T1_plugin", p1_0, p1_e, float((p1_0.argmax(-1) == Y0).mean()))
rng = np.random.default_rng(0)
for s in [0.3, 1.0, 2.0]:
    z0 = rng.normal(0, s, lb0.shape); ze = z0[bi] + rng.normal(0, s * 0.3, lbe.shape)   # noise correlated between base and edit
    score(f"bayes_plus_logit_noise_{s}", np.nan_to_num(lb0, nan=-50) + z0, np.nan_to_num(lbe, nan=-50) + ze, float(((np.nan_to_num(lb0, nan=-50) + z0).argmax(-1) == Y0).mean()))
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp1_reference_rules_posthoc.json"), "w"), indent=1)
for k, v in res.items(): print(k, {a: round(b, 3) for a, b in v.items()})
