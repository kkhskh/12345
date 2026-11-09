"""Review response (R2-stat F5, F6, F7): an edit family with a HIGH density ratio, and unfiltered / clustered reporting.
Family NP ('make the noise group perfect'): for a natural T4 episode and query q, set the noise group's second source equal
to its first in every question k != q.  Endpoints lie in the event A_q = {>= 2 groups agree in every k != q}, natural
probability ~0.13, so the family's density ratio is >= ~7.7 at the event level.
Scored: Bayes margin change at q for the base top-2 pair (c1, c2), all S1 endpoints, by |dm| bin including (0, 0.5].
Models: the 10 learned T4 models; constructed CX4 = Bayes except on the region R = {>= 2 groups have total count >= 5},
where the logits of the top-2 candidates are swapped (R's natural mass is reported; R is where NP endpoints concentrate).
Clustered 95% intervals: bootstrap over base episodes.  Stream 6_180_000 (new; not the frozen PP1 stream).
Writes logs/pp1_highratio.json."""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src")); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, gen_family as G, pp2_sig as S
from pp2_dev_models import M
f4 = G.Fam(pc=1.0, rho=1.0); K, Q = 3, 6; NB = 2000
d = G.sample(f4, NB, np.random.default_rng(6_180_000)); X0, cfg = d["X"], d["cfg"]; cf = f4.configs
noi = np.array([3 - cf[c][0] - cf[c][1] for c in cfg]); bi, qq, Xe = [], [], []
for q in range(Q):
    X = X0.copy(); n_ = np.arange(NB)
    for k in range(Q):
        if k != q: X[n_, k, 2 * noi + 1] = X[n_, k, 2 * noi]
    Xe.append(X); bi.append(np.arange(NB)); qq.append(np.full(NB, q))
Xe, bi, qq = map(np.concatenate, (Xe, bi, qq)); n = len(Xe); ar = np.arange(n)
with np.errstate(divide="ignore", invalid="ignore"):
    l0, le = np.log(G.posterior(f4, X0)["post_y"]), np.log(G.posterior(f4, Xe)["post_y"])
S1 = np.isfinite(le).all((1, 2)); order = np.argsort(-np.nan_to_num(l0, nan=-9), -1); c1, c2 = order[bi, qq, 0], order[bi, qq, 1]
mg = lambda l, rows: l[rows, qq, c1] - l[rows, qq, c2]
dm = mg(le, ar) - mg(l0, bi)
Dn = G.sample(f4, 20000, np.random.default_rng(6_150_000))["X"]
cnt = lambda X: np.stack([(X[:, :, a] == X[:, :, b]).sum(1) for a, b in G.GROUPS], -1)
inR = lambda X: (cnt(X) >= 5).sum(-1) >= 2
Aq = lambda X, q: (np.stack([(X[:, [k for k in range(Q) if k != q], a] == X[:, [k for k in range(Q) if k != q], b]).all(1) for a, b in G.GROUPS], -1).sum(-1) >= 2)
res = dict(n_edits=int(n), n_S1=int(S1.sum()), natural_P_Aq=float(np.mean([Aq(Dn, q).mean() for q in range(Q)])), natural_mass_R=float(inR(Dn).mean()),
           frac_endpoints_in_R=float(inR(Xe)[S1].mean()), frac_bases_in_R=float(inR(X0).mean()))
bins = [(0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 1e9)]
def score(L0, Le, name):
    dmh = mg(Le, ar) - mg(L0, bi); ok = np.sign(dmh) == np.sign(dm); sel = S1 & (dm != 0)
    r = dict(sign_agreement_all_S1=float(ok[sel].mean()), sign_agreement_absdm_gt_0_5=float(ok[sel & (np.abs(dm) > 0.5)].mean()))
    for lo, hi in bins:
        m = sel & (np.abs(dm) >= lo) & (np.abs(dm) < hi); r[f"|dm| {lo}-{hi}"] = (int(m.sum()), float(ok[m].mean()) if m.any() else None)
    rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        e = rng.integers(0, NB, NB); w = np.bincount(e, minlength=NB)[bi] * sel; bs.append((ok * w).sum() / w.sum())
    r["clustered_CI95_all_S1"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(name, {k: v for k, v in r.items()}, flush=True); return r
for nm, (k_, pth) in M.items():
    ap = S.load(k_, os.path.join(LR, pth)); L0 = ap(X0)[0]; Le = np.concatenate([ap(Xe[i:i + 6000])[0] for i in range(0, n, 6000)])
    L0 = L0 - np.log(np.exp(L0).sum(-1, keepdims=True)); Le = Le - np.log(np.exp(Le).sum(-1, keepdims=True))
    res[nm] = score(L0, Le, nm); Ln = ap(Dn[:3000])[0]; Ln = Ln - np.log(np.exp(Ln).sum(-1, keepdims=True)); Pn = G.posterior(f4, Dn[:3000])["post_y"]
    res[nm]["natural_KL"] = float((Pn * (np.log(Pn + 1e-12) - Ln)).sum(-1).mean())
def cx4(L, X):
    L = np.nan_to_num(L, nan=-50.0, neginf=-50.0).copy(); r = np.where(inR(X))[0]
    s = np.argsort(-L[r], -1); a, b = s[..., 0], s[..., 1]; ii = np.arange(len(r))[:, None]; qa = np.arange(Q)[None]
    la, lb_ = L[r][ii, qa, a].copy(), L[r][ii, qa, b].copy(); Lr = L[r]; Lr[ii, qa, a] = lb_; Lr[ii, qa, b] = la; L[r] = Lr
    return L - np.log(np.exp(L).sum(-1, keepdims=True))
res["CX4_swap_on_R"] = score(cx4(l0, X0), cx4(le, Xe), "CX4")
Pn = G.posterior(f4, Dn)["post_y"]; Lc = cx4(np.log(Pn), Dn); res["CX4_swap_on_R"]["natural_KL"] = float((Pn * (np.log(Pn + 1e-12) - Lc)).sum(-1).mean())
res["CX4_swap_on_R"]["natural_acc"] = float((Lc.argmax(-1) == G.sample(f4, 20000, np.random.default_rng(6_150_000))["Y"]).mean())
print({k: v for k, v in res.items() if not isinstance(v, dict)})
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp1_highratio.json"), "w"), indent=1)
