"""Formal review check 2: Lemma 1/2, approximation-table rows, Proposition 1 (KL vs B on a finite domain), Counterexample 2."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(PKG, "src")); import cri as R; E = R.E
rng = np.random.default_rng(0); K = 3
c = lambda l: l - l.mean(-1, keepdims=True)
sm = lambda l: np.exp(c(l)) / np.exp(c(l)).sum(-1, keepdims=True)
# Lemma 2 (random algebra check)
viol = 0
for _ in range(20000):
    cs = c(rng.normal(size=(2, K))); e = rng.normal(size=(2, K)) * rng.random(); z = rng.normal(size=K) * rng.random()
    lx, lxp = cs[0] + e[0], cs[1] + e[1]; ldo = lxp + z
    eta = np.abs(c(e)).max(); zeta = np.abs(c(z)).max()
    viol += np.abs(c(ldo) - c(lx) - (cs[1] - cs[0])).max() > 2 * eta + zeta + 1e-12
print("Lemma 2 violations:", viol)
# TV row: is ||p-q||_inf <= TV (so TV<=eps => inf-norm<=eps)?  and does 'TV >= eps/2' give the needed direction?
P = rng.dirichlet(np.ones(K), 50000); Qd = rng.dirichlet(np.ones(K), 50000)
tv = 0.5 * np.abs(P - Qd).sum(1); inf = np.abs(P - Qd).max(1)
print("max(inf/TV) = %.4f (<=1 means TV<=eps => ||.||_inf<=eps); min(inf/TV) = %.4f" % ((inf / tv).max(), (inf / tv).min()))
# pointwise-probability row: |e|_inf <= log((pmin+eps)/(pmin-eps)) when p*>=pmin>eps
viol = 0; worst = 0
for _ in range(50000):
    p = rng.dirichlet(np.ones(K)); pmin = p.min(); eps = pmin * rng.random() * 0.99
    d = rng.uniform(-eps, eps, K); d -= d.mean(); pm = np.clip(p + d, 1e-15, 1); pm /= pm.sum()
    if np.abs(pm - p).max() > eps: continue
    e = np.abs(c(np.log(pm)) - c(np.log(p))).max(); b = np.log((pmin + eps) / (pmin - eps)); worst = max(worst, e / b); viol += e > b + 1e-12
print("prob-error row: violations %d, max e/bound %.3f" % (viol, worst))
# T4 posterior floor
d = E.sample(20000, 11); X = d["X"]; pb = E.bayes_from(X, E.loo(X)); print("T4 min p*_y over 120k decisions: %.4g" % np.nanmin(pb))
# Proposition 1: pointwise KL of the construction vs B, and KL lower bound for ANY log-odds error of size B
pst = pb.reshape(-1, K)[:5]
for B in (1, 5, 20, 100):
    for p in pst[:2]:
        yh = p.argmax(); pm = sm(np.log(p) + B * (np.eye(K)[yh] - 1 / K)); kl = (p * np.log(p / pm)).sum()
        print("B=%4d p*=%s KL=%.4f  bound B(1-p_yhat)=%.4f  ||Delta||_inf=%.3f*B" % (B, np.round(p, 3), kl, B * (1 - p[yh]), 2 / 3))
# minimal over models: min KL(p*||p_M) subject to |c(log p_M)-c(log p*)|_inf >= B, by random search lower envelope
p = pst[0]; best = {}
for B in (2, 5, 10, 20):
    m = np.inf
    for _ in range(40000):
        u = rng.normal(size=K); u = c(u); u = u / np.abs(u).max() * B; pm = sm(np.log(p) + u); m = min(m, (p * np.log(p / pm)).sum())
    best[B] = m
print("approx min pointwise KL at log-odds error B (grows ~linearly):", {k: round(v, 3) for k, v in best.items()})
# smallest decision-input probability: crude lower bound (one config term only)
e_ = 0.1; per_q = (1 / 3) * (1 / 9) * (2 * 0.8 * e_ + e_ ** 2) / 3
print("lower bound on min P(x) for a positive-probability episode ~ %.3g  => Prop 1 needs kappa >= ~P(S)*B*(1-p_yhat) > 0" % (per_q ** 6 / 6))
# Counterexample 2: beta<0 breaks Bayes argmax
n = E.loo(X); q = 0; tot = n[:, q].sum(-1); pq = pb[:, q]; yh = pq.argmax(-1)
for beta in (0.5, -0.5):
    l = np.log(pq) + beta * tot[:, None] * (np.eye(K)[yh] - 1 / K); print("CE2 beta=%+.1f argmax==Bayes fraction %.4f" % (beta, (l.argmax(-1) == yh).mean()))
