"""Direction C/D/E exact Bayesian boundaries. Writes logs/bayes_boundaries.json and prints a summary.
B0  count-sufficiency in the extended family (rho<1, pc mixture, correlated honest errors kappa)
B1  Theorem U (unimodality): P(role | own count, others' counts) is quasi-concave in the own count for every role,
    monotone for the extreme-agreement roles; checked exhaustively over count triples and a parameter grid
B2  phase diagram of the agreement -> reliability response (monotone / interior peak / boundary drop)
B3  Theorem D (exact margin-change identity) checked against brute force on natural single-report edits
B4  identifiability: noise re-pairing invariance; honest-copy vs colluder exchangeability; unknown-pairing Bayes cost"""
import itertools, json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import gen_family as G

out = {}
def P(*a): print(*a, flush=True)

# ---------------------------------------------------------------- helpers: exact posterior over configs from counts
def role_ll(f, n, M):
    ap = G.agree_probs(f)
    def ll(a):
        with np.errstate(divide="ignore"):
            return n * np.log(a) + (M - n) * np.log1p(-a) if a < 1 else np.where(n == M, 0.0, -np.inf)
    return {r: ll(ap[r]) for r in ["rel", "noise", "coll"]}

def post_roles_from_counts(f, counts, M):
    """counts (...,3) -> P(config | counts) (...,C) using only agreement counts (Theorem S, extended)."""
    lls = [role_ll(f, counts[..., g], M) for g in range(3)]
    lp = []
    for (g, h), pr in zip(f.configs, f.prior):
        s = np.log(pr)
        for k in range(3):
            r = "rel" if k == g else ("coll" if k == h else "noise"); s = s + lls[k][r]
        lp.append(s)
    lp = np.stack(lp, -1); m = lp.max(-1, keepdims=True); w = np.exp(lp - m); return w / w.sum(-1, keepdims=True)

def p_role(f, post, g, role):
    idx = [i for i, (a, b) in enumerate(f.configs) if (role == "rel" and a == g) or (role == "coll" and b == g) or (role == "noise" and a != g and b != g)]
    return post[..., idx].sum(-1)

# ---------------------------------------------------------------- B0 sufficiency in the extended family
b0 = {}
for nm, f in {"rho0.8_pc0.5_kappa0.3": G.Fam(rho=0.8, pc=0.5, kappa=0.3), "rho1_pc1": G.Fam(), "rho0.6_pc0.2": G.Fam(rho=0.6, pc=0.2)}.items():
    d = G.sample(f, 4000, np.random.default_rng(11)); X = d["X"]
    full = G.posterior(f, X)["post_cfg"]
    A = np.stack([(X[:, :, a] == X[:, :, b]) for a, b in G.GROUPS], -1).sum(1)
    cnt = post_roles_from_counts(f, A, f.Q)
    b0[nm] = float(np.nanmax(np.abs(full - cnt)))
out["B0_count_sufficiency_max_err"] = b0; P("B0", b0)

# ---------------------------------------------------------------- B1 + B2 over a grid
grid = list(itertools.product([1.0, 0.95, 0.9, 0.8, 0.7, 0.5], [0.0, 0.05, 0.2, 0.5, 1.0], [0.6, 0.8, 0.95], [2, 3, 5], [3, 6, 10], [0.0, 0.3]))
viol = 0; checked = 0; phase = []
for rho, pc, alpha, K, Q, kappa in grid:
    f = G.Fam(K=K, Q=Q, alpha=alpha, rho=rho, pc=pc, kappa=kappa)
    ap = G.agree_probs(f)
    if not (ap["noise"] < ap["rel"]):
        continue                                           # degenerate: reliable not more agreeing than noise
    M = Q - 1
    n = np.arange(M + 1)
    others = np.array(list(itertools.product(range(M + 1), repeat=2)))
    C = np.zeros((len(others), M + 1, 3)); C[:, :, 0] = n[None, :]; C[:, :, 1] = others[:, None, 0]; C[:, :, 2] = others[:, None, 1]
    post = post_roles_from_counts(f, C, M)
    pr = p_role(f, post, 0, "rel")                         # (others, M+1)
    ok = np.isfinite(pr).all(1) & ~np.isnan(pr).any(1)
    # quasi-concavity: no interior strict local minimum (increase after a decrease)
    dlt = np.diff(pr, axis=1); tol = 1e-12
    for row, okr in zip(dlt, ok):
        if not okr: continue
        checked += 1
        s = np.sign(np.where(np.abs(row) < tol, 0, row)); s = s[s != 0]
        if len(s) > 1 and np.any((s[:-1] < 0) & (s[1:] > 0)): viol += 1
    # weight the 'others' rows by their probability of occurring next to a group of each role (natural episodes)
    d = G.sample(f, 3000, np.random.default_rng(5)); A = np.stack([(d["X"][:, :, a] == d["X"][:, :, b]) for a, b in G.GROUPS], -1)
    loo = (A.sum(1)[:, None, :] - A).reshape(-1, 3)        # leave-one-out counts, rows = (episode, question)
    key = loo[:, 1] * (M + 1) + loo[:, 2]; wts = np.bincount(key, minlength=(M + 1) ** 2).astype(float)
    wts = wts / wts.sum()
    inc = np.all(dlt >= -1e-12, 1); drop_top = dlt[:, -1] < -1e-12
    peak = np.argmax(np.nan_to_num(pr, nan=-1), 1)
    phase.append(dict(rho=rho, pc=pc, alpha=alpha, K=K, Q=Q, kappa=kappa, a_rel=ap["rel"], a_noise=ap["noise"], a_coll=ap["coll"],
                      frac_monotone_increasing=float((wts * inc).sum()), frac_drop_at_full_agreement=float((wts * drop_top).sum()),
                      mean_peak=float((wts * peak).sum()), M=M))
out["B1_unimodality"] = dict(rows_checked=checked, violations=viol); P("B1 unimodality: rows", checked, "violations", viol)
out["B2_phase"] = phase
# compact print: K=3, Q=6, alpha=0.8, kappa=0
for r in phase:
    if r["K"] == 3 and r["Q"] == 6 and r["alpha"] == 0.8 and r["kappa"] == 0.0:
        P(f"  PH rho {r['rho']:.2f} pc {r['pc']:.2f}: monotone {r['frac_monotone_increasing']:.3f} drop@M {r['frac_drop_at_full_agreement']:.3f} peak {r['mean_peak']:.2f} (a_coll {r['a_coll']:.3f})")

# ---------------------------------------------------------------- B3 exact margin-change identity
f = G.Fam(); d = G.sample(f, 300, np.random.default_rng(21)); X0 = d["X"]; K = 3
errs = []
for q, k, g in [(0, 1, 0), (2, 4, 1), (5, 3, 2), (1, 0, 2)]:
    a, b = G.GROUPS[g]; X1 = X0.copy(); agree = X1[:, k, a] == X1[:, k, b]
    X1[:, k, b] = np.where(agree, (X1[:, k, a] + 1) % K, X1[:, k, a])
    p0, p1 = G.posterior(f, X0), G.posterior(f, X1)
    J0 = G.joint(f, X0)[:, q]                                      # N,C,K  (X_q unchanged by the edit)
    pim0, pim1 = p0["pi_minus"][:, q], p1["pi_minus"][:, q]        # N,C
    c1, c2 = 0, 1
    with np.errstate(divide="ignore", invalid="ignore"):
        dm_direct = (np.log(p1["post_y"][:, q, c1]) - np.log(p1["post_y"][:, q, c2])) - (np.log(p0["post_y"][:, q, c1]) - np.log(p0["post_y"][:, q, c2]))
        r = pim1 / pim0                                            # config-wise likelihood ratio of the edit (normalised)
        pic1 = pim0 * J0[..., c1]; pic1 /= pic1.sum(-1, keepdims=True)
        pic2 = pim0 * J0[..., c2]; pic2 /= pic2.sum(-1, keepdims=True)
        dm_formula = np.log(np.nansum(pic1 * r, -1)) - np.log(np.nansum(pic2 * r, -1))
    absc = ~((pim0 == 0) & (pim1 > 0)).any(-1)                     # identity requires absolute continuity of the edit
    okk = np.isfinite(dm_direct) & np.isfinite(dm_formula) & absc
    out.setdefault("B3_n_checked", []).append(int(okk.sum())); out.setdefault("B3_n_not_abs_continuous", []).append(int((~absc).sum()))
    errs.append(float(np.max(np.abs(dm_direct - dm_formula)[okk])) if okk.any() else None)
out["B3_margin_identity_max_err"] = errs; P("B3 margin identity max err", errs)

# ---------------------------------------------------------------- B4 identifiability
b4 = {}
# (a) re-pairing noise sources leaves P(X | config) unchanged: config (0, None) under pairing {01,23,45} vs {01,24,35}
f1 = G.Fam(pc=0.0); X = G.sample(f1, 2000, np.random.default_rng(3))["X"]
Ja = G.joint(f1, X, [(0, 1), (2, 3), (4, 5)])[:, :, 0].sum(-1); Jb = G.joint(f1, X, [(0, 1), (2, 4), (3, 5)])[:, :, 0].sum(-1)
b4["noise_repairing_max_abs_loglik_diff"] = float(np.max(np.abs(np.log(Ja) - np.log(Jb))))
# (b) honest-copy (kappa) vs colluder (rho) exchangeability when a_rel == a_coll
for kap, rho in [(1.0, 1.0), (0.5, None)]:
    if rho is None:
        a0 = G.agree_probs(G.Fam())["rel"]; arel = kap + (1 - kap) * a0; rho = (arel - 1 / 3) / (1 - 1 / 3)
    f = G.Fam(pc=1.0, rho=rho, kappa=kap)
    d = G.sample(f, 4000, np.random.default_rng(4)); X = d["X"]
    J = G.joint(f, X).sum(-1)                                           # N,Q,C  marginal over y
    cf = f.configs; sw = [cf.index((h, g)) for (g, h) in cf]             # exchange reliable and colluder roles
    ld = np.abs(np.log(J).sum(1) - np.log(J[:, :, sw]).sum(1))
    post = G.posterior(f, X)
    b4[f"exchange_kappa{kap}_rho{rho:.4f}"] = dict(a=G.agree_probs(f), max_abs_episode_loglik_diff=float(np.nanmax(ld)),
                                                  bayes_acc=float((post["post_y"].argmax(-1) == d["Y"]).mean()),
                                                  own_acc=float((post["own_y"].argmax(-1) == d["Y"]).mean()))
    # contrast: a nearby non-degenerate setting
    f2 = G.Fam(pc=1.0, rho=1.0, kappa=0.0); d2 = G.sample(f2, 4000, np.random.default_rng(4)); p2 = G.posterior(f2, d2["X"])
    b4["reference_T4_bayes_acc"] = float((p2["post_y"].argmax(-1) == d2["Y"]).mean())
# (c) unknown pairing: exact Bayes accuracy known vs unknown pairing
for nm, f in {"T1": G.Fam(pc=0.0), "T4": G.Fam(pc=1.0, rho=1.0)}.items():
    fp = G.Fam(pc=f.pc, rho=f.rho, perm=True)
    d = G.sample(fp, 1500, np.random.default_rng(8)); Xp = d["X"]
    pu = G.posterior_unknown(f, Xp)
    # known pairing: undo the permutation using the true perms
    inv = np.argsort(d["perms"], -1); Xk = np.take_along_axis(Xp, inv[:, None, :], -1)
    pk = G.posterior(f, Xk)
    true_pair = []  # posterior mass on the true pairing of the informative groups is reported via accuracy only
    b4[f"unknown_pairing_{nm}"] = dict(bayes_known=float((pk["post_y"].argmax(-1) == d["Y"]).mean()),
                                       bayes_unknown=float((pu["post_y"].argmax(-1) == d["Y"]).mean()),
                                       own=float((pk["own_y"].argmax(-1) == d["Y"]).mean()))
out["B4_identifiability"] = b4; P("B4", json.dumps(b4, indent=0, default=float)[:1500])
json.dump(out, open(os.path.join(HERE, "..", "logs", "bayes_boundaries.json"), "w"), indent=1, default=float)
