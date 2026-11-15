exec(open("/tmp/review13b_math/check2_post.py").read().split("for beta in")[0])
for beta in (1.0, 0.6):
    env = GL.Env(3, 2, beta=beta); rng = np.random.default_rng(999 + int(10*beta)); n = 40000
    d = GL.sample(env, n, rng); X = d["X"]; q = rng.integers(0, 6, n); y = d["Y"][np.arange(n), q]
    for nm, P in (("FULL", XB.posterior_shared(env, X, q)), ("PERGROUP", censored_post(X, q, env, "PERGROUP"))):
        mx = P.max(-1, keepdims=True); tie = np.isclose(P, mx, rtol=1e-9, atol=1e-12); nt = tie.sum(-1)
        acc_rand = (tie[np.arange(n), y] / nt).mean(); acc_first = (P.argmax(-1) == y).mean()
        print(beta, nm, "tie frac", round(float((nt > 1).mean()), 4), "acc argmax-first", round(float(acc_first), 4), "acc random tiebreak (expected)", round(float(acc_rand), 4))
