exec(open("/tmp/review13b_math/check2_post.py").read().split("for beta in")[0])
env = GL.Env(3, 2, beta=0.6); rng = np.random.default_rng(4242); n = 40000
d = GL.sample(env, n, rng); X = d["X"]; q = rng.integers(0, 6, n)
Pf = XB.posterior_shared(env, X, q); Pl = censored_post(X, q, env, "LOCALQ")
def best(P): mx = P.max(-1, keepdims=True); return np.isclose(P, mx, rtol=1e-9, atol=1e-12)
bf, bl = best(Pf), best(Pl)
print("argmax differs", (Pf.argmax(-1) != Pl.argmax(-1)).mean(), " tie-aware: best sets disjoint", (~(bf & bl).any(-1)).mean(), " best sets differ", (bf != bl).any(-1).mean())
