import sys, numpy as np
exec(open("pm_alt.py").read().split("specs = sys.argv[1:]")[0])
R = {}; yB = cenl(np.log(GL.posterior_closed(RP.ENV1, X, q) + 1e-12))
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    L = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), X))[ii, q]; y = cenl(np.log(RP.sm(L) + 1e-12))
    m4 = fit("MIX4", y, [[1, 1, -5, 1], [0.8, 0.5, -8, 1]]); pm = fit("PM", y, [[1, 1, -5, 1, 1.0], [0.8, 0.5, -8, 1, 0.5]])
    R[sp[:6]] = dict(bayes=(y - yB)[te].ravel(), mix4=(y[te] - fam("MIX4", m4["params"], te)).ravel(), pm=(y[te] - fam("PM", pm["params"], te)).ravel())
    print(sp[:6], "fitted", flush=True)
ks = list(R)
for kind in ("bayes", "mix4", "pm"):
    print(kind, {f"{a}|{b}": round(float(np.corrcoef(R[a][kind], R[b][kind])[0, 1]), 3) for i, a in enumerate(ks) for b in ks[i + 1:]})
