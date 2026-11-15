import sys, numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:]
exec(open("pm_alt.py").read().split("specs = sys.argv[1:]")[0])
import builtins
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    L = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), X))[ii, q]; y = cenl(np.log(RP.sm(L) + 1e-12))
    pm = fit("PM", y, [[1, 1, -5, 1, 1.0], [0.8, 0.5, -8, 1, 0.5]]); errs = {"hat": ((fam("PM", pm["params"], te) - y[te]) ** 2).sum(1)}
    for r_ in (0.6, 0.9):
        globals()["RHO"] = r_; f_ = fit("PMfix", y, [pm["params"][:4]]); errs[r_] = ((fam("PMfix", f_["params"], te) - y[te]) ** 2).sum(1)
    ss = ((y[te] - y[te].mean()) ** 2).sum() / len(te); bs = np.random.default_rng(0)
    for r_ in (0.6, 0.9):
        d = (errs[r_] - errs["hat"]) / ss; B = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(sp[:6], "rho_hat", round(pm["params"][4], 3), f"R2(hat)-R2(rho={r_})", round(d.mean(), 5), "95% boot CI", np.round(np.quantile(B, [0.025, 0.975]), 5), flush=True)
