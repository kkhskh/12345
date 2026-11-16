"""Exact reference values of the amended agreement index A_p (partial coefficient on the gate-free swap response, jointly with the Bayes
swap response) for the order-1 / order-2 optima, the gate-free rule and Bayes, on the score_X stream.  Writes logs/score_X_amend_refs.json."""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, gatelib as GL
import importlib.util
spec = importlib.util.spec_from_file_location("oc", os.path.join(HERE, "g_order_ceilings.py"))
rng = np.random.default_rng(98_500_000); dd = GL.TX.sample_gate(3000, rng); X = dd["X"]; Xw = GL.swap35(X)
def cen(P): L = np.log(P + 1e-12); return L - L.mean(-1, keepdims=True)
dG = cen(GL.rule("gatefree", Xw)) - cen(GL.rule("gatefree", X)); dBw = cen(GL.rule("bayes", Xw)) - cen(GL.rule("bayes", X)); mw = np.abs(dG).max(-1) > 0.05
def partial(dm): Z = np.stack([dG[mw].ravel(), dBw[mw].ravel()], 1); return float(np.linalg.lstsq(Z, dm[mw].ravel(), rcond=None)[0][0])
# order-k optima evaluated on the stream via a lookup over all 3^6 inputs
src = open(os.path.join(HERE, "g_order_ceilings.py")).read().split("out = {}")[0]; ns = {}; exec(src, ns)
idx = {tuple(x): i for i, x in enumerate(ns["Xall"])}; out = {}
for k in (1, 2):
    P, _ = ns["fit"](k); look = lambda Z: P[[idx[tuple(z)] for z in Z.reshape(-1, 6)]].reshape(Z.shape[:2] + (3,))
    out[f"order{k}"] = partial(cen(look(Xw)) - cen(look(X)))
out["bayes"] = partial(dBw); out["gatefree"] = partial(dG)
out["scaled_bayes_t0.1"] = partial(0.1 * dBw)
json.dump(out, open(os.path.join(HERE, "..", "logs", "score_X_amend_refs.json"), "w"), indent=1); print(out)
