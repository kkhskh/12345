"""Evaluate any logits function X -> (N,6,3) on an evaluation set built by r1lib.build_eval."""
import numpy as np, r1lib as L
def run(fn, ev):
    lg = {}
    for k, (X, q) in L.eval_inputs(ev).items():
        A = fn(X); c = A[np.arange(len(X)), q]; lg[k] = c - c.mean(-1, keepdims=True)
        if k == "base": Lall = A - A.mean(-1, keepdims=True)
    return dict(quality=L.quality(ev, Lall), families=L.evaluate_logits(ev, lg))
def headline(r):
    F = r["families"]; g = lambda f, rg, k="E": F[f].get(rg, {}).get(k, float("nan"))
    return dict(KL=r["quality"]["KL"], acc=r["quality"]["acc"], R1=g("F45", "R1"), R2=g("F45", "R2"), R3=g("F45", "R3"), dm_R1=g("F45", "R1", "dmM"),
                F34R1=g("F34", "R1"), S45R1=g("S45", "R1"), S34R1=g("S34", "R1"), F54o=g("F54o", "all", "dmM"), F54oE=g("F54o", "all"),
                F0abs=g("F0", "all", "abs_dmM"), FLOCabs=g("FLOC", "all", "abs_dmM"), FIDabs=g("FID", "all", "abs_dmM"))
