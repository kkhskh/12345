"""XV diagnostics against exact references (one condition).  References over all 729 inputs: Bayes, the exact gate-free mixture, and
the expected-NLL-optimal order-1 and order-a (= best rule without the gate) log-linear rules; cached in logs/refs_<cond>.npz.
Edits: gate flip (last gate source redrawn to switch g), swap(4,5) (pairing change, multiset of evidence values preserved).
G   = gate-flip fidelity slope of the model on the Bayes response (order <= a rules: 0 exactly; Bayes 1).
A_p = partial agreement index: coefficient on the gate-free rule's swap response in the least-squares regression of the model's swap
      response on (gate-free, Bayes, order-1 optimum) responses, over swaps where the gate-free rule changes (max |d| > 0.05)."""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import numpy as np, xvtask as T
K = 3; IDX = np.array([243, 81, 27, 9, 3, 1])
def parse(cond): a, b, nu = cond.split(":"); return int(a), float(b), float(nu)
def refs(cond, cache_dir):
    f = os.path.join(cache_dir, "refs_" + cond.replace(":", "_") + ".npz")
    if os.path.exists(f): z = np.load(f); return {k: z[k] for k in z.files}
    import importlib.util; spec = importlib.util.spec_from_file_location("xm", os.path.join(HERE, "..", "scripts", "xv_math_core.py")); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    a, b, nu = parse(cond); px, PB = T.joint(a, b, nu); R = dict(px=px, bayes=PB, gatefree=T.gatefree(T.XALL, a, b, nu))
    R["order1"], _ = M.fit(1, px, PB); R["ordera"], _ = M.fit(a, px, PB); np.savez(f, **R); return R
def lookup(R, name, X): return R[name][(X * IDX).sum(-1)]
def cen(P): L = np.log(P + 1e-12); return L - L.mean(-1, keepdims=True)
class Diag:
    def __init__(self, cond, stream, n, cache_dir):
        self.a, self.b, self.nu = parse(cond); rng = np.random.default_rng(stream); d = T.sample(n, rng, self.a, self.b, self.nu)
        self.X, self.Y = d["X"], d["Y"]; self.Xf = T.gate_flip(self.X, self.a, rng); self.Xw = T.swap45(self.X); self.R = refs(cond, cache_dir)
        L = lambda nm, X: lookup(self.R, nm, X)
        self.PB = L("bayes", self.X); self.dB = cen(L("bayes", self.Xf)) - cen(self.PB)
        self.dG = cen(L("gatefree", self.Xw)) - cen(L("gatefree", self.X)); self.dBw = cen(L("bayes", self.Xw)) - cen(self.PB); self.dO = cen(L("order1", self.Xw)) - cen(L("order1", self.X))
        self.mw = np.abs(self.dG).max(-1) > 0.05
    def score(self, P0, Pf, Pw):
        dmf, dmw = cen(Pf) - cen(P0), cen(Pw) - cen(P0)
        G = float((dmf * self.dB).sum() / (self.dB ** 2).sum())
        Z = np.stack([self.dG[self.mw].ravel(), self.dBw[self.mw].ravel(), self.dO[self.mw].ravel()], 1); c = np.linalg.lstsq(Z, dmw[self.mw].ravel(), rcond=None)[0]
        return dict(NLL=float(-(self.PB * np.log(P0 + 1e-12)).sum(-1).mean()), acc=float((P0.argmax(-1) == self.Y).mean()),
                    KL_bayes=float((self.PB * (np.log(self.PB + 1e-12) - np.log(P0 + 1e-12))).sum(-1).mean()), G=G, A_p=float(c[0]))
    def ref_scores(self):
        out = {}
        for nm in ("order1", "ordera", "gatefree", "bayes"):
            f = lambda X: lookup(self.R, nm, X); out[nm] = self.score(f(self.X), f(self.Xf), f(self.Xw))
        return out
