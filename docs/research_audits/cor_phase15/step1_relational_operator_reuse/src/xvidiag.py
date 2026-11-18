"""XVI diagnostics against exact references (one condition, S = 6 neural format).  Rebuilt after the XV reviews (mandate section 6).

References over all 729 inputs (cached in logs/refs_<cond>.npz): Bayes, the exact gate-free mixture, and the expected-NLL-optimal
order-1 and order-a log-linear rules (order a = the best rule that cannot use the gate; exact KL from Bayes = the gate's value).
Edits on the evaluation set: gate flip (last gate source redrawn so that g switches) and pair-single swap (XV's swap(4,5)).

Decision-level measures (invariant to any confidence change that preserves the model's own argmax; mandate D3):
  F_G  gate-flip following: over flips that change the Bayes argmax (unique before and after), the fraction on which the model's
       argmax moves to the new Bayes argmax.  Every rule that ignores the gate has F_G = 0; Bayes 1.
  (A decision-level agreement measure was tried in development and rejected: every pair-single swap that changes the order-a
   optimum's argmax also changes the order-1 optimum's, so following cannot separate agreement use from source-wise use.)
Graded measures (confidence dependent; reported with confidence controls):
  G    XV's gate-flip slope on the Bayes response; A_p XV's partial agreement index.
  sB   least-squares slope of the model's centred log-probabilities on Bayes's (overall sharpness relative to Bayes);
  G_conf = G / sB (a tempered Bayes has G_conf = 1 at every temperature; a gate-free rule 0).
  A_conf = A_p * sB(order-a optimum) / sB: XV's agreement index normalised for confidence (a tempered order-a optimum has A_conf = 1
       at every temperature; order-1 and Bayes 0).  Undefined (None) when sB <= 0.05.
  conf mean max-probability; ent mean entropy (nats).
  KL   expected KL(Bayes || model); NLL against the Bayes posterior; acc against sampled labels."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import numpy as np, xvitask as T
K = 3; IDX = np.array([243, 81, 27, 9, 3, 1])


def parse(cond):
    a, b1, b0, nu = cond.split(":"); return int(a), float(b1), float(b0), float(nu)


def refs(cond, cache_dir):
    f = os.path.join(cache_dir, "refs_" + cond.replace(":", "_") + ".npz")
    if os.path.exists(f): z = np.load(f); return {k: z[k] for k in z.files}
    import xvi_math_core as M
    a, b1, b0, nu = parse(cond); px, PB = T.joint(a, b1, b0, nu); XA = T.xall(6)
    R = dict(px=px, bayes=PB, gatefree=T.gatefree(XA, a, b1, b0, nu))
    R["order1"], _ = M.fit(1, px, PB, XA); R["ordera"], _ = M.fit(a, px, PB, XA); np.savez(f, **R); return R


def lookup(R, name, X): return R[name][(X * IDX).sum(-1)]
def cen(P): L = np.log(P + 1e-12); return L - L.mean(-1, keepdims=True)
def unique(P, tol=1e-6): s = np.sort(P, -1); return s[..., -1] - s[..., -2] > tol
def temper(P, t): L = np.log(P + 1e-15) / t; L -= L.max(-1, keepdims=True); E = np.exp(L); return E / E.sum(-1, keepdims=True)


class Diag:
    def __init__(self, cond, stream, n, cache_dir):
        self.cond = cond; self.a, self.b1, self.b0, self.nu = parse(cond); rng = np.random.default_rng(stream)
        d = T.sample(n, rng, self.a, self.b1, self.b0, self.nu); self.X, self.Y = d["X"], d["Y"]
        self.Xf = T.gate_flip(self.X, self.a, rng); self.Xw = T.swap_pair_single(self.X); self.R = refs(cond, cache_dir)
        L = lambda nm, X: lookup(self.R, nm, X)
        self.PB = L("bayes", self.X); self.PBf = L("bayes", self.Xf); self.dB = cen(self.PBf) - cen(self.PB)
        self.dG = cen(L("gatefree", self.Xw)) - cen(L("gatefree", self.X)); self.dBw = cen(L("bayes", self.Xw)) - cen(self.PB)
        self.dO = cen(L("order1", self.Xw)) - cen(L("order1", self.X)); self.mw = np.abs(self.dG).max(-1) > 0.05
        self.decG = unique(self.PB) & unique(self.PBf) & (self.PB.argmax(-1) != self.PBf.argmax(-1)); self.newG = self.PBf.argmax(-1)
        self.sBa = 1.0; self.sBa = self._sB(L("ordera", self.X))

    def _sB(self, P0): c = cen(self.PB).ravel(); return float((cen(P0).ravel() * c).sum() / (c ** 2).sum())

    def score(self, P0, Pf, Pw):
        dmf, dmw = cen(Pf) - cen(P0), cen(Pw) - cen(P0)
        G = float((dmf * self.dB).sum() / (self.dB ** 2).sum())
        Z = np.stack([self.dG[self.mw].ravel(), self.dBw[self.mw].ravel(), self.dO[self.mw].ravel()], 1); c = np.linalg.lstsq(Z, dmw[self.mw].ravel(), rcond=None)[0]
        a0, af = P0.argmax(-1), Pf.argmax(-1)
        FG = float(((af != a0) & (af == self.newG))[self.decG].mean()); sB = self._sB(P0)
        return dict(NLL=float(-(self.PB * np.log(P0 + 1e-12)).sum(-1).mean()), acc=float((a0 == self.Y).mean()),
                    KL=float((self.PB * (np.log(self.PB + 1e-12) - np.log(P0 + 1e-12))).sum(-1).mean()), G=G, A_p=float(c[0]),
                    F_G=FG, sB=sB, G_conf=(G / sB if sB > 0.05 else None), A_conf=(float(c[0]) * self.sBa / sB if sB > 0.05 else None),
                    conf=float(P0.max(-1).mean()), ent=float(-(P0 * np.log(P0 + 1e-12)).sum(-1).mean()))

    def ref_scores(self):
        out = {}
        for nm in ("order1", "ordera", "gatefree", "bayes"):
            f = lambda X: lookup(self.R, nm, X); out[nm] = self.score(f(self.X), f(self.Xf), f(self.Xw))
        for t in (0.5, 2.0):      # confidence-sensitive null models (mandate D3): tempered Bayes and tempered gate-free optimum
            for nm in ("bayes", "ordera"):
                f = lambda X: temper(lookup(self.R, nm, X), t); out[f"{nm}_T{t}"] = self.score(f(self.X), f(self.Xf), f(self.Xw))
        out["n_decisive_flips"] = int(self.decG.sum()); out["sB_ordera"] = self.sBa
        return out
