"""XIX measurement and classification primitives (to be frozen).  No fitted behavioural parameter enters evaluability (mandate §9 F1).
  necessity(M, stream)  : KL to Bayes on 1500 natural episodes, with no block / E-route block (layer-0 readout -> own-question sources, own0)
                          / M-route block (layer-2 readout -> own-question sources, own2); nec_r = KL(blocked) - KL(none); acc = Bayes-argmax agreement
  causal_class          : EARLY if nec_E >= 0.30 and nec_M <= 0.10; MIXTURE if nec_M >= 0.30 and nec_E <= 0.15; else HYBRID
  competent             : KL <= 0.045 and acc >= 0.80 (absolute; no baseline-relative accuracy rule)
  profile(M, stream)    : residual-to-Bayes log-probability profile on 1500 natural + 1500 relational-edit episodes (centred per episode)
  functional_score      : corr(profile, mean early reference profile) - corr(profile, mean mixture reference profile)
  functional_class      : EARLY if f >= +0.05; MIXTURE if f <= -0.05; else AMBIGUOUS
  phi_nec(arm, B, I)    : 0.5 * sum_r [l(nec_r^arm) - l(nec_r^B)] / [l(nec_r^I) - l(nec_r^B)], l(x) = log(max(x, 0) + 0.01)"""
import os, sys, math
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import numpy as np, jax
import xviiim as XM
MF = XM.MFW
_SETS = {}
def _nat(stream):
    if ("nat", stream) not in _SETS:
        rng = np.random.default_rng(stream + 7); X = XM.NV.Bm.sample_fast(XM.NV.ENV, 1500, rng)["X"]; q = rng.integers(0, 6, 1500); _SETS[("nat", stream)] = (X, q, XM.GL.posterior_closed(XM.RP.ENV1, X, q))
    return _SETS[("nat", stream)]
def necessity(M, stream):
    X, q, PB = _nat(stream); i = np.arange(len(q))
    def kl(L): P = XM.RP.sm(np.asarray(L)[i, q]); return float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()), float((P.argmax(-1) == PB.argmax(-1)).mean())
    k0, acc = kl(MF.forward(M, X)); kE, _ = kl(MF.forward(M, X, biases=dict([MF.edges("own0")]))); kM, _ = kl(MF.forward(M, X, biases=dict([MF.edges("own2")])))
    return dict(KL=k0, acc=acc, nec_E=kE - k0, nec_M=kM - k0)
def causal_class(n): return "EARLY" if (n["nec_E"] >= 0.30 and n["nec_M"] <= 0.10) else ("MIXTURE" if (n["nec_M"] >= 0.30 and n["nec_E"] <= 0.15) else "HYBRID")
def competent(n): return bool(n["KL"] <= 0.045 and n["acc"] >= 0.80)
def profile(M, stream):
    if ("prof", stream) not in _SETS:
        S = XM.MT.sets(stream, n=1500); PR = S["PR"]; Xp = np.concatenate([S["Xn"][:1500], PR["Xd"]]); qp = np.concatenate([S["qn"][:1500], PR["q"]]); PBp = np.concatenate([S["PBn"][:1500], PR["PBd"]])
        _SETS[("prof", stream)] = (Xp, qp, PBp)
    Xp, qp, PBp = _SETS[("prof", stream)]; f = jax.jit(lambda X_: XM.F.forward(M, X_)); L = np.asarray(XM.F.batched(f, Xp))[np.arange(len(qp)), qp]
    R = np.log(XM.RP.sm(L) + 1e-9) - np.log(PBp + 1e-9); R = R - R.mean(-1, keepdims=True); return R.ravel()
def functional_score(prof, early_refs, mix_refs):
    cE = np.mean(list(early_refs.values()), 0); cM = np.mean(list(mix_refs.values()), 0); return float(np.corrcoef(prof, cE)[0, 1] - np.corrcoef(prof, cM)[0, 1])
def functional_class(f): return "EARLY" if f >= 0.05 else ("MIXTURE" if f <= -0.05 else "AMBIGUOUS")
ell = lambda x: math.log(max(x, 0.0) + 0.01)
def phi_nec(a, b, i):
    v = []
    for r in ("E", "M"):
        den = ell(i[f"nec_{r}"]) - ell(b[f"nec_{r}"])
        if abs(den) < 1e-9: return float("nan")
        v.append((ell(a[f"nec_{r}"]) - ell(b[f"nec_{r}"])) / den)
    return 0.5 * sum(v)
