"""Track B functional battery for one checkpoint (v2 after reviews A B5 and B BLK-2, MAJ-2/3/4).
python -m cor_research.scoring.functional RUN_DIR PARENT STEP OUT.json [--quick]
All inputs are on-support (natural T4 episodes or role-preserving hybrids); sets come from the frozen streams (protocol TRACK_B.sets).
  precision   KL, NLL, accuracy, ECE, Bayes regret; precision family F0' = (1-eps) softmax(tau log Bayes(alpha')) + eps/K with alpha' profiled
              on a grid (0.60 ... 0.95), fitted on half A, evaluated on half B (held-out KL)
  rules       rule-mixture fits over F0' (descriptive only, HB_a): Q = (1-eps) softmax(tau log((1-w) Bayes(alpha'*) + w P_k)) + eps/K
  nulls       parametric-precision null Bayes(alpha_null) with KL(Bayes || null) = the model's KL, and an iid logit-noise null at the same KL;
              the same rule-mixture fits on each null (HB_b, descriptive)
  historical  nested power-mean with lapse (xviiim.rho_fit_lapse; descriptive, non-identifiable parameters)
  decisions   on the on-support discriminating set (saved by trackb_run for HB_c / HB_c_dir; calibration-invariant)
  interchange other-question positions at the residual input of layers 0..3 patched from the HYBRID run x_h (source-matched: x_h has the
              base's own-question reports, so Bayes(x_h) is the correct ideal reference at every layer); secondary: from the donor run x'
              (unmatched, labelled); source-matched null: patching from the base run itself must reproduce the base logits (max diff
              reported); IIA_l (decision-changing pairs) and IIA_l normalised by the model's own following of Bayes(x_h) on the hybrid input.
Every number is descriptive for this checkpoint; class comparisons happen only in the frozen Track B scorer."""
import os, sys, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E
E.configure(); E.legacy_paths()
import numpy as np
from scipy.optimize import minimize
from cor_research.generators import t4 as T4
FROZEN = dict(nat_stream=91_400_000, disc_stream=91_410_000, pair_stream=91_420_000, n_nat=12000, n_disc_pool=40000, n_pairs=3000)
ALPHA_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90, 0.95]   # covers the null's bisection range (review B r2 N7)


def kl(P, Q): return float((P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1).mean())
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def sig(x): return 1 / (1 + np.exp(-x))


def family(th, Pbase, Pk=None):
    tau, eps = np.exp(th[0]), sig(th[1])
    base = np.log(Pbase + 1e-12) if Pk is None else np.log((1 - sig(th[2])) * Pbase + sig(th[2]) * Pk + 1e-12)
    return (1 - eps) * softmax(tau * base) + eps / 3


def fit(Pm, Pbase, Pk=None):
    obj = lambda th: kl(Pm, family(th, Pbase, Pk)); best = None
    for w0 in ([-4.0] if Pk is None else [-4.0, -1.0, 1.0]):
        r = minimize(obj, np.array([0.0, -5.0] + ([] if Pk is None else [w0])), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
        if best is None or r.fun < best.fun: best = r
    return best.x, float(best.fun)


def ece(P, PB, bins=10):
    c = P.max(-1); a = PB[np.arange(len(P)), P.argmax(-1)]; e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c >= lo) & (c < lo + 1 / bins)
        if m.any(): e += m.mean() * abs(c[m].mean() - a[m].mean())
    return float(e)


def rule_fits(Pm, R, Pa, A, B):
    """precision family F0' (alpha' profiled) and rule mixtures over Bayes(alpha'*); fitted on A, held-out KL on B"""
    iA = np.where(A)[0]; A1, A2 = iA[: len(iA) // 2], iA[len(iA) // 2:]; prof = {}          # fit on A1, select alpha* on A2, evaluate on B (review B r2 N7)
    for a, Pb in Pa.items(): th, _ = fit(Pm[A1], Pb[A1]); prof[a] = (kl(Pm[A2], family(th, Pb[A2])), th)
    a_star = min(prof, key=lambda a: prof[a][0]); Pb = Pa[a_star]; th0, _ = fit(Pm[A], Pb[A]); prof[a_star] = (kl(Pm[B], family(th0, Pb[B])), th0)
    out = dict(F0p=dict(alpha_prime=a_star, tau=float(np.exp(th0[0])), eps=float(sig(th0[1])), KL_heldout=prof[a_star][0], profile={str(a): v[0] for a, v in prof.items()}), rules={})
    for k in T4.RULES:
        th, _ = fit(Pm[A], Pb[A], R[k][A]); KLk = kl(Pm[B], family(th, Pb[B], R[k][B]))
        out["rules"][k] = dict(w=float(sig(th[2])), KL_heldout=KLk, dKL_heldout=out["F0p"]["KL_heldout"] - KLk)
    return out


def battery(Pm, R, Pa, Y=None):
    n = len(Pm); A, B = np.arange(n) < n // 2, np.arange(n) >= n // 2; PB = R["bayes"]; out = {}
    out["KL"] = kl(PB, Pm); out["KL_model_to_bayes"] = kl(Pm, PB); out["acc_bayes_argmax"] = float((Pm.argmax(-1) == PB.argmax(-1)).mean())
    out["bayes_regret"] = float((PB.max(-1) - PB[np.arange(n), Pm.argmax(-1)]).mean()); out["ECE"] = ece(Pm, PB)
    out["NLL_expected"] = float(-(PB * np.log(Pm + 1e-12)).sum(-1).mean()); out["NLL_bayes"] = float(-(PB * np.log(PB + 1e-12)).sum(-1).mean())
    if Y is not None: out["acc_true"] = float((Pm.argmax(-1) == Y).mean())
    out.update(rule_fits(Pm, R, Pa, A, B)); return out


def nulls(R, Pa, target_kl, X, q, seed=7):
    """(i) parametric-precision null: Bayes(alpha) with KL(Bayes || Bayes(alpha)) = target (bisection over alpha < 0.8);
    (ii) iid logit-noise null at the same KL.  Each gets the same rule-mixture fits."""
    import genlib as GL
    lo, hi = 0.40, 0.80
    for _ in range(25):
        a = 0.5 * (lo + hi); P = T4.bayes(X, q, GL.Env(3, 1, alpha=a)); lo, hi = (lo, a) if kl(R["bayes"], P) < target_kl else (a, hi)
    a = 0.5 * (lo + hi); Pnull = T4.bayes(X, q, GL.Env(3, 1, alpha=a)); n = len(q); A, B = np.arange(n) < n // 2, np.arange(n) >= n // 2
    par = rule_fits(Pnull, R, Pa, A, B); par["alpha_null"] = a; par["KL"] = kl(R["bayes"], Pnull)
    lB = np.log(R["bayes"] + 1e-12); z0 = np.random.default_rng(seed).standard_normal(lB.shape); lo, hi = 0.0, 3.0
    for _ in range(30):
        s = 0.5 * (lo + hi); lo, hi = (s, hi) if kl(R["bayes"], softmax(lB + s * z0)) < target_kl else (lo, s)
    Pn = softmax(lB + 0.5 * (lo + hi) * z0); iid = rule_fits(Pn, R, Pa, A, B); iid["sigma"] = 0.5 * (lo + hi)
    return dict(parametric=par, iid=iid)


def interchange(M, P):
    import jax.numpy as jnp, xviiim as XM
    F = XM.F; n = len(P["q"]); q = P["q"]; I = np.arange(n); out = {}
    pos_other = np.zeros((n, 43), bool)
    for qq in range(6):
        m = q != qq; pos_other[m, 6 * qq:6 * qq + 6] = True; pos_other[m, 36 + qq] = True
    pos_read = np.zeros((n, 43), bool); pos_read[I, 36 + q] = True
    def rec_run(X): rec = {}; L = np.asarray(F.forward(M, jnp.asarray(X), rec=rec)); return L, rec
    def patched(X, layer, donor, mask): return np.asarray(F.forward(M, jnp.asarray(X), resid_patch={layer: (lambda x: jnp.where(jnp.asarray(mask)[..., None], donor, x))}))
    chg = P["PB"].argmax(-1) != P["PBh"].argmax(-1); out["n_pairs"] = int(n); out["n_decision_changing"] = int(chg.sum())
    kinds = ("OQI", "OQI_unmatched", "RS", "NULL_base"); res = {f"{k}_{l}": [] for k in kinds for l in range(4)}; base_l, hyb_l = [], []; null_maxdiff = 0.0
    for s in range(0, n, 500):
        sl = slice(s, s + 500); Lb, rb = rec_run(P["X"][sl]); Lh, rh = rec_run(P["Xh"][sl]); _, rd = rec_run(P["Xd"][sl]); qq = q[sl]; ii = np.arange(len(qq))
        base_l.append(Lb[ii, qq]); hyb_l.append(Lh[ii, qq])
        for l in range(4):
            k = f"resid{l}"
            res[f"OQI_{l}"].append(patched(P["X"][sl], l, rh[k], pos_other[sl])[ii, qq])
            res[f"OQI_unmatched_{l}"].append(patched(P["X"][sl], l, rd[k], pos_other[sl])[ii, qq])
            res[f"RS_{l}"].append(patched(P["X"][sl], l, rh[k], pos_read[sl])[ii, qq])
            Ln = patched(P["X"][sl], l, rb[k], pos_other[sl])[ii, qq]; res[f"NULL_base_{l}"].append(Ln); null_maxdiff = max(null_maxdiff, float(np.abs(Ln - Lb[ii, qq]).max()))
    sm = XM.RP.sm; Pb = sm(np.concatenate(base_l)); Ph = sm(np.concatenate(hyb_l))
    follow = float((Ph.argmax(-1) == P["PBh"].argmax(-1))[chg].mean()) if chg.any() else float("nan")
    out["model_follow_on_hybrid_input"] = follow; out["null_base_patch_max_logit_diff"] = null_maxdiff
    for k, v in res.items():
        Pp = sm(np.concatenate(v)); iia = float((Pp.argmax(-1) == P["PBh"].argmax(-1))[chg].mean()) if chg.any() else float("nan")
        out[k] = dict(IIA=iia, IIA_norm=(iia / follow if follow and follow > 0 else float("nan")), stay_base=float((Pp.argmax(-1) == P["PB"].argmax(-1))[chg].mean()) if chg.any() else float("nan"),
                      KL_to_hybrid_bayes=kl(P["PBh"], Pp), KL_to_base_bayes=kl(P["PB"], Pp))
    return out


def run_checkpoint(run_dir, parent, step, cfg=FROZEN, want_decisions=False):
    from cor_research.models import load as LD
    import xviiim as XM, genlib as GL
    t = time.time(); M = LD.load(run_dir, parent, step); f = LD.forward_fn(M)
    X, q, r, Y = T4.sample(cfg["n_nat"], cfg["nat_stream"]); R = T4.rules(X, q); I = np.arange(len(q)); Pa = {a: T4.bayes(X, q, GL.Env(3, 1, alpha=a)) for a in ALPHA_GRID}
    Ln = f(X)[I, q]; Pm = XM.RP.sm(Ln); nat = battery(Pm, R, Pa, Y[I, q]); nat["nulls"] = nulls(R, Pa, nat["KL"], X, q)
    try: nat["historical_powermean"] = XM.rho_fit_lapse(Ln[:3000], dict(Xn=X[:3000], qn=q[:3000]))
    except Exception as e: nat["historical_powermean"] = dict(error=str(e)[:200])
    Xd, qd, info = T4.discriminating(cfg["n_disc_pool"], cfg["disc_stream"]); Rd = T4.rules(Xd, qd); Pd = XM.RP.sm(f(Xd)[np.arange(len(qd)), qd])
    disc = dict(info=info, model_agree={k: float((Pd.argmax(-1) == Rd[k].argmax(-1)).mean()) for k in ("bayes",) + T4.RULES}, KL=kl(Rd["bayes"], Pd),
                note="decision agreement is invariant to monotone output calibration (temperature, lapse)")
    P = T4.relational_pairs(cfg["n_pairs"], cfg["pair_stream"]); ic = interchange(M, P)
    rec = dict(run=os.path.basename(os.path.normpath(run_dir)), parent=parent, step=int(step), natural=nat, discriminating=disc, interchange=ic, frozen_sets=cfg,
               ckpt_sha256=E.sha256(os.path.join(run_dir, f"step{step}.npz")), code_sha256=code_hashes(), seconds=time.time() - t)
    if want_decisions: rec["_decisions"] = dict(dec=Pd.argmax(-1).astype(np.int8), rule_dec={k: T4.decide(Rd[k]).astype(np.int8) for k in ("bayes",) + T4.RULES})   # -1 = tie
    return rec


def code_hashes():
    from cor_research.scoring import measure as MS
    return dict(MS.code_hashes(), **{"functional.py": E.sha256(os.path.abspath(__file__)), "t4.py": E.sha256(T4.__file__)})


if __name__ == "__main__":
    d, parent, step, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    cfg = dict(FROZEN)
    if "--quick" in sys.argv: cfg.update(n_nat=1200, n_disc_pool=4000, n_pairs=300)       # smoke tests only
    rec = run_checkpoint(d, parent, step, cfg); E.atomic_json(rec, out)
    print(json.dumps(dict(run=rec["run"], step=step, KL=rec["natural"]["KL"], F0p=rec["natural"]["F0p"]["alpha_prime"], w={k: round(v["w"], 3) for k, v in rec["natural"]["rules"].items()},
                          IIA={k: round(v["IIA"], 3) for k, v in rec["interchange"].items() if isinstance(v, dict) and k.startswith("OQI_")}, null=rec["interchange"]["null_base_patch_max_logit_diff"], sec=round(rec["seconds"], 1))))
