"""XVIII measurement library.  Loads XVIII (or XIV) fine-tuning checkpoints by directory and computes the three organisational diagnostics
(route-structural, behavioural, causal-necessity), competence, and the early-window route shares.  All definitions are XIV's/XIII-C's
(src/metrics.py, src/maskfwd.py of step1_cross_family_relational_foundations) except the classification rule below.

Classification (frozen in FREEZE_XVIII; development values in XIV_RESULT_RECONSTRUCTION.md §4):
  structural  : XIV cls1211 -- EARLY if mix_mass < 0.5 and lik_RO1 >= 0.4; MIXTURE if mix_mass >= 0.8 and lik_RO1 <= 0.15; else HYBRID
  behavioural : rho <= 0.60 -> EARLY; rho >= 0.66 -> MIXTURE; else HYBRID
  causal      : nec_E >= 0.30 and nec_M <= 0.10 -> EARLY; nec_M >= 0.30 and nec_E <= 0.15 -> MIXTURE; else HYBRID
  final       : the common label if all three agree, otherwise HYBRID (reported with the three components)
  competent   : KL(Bayes||model) <= 0.06 on the measurement stream."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); X14 = os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src"); sys.path.insert(0, X14)
import numpy as np, jax, jax.numpy as jnp
import native as NV, ftlib as F, metrics as MT, maskfwd as MFW, relpairs as RP, genlib as GL
NL, N = NV.NL, NV.N


def load_dir(D, parent, step):
    pf = NV.load_parent(parent); fresh = lambda s: N.init("M", NL.ENV, s)
    tmpl = dict(gf=pf["gf"], bf=pf["bf"], WA=fresh(7777)["WA"], new=[fresh(7777)["layers"][2]], emb={k: pf[k] for k in NL.FROZEN_KEYS}, pre=[pf["layers"][l] for l in range(2)])
    z = np.load(os.path.join(D, f"step{step}.npz"))
    t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tmpl), [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])
    return dict(tag=os.path.basename(D) + f"@{step}", parent=parent, spec="full_c1", emb=t["emb"], layers=list(t["pre"]) + [t["new"][0]], gf=t["gf"], bf=t["bf"], WA=t["WA"], dir=D, step=step)


def nat(stream, n=1500):
    rng = np.random.default_rng(stream + 7); X = NV.Bm.sample_fast(NV.ENV, n, rng)["X"]; q = rng.integers(0, 6, n); return X, q, GL.posterior_closed(RP.ENV1, X, q)


def kl_of(L, q, PB):
    i = np.arange(len(q)); P = RP.sm(np.asarray(L)[i, q]); return float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean())


def necessity(M, NAT):
    X, q, PB = NAT; base = kl_of(MFW.forward(M, X), q, PB)
    kE = kl_of(MFW.forward(M, X, biases=dict([MFW.edges("ro0")])), q, PB); kM = kl_of(MFW.forward(M, X, biases=dict([MFW.edges("own2")])), q, PB)
    k0 = kl_of(MFW.forward(M, X, biases=dict([MFW.edges("own0")])), q, PB)
    return dict(KL_nat=base, nec_E=kE - base, nec_M=kM - base, nec_E_own0=k0 - base)


def shares(M, X):
    """Mean readout attention share on the route edges: layer-0 own-q, layer-0 all sources, layer-2 own-q (mean and max over heads)."""
    rec = {}; F.forward(M, jnp.asarray(X), rec=rec); out = {}
    own = np.zeros((6, 43), bool)
    for q in range(6): own[q, q * 6:(q + 1) * 6] = True
    for l in (0, 2):
        A = np.asarray(rec[f"pattern{l}"])[:, :, 36:42, :]
        so = (A * own).sum(-1).mean((0, 2)); out[f"own{l}_mean"] = float(so.mean()); out[f"own{l}_maxhead"] = float(so.max())
        if l == 0: out["src0_mean"] = float(A[..., 0:36].sum(-1).mean())
    return out


def cls_struct(r): return "EARLY" if (r["mix_mass"] < 0.5 and r["lik_RO1"] >= 0.4) else ("MIXTURE" if (r["mix_mass"] >= 0.8 and r["lik_RO1"] <= 0.15) else "HYBRID")
def cls_behav(r): return "EARLY" if r["rho"] <= 0.60 else ("MIXTURE" if r["rho"] >= 0.66 else "HYBRID")
def cls_causal(r): return "EARLY" if (r["nec_E"] >= 0.30 and r["nec_M"] <= 0.10) else ("MIXTURE" if (r["nec_M"] >= 0.30 and r["nec_E"] <= 0.15) else "HYBRID")
def classify(r):
    c = (cls_struct(r), cls_behav(r), cls_causal(r)); return dict(cls_struct=c[0], cls_behav=c[1], cls_causal=c[2], cls=c[0] if len(set(c)) == 1 else "HYBRID",
                                                                  competent=bool(r["KL"] <= 0.06))


def measure_final(M, stream):
    S = MT.sets(stream); r = dict(lik_RO1=MT.avail(M, S["PL"], 1))
    mass, share = RP.head_stats(M, S["Xn"][:1500], S["qn"][:1500], F, jnp); r["mix_mass"] = float(mass.max())
    b, Ln = MT.behaviour(M, S); r.update(b); r.update(MT.rho_fit(Ln, S)); r.update(necessity(M, nat(stream))); r.update(classify(r)); return r
