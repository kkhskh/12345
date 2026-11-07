"""Mechanism identification for the learned models (PREREGISTRATION_LEARNED_MODELS.md §6-§7, deviations D1-D3).

Usage: python3 learned_analysis.py <tag> <seeds>     e.g.  dev 0,1,2   |  heldout 3,4
No probe, direction, unit or site is selected by the evaluated margin.
"""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings('ignore')
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learned as Lm  # noqa: E402
from semantics import sample_binary, sample_k, semantic_agreement  # noqa: E402

ROOT = HERE.parent
CK = ROOT / "checkpoints"
BIN = ["T1", "T2", "T3", "T4"]


def load(task, arch, seed):
    with open(CK / f"{task}_{arch}_s{seed}.pkl", "rb") as fh:
        return pickle.load(fh)


def inp_bin(a, b, p):
    return dict(a=jnp.asarray(a, jnp.float32), b=jnp.asarray(b, jnp.float32), p=jnp.asarray(p, jnp.float32))


def fwd_margin(P, task, inp, hook=None):
    return np.asarray(Lm.margin(Lm.forward(P, task, inp, hook)))


def capture(P, task, inp):
    store = {}
    def hk(name, arr):
        store[name] = np.asarray(arr); return arr
    lg = np.asarray(Lm.forward(P, task, inp, hk))
    return lg, store


def grid(n=61, lim=3.0):
    g = np.linspace(-lim, lim, n)
    A, B = np.meshgrid(g, g, indexing="ij")
    return A.ravel(), B.ravel(), g


def dsum(vals, s, nb=121, lo=-6, hi=6):
    idx = np.clip(((s - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
    means = np.zeros(nb); cnt = np.bincount(idx, minlength=nb)
    np.add.at(means, idx, vals); means = means / np.maximum(cnt, 1)
    fit = means[idx]
    v = vals.var()
    return float(((vals - fit) ** 2).mean() / v) if v > 1e-12 else 0.0, means, idx


def matched_pairs():
    ag, cf = [], []
    for S in [-2, -1, -0.5, 0.5, 1, 2]:
        for D in np.linspace(0.02 * abs(S), 0.98 * abs(S), 50):
            ag.append(((S + D) / 2, (S - D) / 2))
        for D in np.linspace(abs(S) + 0.5, abs(S) + 3, 50):
            cf.append(((S + D) / 2, (S - D) / 2))
    return np.array(ag), np.array(cf)


def additive_fit(Z):
    """Z: (n, n) values on the product grid. Best additive fit = two-way ANOVA main effects."""
    return Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean()


def r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return 1 - res.var() / y.var() if y.var() > 1e-12 else 0.0


def analyse_binary(task, arch, seed, pvals, P=None):
    P = load(task, arch, seed) if P is None else P
    L = len(P["layers"]); has_mlp = "W1" in P["layers"][0]
    A, B, g = grid()
    out = {}
    agp, cfp = matched_pairs()
    for pv in pvals:
        key = f"p={pv:+d}" if task == "T3" else "all"
        pp = np.full_like(A, float(pv))
        m = fwd_margin(P, task, inp_bin(A, B, pp))
        D_sum, _, _ = dsum(m, A + B)
        ma = fwd_margin(P, task, inp_bin(agp[:, 0], agp[:, 1], np.full(len(agp), float(pv))))
        mc = fwd_margin(P, task, inp_bin(cfp[:, 0], cfp[:, 1], np.full(len(cfp), float(pv))))
        dAg = float(np.mean(ma - mc)); flip = float(np.mean(np.sign(ma) != np.sign(mc)))
        # signed by S so that 'agreement raises support for the sum's candidate' is positive
        Ssign = np.sign(agp[:, 0] + agp[:, 1])
        dAg_signed = float(np.mean(Ssign * (ma - mc)))
        out[key] = dict(D_sum=D_sum, d_agree=dAg, d_agree_signed_by_S=dAg_signed, flip=flip)

    # --- probes (generator seed 3000, training mixture) ---
    rng = np.random.default_rng(3000)
    bt = sample_binary(task, 4000, rng)
    _, st = capture(P, task, inp_bin(bt.a, bt.b, bt.p))
    tr, te = slice(0, 2000), slice(2000, 4000)
    targets = {"a": bt.a, "b": bt.b, "a+b": bt.a + bt.b, "ab": bt.a * bt.b}
    s_lab = (bt.a * bt.b > 0).astype(int)
    probes = {}
    agree_dirs = {}
    sites = [f"resid_mid{l}" for l in range(L)] + [f"resid_post{l}" for l in range(L)] + \
            ([f"mlp_hidden{l}" for l in range(L)] if has_mlp else [])
    for site in sites:
        X = st[site][:, -1]
        res = {}
        for nm, y in targets.items():
            rg = Ridge(alpha=1e-3).fit(X[tr], y[tr]); res[nm] = float(rg.score(X[te], y[te]))
        lr = LogisticRegression(C=10.0, max_iter=3000).fit(X[tr], s_lab[tr])
        res["sign(ab) acc"] = float(lr.score(X[te], s_lab[te]))
        probes[site] = res
        if site.startswith("resid"):
            w = lr.coef_[0]; agree_dirs[site] = w / np.linalg.norm(w)
    # --- interchange along the agreement-probe direction (and a random control) ---
    rnd = np.random.default_rng(999).normal(size=Lm.D_MODEL); rnd /= np.linalg.norm(rnd)
    pv0 = float(pvals[0])
    inter = {}
    base_in = inp_bin(agp[:, 0], agp[:, 1], np.full(len(agp), pv0))
    src_in = inp_bin(cfp[:, 0], cfp[:, 1], np.full(len(cfp), pv0))
    _, st_src = capture(P, task, src_in)
    mb = fwd_margin(P, task, base_in); ms_ = fwd_margin(P, task, src_in)
    for site, w in agree_dirs.items():
        for nm, u in (("probe", w), ("random", rnd)):
            src_comp = st_src[site][:, -1] @ u
            def hk(name, arr, site=site, u=u, src_comp=src_comp):
                if name == site:
                    a_ = np.asarray(arr).copy()
                    a_[:, -1] = a_[:, -1] + np.outer(src_comp - a_[:, -1] @ u, u)
                    return jnp.asarray(a_)
                return arr
            mp = fwd_margin(P, task, base_in, hk)
            inter[f"{site}/{nm}"] = dict(
                frac_of_gap=float(np.median((mp - mb) / np.where(np.abs(ms_ - mb) > 1e-6, ms_ - mb, np.nan)[~np.isnan(ms_ - mb)].mean() if False else np.nanmedian((mp - mb) / np.where(np.abs(ms_ - mb) > 1e-6, ms_ - mb, np.nan)))),
                decision_to_source=float(np.mean(np.sign(mp) == np.sign(ms_))),
                decision_base_vs_source_differ=float(np.mean(np.sign(mb) != np.sign(ms_))))
        # full-site patch at R (upper bound)
        def hkf(name, arr, site=site):
            if name == site:
                a_ = np.asarray(arr).copy(); a_[:, -1] = st_src[site][:, -1]; return jnp.asarray(a_)
            return arr
        mp = fwd_margin(P, task, base_in, hkf)
        inter[f"{site}/full"] = dict(decision_to_source=float(np.mean(np.sign(mp) == np.sign(ms_))))
    # --- units ---
    units = {}
    if has_mlp:
        _, stg = capture(P, task, inp_bin(A, B, np.full_like(A, pv0)))
        n = len(g)
        ab = A * B; sg = np.sign(ab)
        _, sta = capture(P, task, base_in); _, stc = capture(P, task, src_in)
        gen_mean = {l: st[f"mlp_hidden{l}"][:, -1].mean(0) for l in range(L)}
        dirv = np.asarray(P["gf"] * (P["WU"][:, 0] - P["WU"][:, 1]))
        for l in range(L):
            H = stg[f"mlp_hidden{l}"][:, -1]
            W2 = np.asarray(P["layers"][l]["W2"])
            rel = []
            for u in range(H.shape[1]):
                y = H[:, u]
                if y.var() < 1e-10:
                    continue
                ds, means, idx = dsum(y, A + B)
                addf = additive_fit(y.reshape(n, n)).ravel()
                sumf = means[idx]
                base_r2 = r2(np.column_stack([addf, sumf]), y)
                full_r2 = r2(np.column_stack([addf, sumf, ab, sg]), y)
                Iag = float(full_r2 - base_r2)
                if Iag >= 0.2 and ds >= 0.2:
                    direct = float(W2[u] @ dirv)
                    cond = float(np.mean(sta[f"mlp_hidden{l}"][:, -1, u] - stc[f"mlp_hidden{l}"][:, -1, u]))
                    rel.append(dict(unit=u, I_agree=Iag, D_sum=ds, direct=direct, cond_diff=cond,
                                    predicted_sign=float(np.sign(direct * cond))))
            units[l] = rel
        # ablation of all relation-sensitive units (mean over generator), per unit sign test, surrogate
        def ablate(sel):
            def hk(name, arr):
                for l, us in sel.items():
                    if name == f"mlp_hidden{l}" and us:
                        a_ = np.asarray(arr).copy(); a_[:, -1, us] = gen_mean[l][us]; return jnp.asarray(a_)
                return arr
            return hk
        sel = {l: [r_["unit"] for r_ in units[l]] for l in range(L)}
        val = Lm.sample(task, 20000, np.random.default_rng(1000))
        def metrics(hk):
            mm = fwd_margin(P, task, inp_bin(A, B, np.full_like(A, pv0)), hk)
            ma = fwd_margin(P, task, base_in, hk); mc = fwd_margin(P, task, src_in, hk)
            lg = np.asarray(Lm.forward(P, task, Lm.to_inp(task, val), hk))
            acc = float((np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0) == val.y).mean())
            return dict(D_sum=dsum(mm, A + B)[0], d_agree=float(np.mean(ma - mc)),
                        flip=float(np.mean(np.sign(ma) != np.sign(mc))), acc=acc)
        units_summary = dict(n_rel={l: len(units[l]) for l in range(L)}, full=metrics(None), ablated=metrics(ablate(sel)))
        # per-unit sign prediction
        hits = tot = 0
        full_d = units_summary["full"]["d_agree"]
        for l in range(L):
            for r_ in units[l]:
                hk = ablate({l: [r_["unit"]]})
                ma = fwd_margin(P, task, base_in, hk); mc = fwd_margin(P, task, src_in, hk)
                contrib = full_d - float(np.mean(ma - mc))
                r_["actual_contrib_sign"] = float(np.sign(contrib))
                tot += 1; hits += r_["predicted_sign"] == np.sign(contrib)
        units_summary["sign_prediction"] = dict(hits=int(hits), n=int(tot))
        # relation-blind surrogate: unit activation replaced by its bin-mean in a+b (fitted on the grid)
        fits = {}
        for l in range(L):
            H = stg[f"mlp_hidden{l}"][:, -1]
            fits[l] = {u: dsum(H[:, u], A + B)[1] for u in sel[l]}
        def surrogate_hook(svals):
            idx = np.clip(((svals + 6) / 12 * 121).astype(int), 0, 120)
            def hk(name, arr):
                for l, us in sel.items():
                    if name == f"mlp_hidden{l}" and us:
                        a_ = np.asarray(arr).copy()
                        for u in us:
                            a_[:, -1, u] = fits[l][u][idx]
                        return jnp.asarray(a_)
                return arr
            return hk
        mm = fwd_margin(P, task, inp_bin(A, B, np.full_like(A, pv0)), surrogate_hook(A + B))
        ma = fwd_margin(P, task, base_in, surrogate_hook(agp.sum(1))); mc = fwd_margin(P, task, src_in, surrogate_hook(cfp.sum(1)))
        lg = np.asarray(Lm.forward(P, task, Lm.to_inp(task, val), surrogate_hook(val.a + val.b)))
        units_summary["surrogate"] = dict(D_sum=dsum(mm, A + B)[0], d_agree=float(np.mean(ma - mc)),
                                          flip=float(np.mean(np.sign(ma) != np.sign(mc))),
                                          acc=float((np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0) == val.y).mean()))
    else:
        units_summary = None
    return dict(grid=out, probes=probes, interchange=inter, units={str(k): v for k, v in units.items()},
                units_summary=units_summary)


def analyse_k(task, arch, seed):
    P = load(task, arch, seed)
    L = len(P["layers"]); has_mlp = "W1" in P["layers"][0]
    rng = np.random.default_rng(3000)
    bt = sample_k(task, 4000, rng)
    inp = Lm.to_inp(task, bt)
    lg, st = capture(P, task, inp)
    # sum-invariance deficit
    base = sample_k(task, 2000, np.random.default_rng(3001))
    r = np.random.default_rng(3002)
    var_in = []
    lg0 = np.asarray(Lm.forward(P, task, Lm.to_inp(task, base)))
    rel_change = []
    allv = []
    for _ in range(16):
        dlt = r.normal(0, 1, base.zA.shape)
        b2 = dict(zA=jnp.asarray(base.zA + dlt, jnp.float32), zB=jnp.asarray(base.zB - dlt, jnp.float32), p=jnp.asarray(base.p))
        l2 = np.asarray(Lm.forward(P, task, b2))
        allv.append(l2 - l2.mean(1, keepdims=True))
        ag0 = semantic_agreement(base.zA, base.zB); ag1 = semantic_agreement(base.zA + dlt, base.zB - dlt)
        rel_change.append((ag0 != ag1, lg0.argmax(1) != l2.argmax(1)))
    allv = np.stack(allv)            # (16, N, K)
    within = allv.var(0).sum(-1).mean()
    total = (lg0 - lg0.mean(1, keepdims=True)).var(0).sum()
    D = float(within / total)
    ch = np.concatenate([c[0] for c in rel_change]); fl = np.concatenate([c[1] for c in rel_change])
    res = dict(Dsum_K=D, P_decision_change_given_agreement_change=float(fl[ch].mean()),
               P_decision_change_given_agreement_same=float(fl[~ch].mean()))
    # probes
    sem = semantic_agreement(bt.zA, bt.zB)
    dot = (bt.zA * bt.zB).sum(1)
    zc = lambda z: z - z.mean(1, keepdims=True)
    cdot = (zc(bt.zA) * zc(bt.zB)).sum(1)
    tr, te = slice(0, 2000), slice(2000, 4000)
    probes = {}
    for site in [f"resid_mid{l}" for l in range(L)] + [f"resid_post{l}" for l in range(L)] + \
            ([f"mlp_hidden{l}" for l in range(L)] if has_mlp else []):
        X = st[site][:, -1]
        pr = {}
        for nm, y in (("dot", dot), ("centred dot", cdot)):
            pr[nm] = float(Ridge(alpha=1e-3).fit(X[tr], y[tr]).score(X[te], y[te]))
        pr["argmax agree acc"] = float(LogisticRegression(C=10.0, max_iter=3000).fit(X[tr], sem[tr]).score(X[te], sem[te]))
        probes[site] = pr
    # units
    units = {}
    if has_mlp:
        s = bt.zA + bt.zB
        def poly2(Z):
            n, k = Z.shape
            cols = [Z] + [Z[:, i:i + 1] * Z[:, j:j + 1] for i in range(k) for j in range(i, k)]
            return np.column_stack(cols)
        Fs = poly2(s); Fab = np.column_stack([poly2(bt.zA), poly2(bt.zB)])
        for l in range(L):
            H = st[f"mlp_hidden{l}"][:, -1]
            rel = []
            for u in range(H.shape[1]):
                y = H[:, u]
                if y.var() < 1e-10:
                    continue
                ds = 1 - r2(Fs, y)
                gain = r2(np.column_stack([Fs, Fab, dot, sem]), y) - r2(np.column_stack([Fs, Fab]), y)
                if gain >= 0.2 and ds >= 0.2:
                    rel.append(dict(unit=u, I_agree=float(gain), D_sum=float(ds)))
            units[str(l)] = rel
    val = sample_k(task, 20000, np.random.default_rng(2000))
    lgv = np.asarray(Lm.forward(P, task, Lm.to_inp(task, val)))
    res["acc"] = float((lgv.argmax(1) == val.y).mean())
    return dict(summary=res, probes=probes, units=units)


def main(tag, seeds):
    out = {}
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    for task in Lm.TASKS:
        for arch in Lm.ARCH:
            for seed in seeds:
                key = f"{task}/{arch}/s{seed}"
                if task in BIN:
                    r = analyse_binary(task, arch, seed, [1, -1] if task == "T3" else [0])
                    log(f"{key}: grid {json.dumps(r['grid'])}")
                    us = r["units_summary"]
                    if us:
                        log(f"   relation units {us['n_rel']}; full {us['full']}; ablated {us['ablated']}; "
                            f"surrogate {us['surrogate']}; sign prediction {us['sign_prediction']}")
                    best = {k: v for k, v in r["interchange"].items() if k.endswith("/probe") or k.endswith("/random")}
                    log("   interchange " + json.dumps({k: round(v["decision_to_source"], 3) for k, v in best.items()}))
                else:
                    r = analyse_k(task, arch, seed)
                    log(f"{key}: {json.dumps(r['summary'])}; relation units "
                        f"{ {k: len(v) for k, v in r['units'].items()} }")
                out[key] = r
    (ROOT / "logs" / f"learned_analysis_{tag}.txt").write_text("\n".join(lines) + "\n")
    (ROOT / "logs" / f"learned_analysis_{tag}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2].split(",")])
