"""Frozen mediator-identification and causal-test protocol (PROTOCOL.md). Usage:
    python3 src/run_protocol.py <tag> <task,...> <arch,...> <seed,...> <ckpt_dir> [dist]
Every number reported for a model comes from this script. Nothing here reads the evaluated margin to select a site,
subspace or dimension (S_grad and S_das are comparison candidates and are labelled as output- / target-using).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ckpt  # noqa: E402
import generators as G  # noqa: E402
import mediator as MD  # noqa: E402
import model as M  # noqa: E402

PRIMARY = {"TF1": [("resid_mid0", "R")], "TF2": [("resid_mid1", "R"), ("resid_post0", "ALL")],
           "TF2AO": [("resid_post0", "ALL")]}
SEEDS = dict(pairs_fit=5001, pairs_eval=6002, bank=6003, ev=6004)
CRIT = dict(cut=0.99, iia=0.95, rand=0.2, nec=0.2, suff=0.95)


def pairs(task, n, seed):
    rng = np.random.default_rng(seed)
    return G.matched_pairs_k(task, n, rng) if task.startswith("K3") else G.matched_pairs(task, n, rng)


def sample(task, n, seed, dist):
    rng = np.random.default_rng(seed)
    return G.sample_k(task, n, rng, dist) if task.startswith("K3") else G.sample(task, n, rng, dist)


def analyse(task, arch, seed, ckdir, dist):
    P = ckpt.load(Path(ckdir) / f"{MD.CODE[task]}_{arch}_s{seed}.npz")
    A, C = pairs(task, 800, SEEDS["pairs_fit"])
    Ae, Ce = pairs(task, 500, SEEDS["pairs_eval"] + (100 if dist == "gauss" else 0))
    bank = sample(task, 1500, SEEDS["bank"], "train")
    ev = sample(task, 3000, SEEDS["ev"], dist)
    out = {}
    for hook, pos in PRIMARY[arch]:
        site = MD.Site(P, task, hook, pos)
        r = {}
        # behaviour on matched pairs and input-level single-source changes
        dA = MD.decision(task, site.get(Ae)[0]); dC = MD.decision(task, site.get(Ce)[0])
        diff = Ae["y"] != Ce["y"]
        r["n_pairs_target_differs"] = int(diff.sum())
        r["matched_pair_decision_diff_rate"] = float((dA != dC).mean())
        r["acc_ev"] = float((MD.decision(task, site.get(ev)[0]) == ev["y"]).mean())
        _, hA = site.get(A); _, hC = site.get(C); _, hb = site.get(bank)
        r["contrast_energy_ratio"] = float(np.mean(np.sum((hA - hC) ** 2, 1)) / np.mean(np.sum((hb - hb.mean(0)) ** 2, 1)))
        if not task.startswith("K3"):
            base, fa, fb = G.single_source_changes(task, 1000, np.random.default_rng(6005))
            d0 = MD.decision(task, site.get(base)[0])
            r["input_level"] = {nm: dict(model_change=float((MD.decision(task, site.get(x)[0]) != d0).mean()),
                                        target_change=float((x["y"] != base["y"]).mean())) for nm, x in (("flip_a_hold_b", fa), ("flip_b_hold_a", fb))}
        mdiff = dA != dC
        r["n_pairs_model_differs"] = int(mdiff.sum())
        if mdiff.sum() < 20:
            r["note"] = "model decisions (almost) never differ on matched pairs: no relation-sensitive behaviour to mediate; contrast energy reported"
            U, k, ev_ = MD.s_rel(site, A, C)
            r.update(k=k, sv_energy=ev_.tolist())
            out[f"{hook}@{pos}"] = r
            continue
        r["iia_full_site"] = MD.iia(site, Ae, Ce, None)[0]
        U, k, ev_ = MD.s_rel(site, A, C)
        r.update(k=k, sv_energy=[float(x) for x in ev_])
        cands = {"S_rel": U, "S_pca": MD.s_pca(site, bank, k), "S_grad": MD.s_grad(site, bank, k),
                 "S_das": MD.s_das(site, A, C, k), "S_alt": MD.s_alt(site, bank, U, k)}
        r["iia"] = {nm: (MD.iia(site, Ae, Ce, V)[0] if V is not None else None) for nm, V in cands.items()}
        rr = [MD.iia(site, Ae, Ce, MD.s_rand(U.shape[0], k, s))[0] for s in range(20)]
        r["iia"]["S_rand_mean"], r["iia"]["S_rand_max"] = float(np.mean(rr)), float(np.max(rr))
        r["cos_Srel_Sgrad"] = float(np.linalg.svd(U.T @ cands["S_grad"], compute_uv=False).max())
        r["cos_Srel_Sdas"] = float(np.linalg.svd(U.T @ cands["S_das"], compute_uv=False).max())
        r["necessity"] = MD.pair_mean_ablation(site, Ae, Ce, U)
        r["sufficiency"] = MD.complement_interchange(site, Ae, Ce, U)
        r["manifold"] = MD.manifold_distance(site, bank, Ae, Ce, U)
        if hook.startswith("resid_mid") and pos == "R" and "W1" in P["layers"][int(hook[-1])]:
            r["path_split"] = MD.path_split(site, Ae, Ce, U, int(hook[-1]))
        # content: is the mediator a cross-source interaction or a source-additive code?
        if task.startswith("K3"):
            kc = MD.KCoords(site, U, bank)
            r["additive_R2"] = kc.additive_r2()
            r["surrogates"] = {kind: MD.surrogate_test(site, U, kc, kind, ev, Ae, Ce) for kind in ("additive", "interaction")}
        else:
            gc = MD.GridCoords(site, U)
            r["additive_R2"] = gc.additive_r2()
            r["surrogates"] = {kind: MD.surrogate_test(site, U, gc, kind, ev, Ae, Ce) for kind in ("exact", "additive", "linear", "aggregate")}
            # descriptive: how well is the interaction part explained by sign(ab)?
            g = gc.g; Aa, Bb = np.meshgrid(g, g, indexing="ij"); sab = np.sign(Aa * Bb).ravel()
            r2s = []
            for Z in gc.C.values():
                Rz = Z - (Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean((0, 1), keepdims=True))
                for j in range(Z.shape[-1]):
                    x = Rz[..., j].ravel()
                    if x.var() > 1e-12 * max(Z[..., j].var(), 1e-30):
                        r2s.append(1 - np.var(x - np.polyval(np.polyfit(sab, x, 1), sab)) / np.var(x))
            r["interaction_R2_by_sign_ab"] = [float(v) for v in r2s]
        # verdict (frozen criteria)
        causal = (r["iia_full_site"] >= CRIT["cut"] and r["iia"]["S_rel"] >= CRIT["iia"] and r["iia"]["S_rand_mean"] <= CRIT["rand"]
                  and r["necessity"]["flip_after"] <= CRIT["nec"] * max(r["necessity"]["flip_before"], 1e-9)
                  and r["sufficiency"] >= CRIT["suff"])
        add = r["surrogates"]["additive"]
        ref = r["surrogates"]["interaction" if task.startswith("K3") else "exact"]   # same fitting family, interaction allowed
        interactive = (ref["pair_flip"] - add["pair_flip"]) >= 0.10 or (ref["acc"] - add["acc"]) >= 0.05
        r["verdict"] = ("RELATION_COMPUTING_MEDIATOR" if causal and interactive else
                        "SOURCE_ADDITIVE_MEDIATOR" if causal else
                        "NO_CUT" if r["iia_full_site"] < CRIT["cut"] else "MEDIATOR_NOT_IDENTIFIED")
        out[f"{hook}@{pos}"] = r
    return out


def main():
    tag, tasks, archs, seeds, ckdir = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), \
        [int(s) for s in sys.argv[4].split(",")], ROOT / sys.argv[5]
    dist = sys.argv[6] if len(sys.argv) > 6 else "train"
    res, lines = {}, []
    T0 = time.time()
    for task in tasks:
        for arch in archs:
            for seed in seeds:
                f = ckdir / f"{MD.CODE[task]}_{arch}_s{seed}.npz"
                if not f.exists():
                    continue
                key = f"{task}/{arch}/s{seed}/{dist}"
                res[key] = analyse(task, arch, seed, ckdir, dist)
                for site, r in res[key].items():
                    s = (f"{key} {site}: verdict={r.get('verdict', '-')} k={r.get('k')} full={r.get('iia_full_site')} "
                         f"IIA={ {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r.get('iia', {}).items()} } "
                         f"nec={r.get('necessity')} suff={r.get('sufficiency')} addR2={r.get('additive_R2')} "
                         f"surr={r.get('surrogates')} pairdiff={r.get('matched_pair_decision_diff_rate')} acc={r.get('acc_ev')}")
                    print(s, flush=True); lines.append(s)
    lines.append(f"wall {time.time() - T0:.0f}s")
    (ROOT / "logs" / f"protocol_{tag}.txt").write_text("\n".join(lines) + "\n")
    (ROOT / "logs" / f"protocol_{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
