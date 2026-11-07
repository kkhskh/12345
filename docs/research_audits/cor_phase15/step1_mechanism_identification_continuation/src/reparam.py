"""Reparameterization attacks on learned checkpoints (§10): which identification outputs survive?

Gauges (all exactly function-preserving for this architecture):
  QK  : WQ_h -> WQ_h M,  WK_h -> WK_h M^{-T}          (random well-conditioned M, per head)
  VO  : WV_h -> WV_h N,  WO_h -> N^{-1} WO_h
  PERM: MLP hidden-unit permutation
  SPLIT: duplicate unit u, halve its output row on both copies (width 32 -> 33)
  HEAD: head permutation
Not applied: residual-basis rotation (learned LN biases feed bias-free Q/K/V maps, so a rotation
cannot be conjugated through LN without adding bias parameters); positive rescaling (not
function-preserving for GELU).
"""
from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learned as Lm  # noqa: E402
import learned_analysis as LA  # noqa: E402


def gauge(P, kind, seed=0):
    P = copy.deepcopy(P)
    r = np.random.default_rng(seed)
    for lay in P["layers"]:
        H, d, dh = lay["WQ"].shape
        if kind == "QK":
            for h in range(H):
                M = np.eye(dh) + 0.5 * r.normal(size=(dh, dh))
                lay["WQ"][h] = lay["WQ"][h] @ M
                lay["WK"][h] = lay["WK"][h] @ np.linalg.inv(M).T
        elif kind == "VO":
            for h in range(H):
                N = np.eye(dh) + 0.5 * r.normal(size=(dh, dh))
                lay["WV"][h] = lay["WV"][h] @ N
                lay["WO"][h] = np.linalg.inv(N) @ lay["WO"][h]
        elif kind == "HEAD":
            perm = r.permutation(H)
            for k in ("WQ", "WK", "WV", "WO"):
                lay[k] = lay[k][perm]
        elif kind == "PERM" and "W1" in lay:
            perm = r.permutation(lay["W1"].shape[1])
            lay["W1"], lay["c1"], lay["W2"] = lay["W1"][:, perm], lay["c1"][perm], lay["W2"][perm]
            lay["_perm"] = perm
        elif kind == "SPLIT" and "W1" in lay:
            u = 0
            lay["W1"] = np.column_stack([lay["W1"], lay["W1"][:, u]])
            lay["c1"] = np.concatenate([lay["c1"], lay["c1"][u:u + 1]])
            W2 = lay["W2"].copy(); W2[u] /= 2
            lay["W2"] = np.vstack([W2, W2[u:u + 1]])
    return P


def main():
    out = {}
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    val = Lm.sample("T1", 4000, np.random.default_rng(5))
    for task, arch, seed in [("T1", "TF1", 0), ("T3", "TF2", 0), ("T4", "TF1", 0)]:
        P0 = LA.load(task, arch, seed)
        val = Lm.sample(task, 4000, np.random.default_rng(5))
        lg0 = np.asarray(Lm.forward(P0, task, Lm.to_inp(task, val)))
        r0 = LA.analyse_binary(task, arch, seed, [1], P=P0)
        for kind in ["QK", "VO", "HEAD", "PERM", "SPLIT"]:
            P1 = gauge(P0, kind, seed=1)
            lg1 = np.asarray(Lm.forward(P1, task, Lm.to_inp(task, val)))
            r1 = LA.analyse_binary(task, arch, seed, [1], P=P1)
            u0 = {l: sorted(x["unit"] for x in v) for l, v in r0["units"].items()}
            u1 = {l: sorted(x["unit"] for x in v) for l, v in r1["units"].items()}
            Iq0 = {l: sorted(round(x["I_agree"], 6) for x in v) for l, v in r0["units"].items()}
            Iq1 = {l: sorted(round(x["I_agree"], 6) for x in v) for l, v in r1["units"].items()}
            pr = max(abs(r0["probes"][s][k] - r1["probes"][s][k]) for s in r0["probes"] for k in r0["probes"][s])
            it = max(abs(r0["interchange"][k]["decision_to_source"] - r1["interchange"][k]["decision_to_source"]) for k in r0["interchange"])
            g0, g1 = r0["grid"], r1["grid"]
            key = list(g0)[0]
            row = dict(max_logit_diff=float(np.abs(lg0 - lg1).max()),
                       unit_indices_same=u0 == u1, I_agree_multiset_same=Iq0 == Iq1,
                       n_rel=(r0["units_summary"]["n_rel"] if r0["units_summary"] else None,
                              r1["units_summary"]["n_rel"] if r1["units_summary"] else None),
                       ablation_d_agree=(r0["units_summary"]["ablated"]["d_agree"] if r0["units_summary"] else None,
                                         r1["units_summary"]["ablated"]["d_agree"] if r1["units_summary"] else None),
                       probe_r2_maxdiff=float(pr), interchange_maxdiff=float(it),
                       grid_d_agree=(g0[key]["d_agree"], g1[key]["d_agree"]))
            out[f"{task}/{arch}/{kind}"] = row
            log(f"{task} {arch} {kind}: {json.dumps(row, default=str)}")
    root = HERE.parent
    (root / "logs" / "reparam.txt").write_text("\n".join(lines) + "\n")
    (root / "logs" / "reparam.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
