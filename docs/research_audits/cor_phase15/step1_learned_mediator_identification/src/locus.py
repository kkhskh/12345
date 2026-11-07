"""Descriptive (PROTOCOL §6): where is the cross-source interaction in the identified mediator created?

For each model whose primary site was judged RELATION_COMPUTING_MEDIATOR, recompute the S_rel coordinates on the
(a, b) grid while freezing one nonlinear source of cross-position coupling at a time, and report the remaining
interaction share (1 - additive R^2) of the S coordinates:
  none          : unmodified
  pattern{l}    : layer-l attention pattern replaced by the pattern at a fixed reference input (a = b = 0.1)
  all patterns  : every layer's pattern frozen
In TF1, with the layer-0 pattern frozen, resid_mid0 at R = E_R + sum_j alpha_j V LN(x_j) with fixed alpha_j, and
each LN(x_j) depends on one source only, so the interaction share must be exactly 0 (structural check).
Binary tasks only.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import jax.numpy as jnp
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ckpt, generators as G, mediator as MD, model as M
from run_protocol import PRIMARY, SEEDS, pairs


def coords(P, task, hook, pos, U, freeze, ref_store, p):
    g = np.linspace(-3, 3, 61); Aa, Bb = np.meshgrid(g, g, indexing="ij")
    d = dict(a=Aa.ravel(), b=Bb.ravel(), p=np.full(Aa.size, p))
    T = M.n_tokens(MD.CODE[task]); idx = [T - 1] if pos == "R" else list(range(T))
    st = {}
    def hk(name, arr):
        if name in freeze:
            arr = jnp.broadcast_to(jnp.asarray(ref_store[name]), arr.shape)
        if name == hook:
            st["x"] = np.asarray(arr)[:, idx, :].reshape(arr.shape[0], -1)
        return arr
    M.forward(P, MD.CODE[task], MD.to_inp(task, d), hk)
    Z = (st["x"] @ U).reshape(61, 61, -1)
    Zs = Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean((0, 1), keepdims=True)
    return float(((Z - Zs) ** 2).sum() / max(((Z - Z.mean((0, 1))) ** 2).sum(), 1e-30))


def main(tags):
    res = {}
    for tag in tags:
        data = json.load(open(ROOT / "logs" / f"protocol_{tag}.json"))
        ckdir = ROOT / ("checkpoints/historical" if tag.startswith("E1") else "checkpoints/new")
        for key, sites in data.items():
            task, arch, s, dist = key.split("/")
            if task.startswith("K3") or dist != "train":
                continue
            seed = int(s[1:])
            P = ckpt.load(ckdir / f"{MD.CODE[task]}_{arch}_s{seed}.npz")
            for site_name, r in sites.items():
                if r.get("verdict") != "RELATION_COMPUTING_MEDIATOR":
                    continue
                hook, pos = site_name.split("@")
                site = MD.Site(P, task, hook, pos)
                A, C = pairs(task, 800, SEEDS["pairs_fit"])
                U, k, _ = MD.s_rel(site, A, C)
                ref_store = {}
                def cap(name, arr):
                    if name.startswith("pattern"):
                        ref_store[name] = np.asarray(arr)
                    return arr
                M.forward(P, MD.CODE[task], MD.to_inp(task, dict(a=np.array([0.1]), b=np.array([0.1]), p=np.array([1.0]))), cap)
                L = len(P["layers"])
                out = {}
                for p in ([-1.0, 1.0] if task == "REL-GATE" else [0.0]):
                    for nm, fr in [("none", [])] + [(f"pattern{l}", [f"pattern{l}"]) for l in range(L)] + \
                                  ([("all_patterns", [f"pattern{l}" for l in range(L)])] if L > 1 else []):
                        out[f"p={p:+.0f}/{nm}"] = coords(P, task, hook, pos, U, fr, ref_store, p)
                res[f"{key}/{site_name}"] = out
                print(key, site_name, {k_: round(v, 5) for k_, v in out.items()}, flush=True)
    (ROOT / "logs" / f"locus_{'_'.join(tags)}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
