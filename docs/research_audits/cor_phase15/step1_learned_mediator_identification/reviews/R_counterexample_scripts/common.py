"""Shared helpers for the adversarial counterexample review (reviews/R_counterexample.md)."""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import ckpt  # noqa: E402
import generators as G  # noqa: E402
import mediator as MD  # noqa: E402
import model as M  # noqa: E402

XNOR_TF2 = [("historical", 2), ("historical", 3), ("historical", 4), ("new", 7), ("new", 8), ("new", 9)]


def load(code, arch, seed, where=None):
    if where is None:
        where = "new" if (seed >= 7 or code == "T6") else "historical"
    return ckpt.load(ROOT / "checkpoints" / where / f"{code}_{arch}_s{seed}.npz")


def pack(task, a, b, p=None):
    p = np.zeros(len(a)) if p is None else p
    y, soft = G.target(task, a, b, p)
    return dict(a=a, b=b, p=p, y=y, soft=soft, z=a + b, s=(a * b > 0).astype(int))


def custom_pairs(task, n, rng, zlo, zhi, da_frac=(0.0, 0.98), dc_extra=None, lim=3.0, p=None):
    """Matched pairs with |z| in [zlo, zhi]. Agreement member |d| = u*|z|, u ~ U[da_frac];
    conflict member |d| = |z| + e, e ~ U[dc_extra] (default: whole feasible range up to 2*lim - |z|)."""
    z = rng.uniform(zlo, zhi, n) * rng.choice([-1, 1], n)
    az = np.abs(z)
    da = rng.uniform(*da_frac, n) * az * rng.choice([-1, 1], n)
    if dc_extra is None:
        e = 0.02 + rng.uniform(0, 1, n) * np.maximum(2 * lim - 2 * az - 0.04, 0)
    else:
        e = rng.uniform(*dc_extra, n)
    dc = (az + e) * rng.choice([-1, 1], n)
    pp = np.zeros(n) if p is None else p
    A = pack(task, (z + da) / 2, (z - da) / 2, pp)
    C = pack(task, (z + dc) / 2, (z - dc) / 2, pp)
    ok = (np.abs(np.concatenate([[A["a"], A["b"], C["a"], C["b"]]]).T).max(1) <= lim) & (A["s"] == 1) & (C["s"] == 0)
    sel = lambda X: {k: v[ok] for k, v in X.items()}
    return sel(A), sel(C)


def causal(site, A, C, U):
    """IIA, necessity (flip_after / flip_before), sufficiency, and n differing pairs."""
    dA, dC = MD.model_dec(site, A), MD.model_dec(site, C)
    n = int((dA != dC).sum())
    if n == 0:
        return dict(n_diff=0)
    iia, _ = MD.iia(site, A, C, U)
    nec = MD.pair_mean_ablation(site, A, C, U)
    suf = MD.complement_interchange(site, A, C, U)
    return dict(n_diff=n, n=len(A["a"]), iia=round(iia, 3), nec=round(nec["flip_after"], 3), suf=round(suf, 3))


def readout_dir(P):
    """Effective readout direction in residual space at R (pre-final-LN), ignoring the final MLP: centred gf*(WU0-WU1)."""
    w = np.asarray(P["gf"]) * (np.asarray(P["WU"])[:, 0] - np.asarray(P["WU"])[:, 1])
    w = w - w.mean()
    return (w / np.linalg.norm(w))[:, None]
