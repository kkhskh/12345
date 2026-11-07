import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ckpt, generators as G, mediator as MD, model as M


def test_matched_pairs_have_identical_aggregate():
    A, C = G.matched_pairs("REL-GATE", 500, np.random.default_rng(0))
    assert np.allclose(A["z"], C["z"]) and (A["p"] == C["p"]).all() and (A["s"] == 1).all() and (C["s"] == 0).all()


def test_checkpoint_roundtrip_and_forward():
    P = ckpt.load(ROOT / "checkpoints/historical/T1_TF1_s0.npz")
    val = M.sample("T1", 4000, np.random.default_rng(1000))
    assert M.accuracy(P, "T1", val) > 0.99


def test_tf1_pattern_freeze_removes_interaction():
    """T-ATT: with the layer-0 pattern frozen, resid_mid0 at R is exactly source-additive."""
    import jax.numpy as jnp
    P = ckpt.load(ROOT / "checkpoints/historical/T1_TF1_s4.npz")
    g = np.linspace(-3, 3, 21); Aa, Bb = np.meshgrid(g, g, indexing="ij")
    d = dict(a=Aa.ravel(), b=Bb.ravel(), p=np.zeros(Aa.size))
    store = {}
    M.forward(P, "T1", MD.to_inp("REL-XNOR", dict(a=np.array([.1]), b=np.array([.1]), p=np.zeros(1))),
              lambda n, a: (store.__setitem__(n, np.asarray(a)) or a) if n == "pattern0" else a)
    st = {}
    def hk(n, a):
        if n == "pattern0":
            return jnp.broadcast_to(jnp.asarray(store["pattern0"]), a.shape)
        if n == "resid_mid0":
            st["x"] = np.asarray(a)[:, -1]
        return a
    M.forward(P, "T1", MD.to_inp("REL-XNOR", d), hk)
    Z = st["x"].reshape(21, 21, -1)
    R = Z - (Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean((0, 1), keepdims=True))
    assert np.abs(R).max() < 1e-4 * np.abs(Z).max()


def test_headline_mediator_xnor_tf2_s2():
    P = ckpt.load(ROOT / "checkpoints/historical/T1_TF2_s2.npz")
    site = MD.Site(P, "REL-XNOR", "resid_mid1", "R")
    A, C = G.matched_pairs("REL-XNOR", 400, np.random.default_rng(11))
    Ae, Ce = G.matched_pairs("REL-XNOR", 300, np.random.default_rng(12))
    U, k, _ = MD.s_rel(site, A, C)
    assert MD.iia(site, Ae, Ce, U)[0] > 0.95
    assert MD.iia(site, Ae, Ce, MD.s_rand(U.shape[0], k, 0))[0] < 0.3


def test_aggregate_only_ceiling():
    assert abs(G.aggregate_only_ceiling("REL-XNOR", n=400000) - 0.754) < 0.01
