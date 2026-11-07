"""R2: mean-ablation vs resample-ablation vs random-unit ablation for the preregistered relation units."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, jax.numpy as jnp
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG / "src"))
import learned as Lm, learned_analysis as LA
def go(task, arch, seed, pv):
    r = LA.analyse_binary(task, arch, seed, [pv]); P = LA.load(task, arch, seed); L = len(P["layers"])
    sel = {l: [u["unit"] for u in r["units"][str(l)]] for l in range(L)}
    val = Lm.sample(task, 20000, np.random.default_rng(1000)); inp = Lm.to_inp(task, val)
    _, st = LA.capture(P, task, inp)
    perm = np.random.default_rng(7).permutation(len(val.y))
    def acc(hk):
        lg = np.asarray(Lm.forward(P, task, inp, hk)); return float((np.where(lg[:, 0] > lg[:, 1], 1., -1.) == val.y).mean())
    def mk(mode, units):
        def hk(name, arr):
            for l, us in units.items():
                if name == f"mlp_hidden{l}" and us:
                    a_ = np.asarray(arr).copy(); H = st[f"mlp_hidden{l}"][:, -1]
                    a_[:, -1, us] = H[:, us].mean(0) if mode == "mean" else H[perm][:, us]
                    return jnp.asarray(a_)
            return arr
        return hk
    rng = np.random.default_rng(11)
    W = P["layers"][0]["W1"].shape[1]
    rnd = {l: list(rng.choice([u for u in range(W) if u not in sel[l]], min(len(sel[l]), W - len(sel[l])), replace=False)) for l in range(L)}
    # post-ablation hidden-state distance to data: fraction of ablated last-position MLP-hidden vectors whose nearest real neighbour distance exceeds 99th pct of real NN distances
    print(task, arch, seed, "n_units", {l: len(v) for l, v in sel.items()}, "full", acc(None),
          "mean-abl", acc(mk("mean", sel)), "resample-abl", acc(mk("resample", sel)),
          "random-units mean-abl", acc(mk("mean", rnd)), "random-units resample", acc(mk("resample", rnd)))
for t, a, s, pv in [("T1", "TF1", 0, 0), ("T1", "TF1", 1, 0), ("T1", "TF2", 0, 0), ("T3", "TF1", 0, 1)]:
    go(t, a, s, pv)
