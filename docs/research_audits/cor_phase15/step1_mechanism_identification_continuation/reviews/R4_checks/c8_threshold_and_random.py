"""R4 check 8: (a) set-level ablation effect vs threshold; (b) random unit sets of equal size."""
import sys; sys.path.insert(0, "../../src")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import jax.numpy as jnp
import learned_analysis as LA, learned as Lm
from semantics import sample_binary
from c1_iagree_synthetic import code_crit
A, B, g = LA.grid(); agp, cfp = LA.matched_pairs()
for task, arch, seed in [("T1", "TF1", 0), ("T1", "TF1", 3), ("T1", "TF2", 0)]:
    P = LA.load(task, arch, seed); L = len(P["layers"])
    _, st = LA.capture(P, task, LA.inp_bin(A, B, 0*A)); bt = sample_binary(task, 4000, np.random.default_rng(3000))
    _, sg = LA.capture(P, task, LA.inp_bin(bt.a, bt.b, bt.p)); gm = {l: sg[f"mlp_hidden{l}"][:, -1].mean(0) for l in range(L)}
    stats = {(l, u): code_crit(st[f"mlp_hidden{l}"][:, -1, u]) for l in range(L) for u in range(32)}
    val = Lm.sample(task, 20000, np.random.default_rng(1000))
    def run(sel):
        def hk(name, arr):
            for l in range(L):
                us = [u for (ll, u) in sel if ll == l]
                if name == f"mlp_hidden{l}" and us:
                    a_ = np.asarray(arr).copy(); a_[:, -1, us] = gm[l][us]; return jnp.asarray(a_)
            return arr
        ma = LA.fwd_margin(P, task, LA.inp_bin(agp[:, 0], agp[:, 1], 0*agp[:, 0]), hk)
        mc = LA.fwd_margin(P, task, LA.inp_bin(cfp[:, 0], cfp[:, 1], 0*cfp[:, 0]), hk)
        lg = np.asarray(Lm.forward(P, task, Lm.to_inp(task, val), hk))
        return float(np.mean(ma - mc)), float((np.where(lg[:, 0] > lg[:, 1], 1., -1.) == val.y).mean())
    print(task, arch, seed, "full", run([]))
    for tau in [0.1, 0.15, 0.2, 0.25, 0.3]:
        sel = [k for k, (ds, ia) in stats.items() if ds >= tau and ia >= tau]
        print(f"  tau={tau}: n={len(sel)}  (d_agree, acc) = {run(sel)}")
    sel = [k for k, (ds, ia) in stats.items() if ds >= .2 and ia >= .2]; allk = list(stats)
    rng = np.random.default_rng(0); res = [run([allk[i] for i in rng.choice(len(allk), len(sel), replace=False)]) for _ in range(10)]
    print(f"  random sets of size {len(sel)}: d_agree {np.round([r[0] for r in res],1)}  acc {np.round([r[1] for r in res],3)}")
    comp = [k for k in allk if k not in sel]; print(f"  complement (n={len(comp)}):", run(comp))
