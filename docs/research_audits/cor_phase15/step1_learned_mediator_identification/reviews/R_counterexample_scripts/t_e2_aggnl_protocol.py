"""(e2) Run the frozen protocol (run_protocol.analyse, unchanged) on AGG-NL with eval pairs drawn near |z|=1.5.
Only the eval-pair generator is replaced; fit pairs, bank, ev and all criteria are the protocol's."""
import sys
import numpy as np
from common import *  # noqa
sys.argv = ["x"]
import run_protocol as RP
orig = RP.pairs
def pairs(task, n, seed):
    if seed in (6002, 6102):
        A, C = custom_pairs(task, n, np.random.default_rng(seed), 1.45, 1.55)
        return A, C
    return orig(task, n, seed)
RP.pairs = pairs
for arch in sys.argv[1:] or ["TF2", "TF1"]:
    pass
for arch in ("TF2", "TF1"):
    for s in range(5):
        A, C = custom_pairs("AGG-NL", 500, np.random.default_rng(6002), 1.45, 1.55)
        # systematic direction: does agreement push the decision 'inside' (+1)?
        P = load("T6", arch, s, "new")
        mA = MD.margin(np.asarray(M.forward(P, "T6", MD.to_inp("AGG-NL", A)))); mC = MD.margin(np.asarray(M.forward(P, "T6", MD.to_inp("AGG-NL", C))))
        r = RP.analyse("AGG-NL", arch, s, ROOT / "checkpoints" / "new", "train")
        key = list(r)[0]; x = r[key]
        print(arch, s, key, "n_model_differs", x["n_pairs_model_differs"], "mean(mA-mC)", round(float((mA - mC).mean()), 3),
              "frac mA>mC", round(float((mA > mC).mean()), 3), "verdict", x.get("verdict"), "k", x.get("k"),
              "cut", x.get("iia_full_site"), "iia", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.get("iia", {}).items()},
              "nec", x.get("necessity", {}).get("flip_after"), "suf", x.get("sufficiency"),
              "surr", {k: (round(v['acc'], 3), round(v['pair_flip'], 3)) for k, v in x.get("surrogates", {}).items()},
              "R2sab", np.round(x.get("interaction_R2_by_sign_ab", []), 3).tolist(), flush=True)
