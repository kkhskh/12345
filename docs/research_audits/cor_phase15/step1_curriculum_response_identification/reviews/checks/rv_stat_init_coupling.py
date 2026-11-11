"""Statistical reviewer: are cT_s and cP_s (same seed s) initialised identically?  Calls the two arms' init paths exactly as
cohort_train.py does (nets.train("T1","M",seed,...,K=3,arch="tf") -> nets.init; train_tasks.train("pcount",seed) -> train_tasks.init)
and the T4-transfer layer-2 reinit seed (train_ext: N.init(size, env, seed + 7777)).  No training."""
import os, sys
import numpy as np
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src"):
    sys.path.insert(0, os.path.join(H, p_))
import jax, nets as N, bayes as B, train_tasks as TT
src = open(os.path.join(H, "step1_latent_reliability/src/nets.py")).read()
i = src.find("def train("); print("nets.train env line:", [l.strip() for l in src[i:i + 900].splitlines() if "Env(" in l or "init(" in l][:3])
lines = []
for s in (60, 61):
    # replicate nets.train's env construction for task T1, K=3
    pT = N.init("M", B.Env("T1", K=3, Q=6), s); pP = TT.init("pcount", s, "M")
    flat = lambda p: {jax.tree_util.keystr(kp): np.asarray(v) for kp, v in jax.tree_util.tree_flatten_with_path(p)[0]}
    fT, fP = flat(pT), flat(pP); common = sorted(set(fT) & set(fP))
    same = [k for k in common if fT[k].shape == fP[k].shape and np.array_equal(fT[k], fP[k])]
    lines.append(f"seed {s}: {len(common)} shared parameter arrays, {len(same)} bit-identical between T1 and P-count init; "
                 f"P-only arrays {sorted(set(fP) - set(fT))[:6]}")
    pO = N.init("M", B.Env("T1", K=3, Q=6), s + 1); fO = flat(pO)
    lines.append(f"  control seed {s} vs {s + 1} (T1 init): identical arrays {sum(np.array_equal(fT[k], fO[k]) for k in common)}")
out = "\n".join(lines); print(out)
open(os.path.join(H, "step1_curriculum_response_identification", "logs", "review_checks", "rv_stat_init_coupling.out"), "w").write(out + "\n")
