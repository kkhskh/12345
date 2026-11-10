"""Quality-manipulation variant: the identical T4 frozen-feature transfer with a smaller (or larger) step budget.
Usage: transfer_budget.py NAME PARENT_NPZ SEED STEPS
Same trainer and settings as every transfer in Investigations III-V (train_ext.train T4; layers 0-1 + embeddings frozen;
layer 2 reset; RNG seed SEED); only `steps` differs (the warmup/cosine schedule scales with it, as in the original code).
Writes checkpoints/budget/NAME.{npz,json}.  Evaluation: 20,000 episodes stream 6_300_000."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src", "step1_relational_mechanism_deepening/src"):
    sys.path.insert(0, os.path.join(ROOT, p_))
import ckpt as C, train_ext as TE
name, parent, seed, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out = os.path.join(HERE, "..", "checkpoints", "budget"); os.makedirs(out, exist_ok=True)
p0 = C.load(parent); p0.pop("WX", None)
q, h, dt = TE.train("T4", seed, steps, p0=p0, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True)
ev = TE.evaluate(q, "T4", 20000, 6_300_000); ev.pop("X"); ev.pop("Y")
C.save(os.path.join(out, f"{name}.npz"), q)
json.dump(dict(name=name, parent=os.path.relpath(parent, ROOT), seed=seed, steps=steps, seconds=dt, eval=ev), open(os.path.join(out, f"{name}.json"), "w"), indent=1, default=float)
