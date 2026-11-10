"""Independent confirmation cohort (mandate §7).  Usage: cohort_train.py ARM SEED      ARM in {T1, P}
Fresh random initialisation; the ORIGINAL protocols of the two curricula, unchanged:
  T1 : step1_latent_reliability nets.train("T1", "M", SEED, 6000 steps, K=3, arch tf)  (protocol of all earlier T1 parents)
  P  : step1_foundational_atlas_continuation train_tasks.train("pcount", SEED, 3000 steps)  (protocol of P-count seeds 0-4)
then the identical frozen-feature T4 transfer of Investigation III (train_ext.train T4, 2000 steps, layers 0-1 + embeddings
frozen, layer 2 reset, T4 RNG seed = SEED; evaluation 20,000 episodes stream 6_300_000).
Seeds 60-67 for both arms; no earlier package used seeds 60-67 for any T1, P-count or T4 model.
Outputs checkpoints/{pre,ft}_{ARM}_s{SEED}.{npz,json} and .log; nothing is read from earlier checkpoints."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
arm, seed = sys.argv[1], int(sys.argv[2]); out = os.path.join(HERE, "..", "checkpoints")
SMOKE = os.environ.get("COHORT_SMOKE") == "1"                     # import/IO smoke test only: tiny step counts, output to /tmp
if SMOKE: out = "/tmp/cohort_smoke"
os.makedirs(out, exist_ok=True); S_T1, S_P, S_FT, N_EV = (10, 10, 10, 500) if SMOKE else (6000, 3000, 2000, 20000)
lf = open(os.path.join(out, f"{arm}_s{seed}.log"), "w"); log = lambda s: (lf.write(s + "\n"), lf.flush())
if arm == "T1":
    sys.path.insert(0, os.path.join(ROOT, "step1_latent_reliability", "src"))
    import nets as N
    p, h, dt = N.train("T1", "M", seed, S_T1, K=3, log=log, arch="tf"); ev = N.evaluate(p, "T1", K=3, seed=999, arch="tf")
else:
    sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src"))
    import train_tasks as TT
    p, h, dt = TT.train("pcount", seed, S_P, log=log); ev = TT.evaluate(p, "pcount", seed=41_000_000)
sys.path.insert(0, os.path.join(ROOT, "step1_relational_computation_atlas", "src"))
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src"))
sys.path.insert(0, os.path.join(ROOT, "step1_relational_mechanism_deepening", "src"))
import ckpt as C, train_ext as TE
C.save(os.path.join(out, f"pre_{arm}_s{seed}.npz"), p)
json.dump(dict(job=f"pre_{arm}_s{seed}", seconds=dt, eval=ev, hist=h), open(os.path.join(out, f"pre_{arm}_s{seed}.json"), "w"), indent=1, default=float)
p0 = dict(p); p0.pop("WX", None)
q, h2, dt2 = TE.train("T4", seed, S_FT, p0=p0, reinit_layers=(2,), freeze_layers=(0, 1), freeze_embed=True, log=log)
ev2 = TE.evaluate(q, "T4", N_EV, 6_300_000); ev2.pop("X"); ev2.pop("Y")
C.save(os.path.join(out, f"ft_{arm}_s{seed}.npz"), q)
json.dump(dict(job=f"ft_{arm}_s{seed}", seconds=dt2, eval=ev2, hist=h2), open(os.path.join(out, f"ft_{arm}_s{seed}.json"), "w"), indent=1, default=float)
