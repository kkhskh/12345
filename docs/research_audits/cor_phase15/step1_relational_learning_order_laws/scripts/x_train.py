"""XV training of fresh GATE-architecture models on a task-family condition (src/xvtask.py).
Recipe identical to XIV g_train.py / the continuation trainer: init train_tasks.init("gate", SEED) (N.init "M": 3 layers, 4 heads, d 64),
AdamW lr 1e-3, warmup 200, cosine decay to STEPS (4000), weight decay 1e-4, batch 256 episodes x 6 questions, loss train_tasks.loss(..., "gate").
Only the data generator differs.  Data rng: 41_000_000 + 1000 * SEED (the same stream seed in every condition: paired seeds).
Usage: x_train.py COND SEED OUTDIR [STOP]   COND = a:beta:nu (e.g. 2:0.9:0.3333).  BLIND=1: no loss value is printed or saved
(confirmation runs); otherwise losses every 100 steps go to OUTDIR/hist.json.  Snapshots: every 25 steps to 2000, every 250 to STOP.
Writes OUTDIR/step<k>.npz, OUTDIR/done.json (timing, config hash), and resume state OUTDIR/resume.pkl every 500 steps."""
import json, os, sys, time, pickle, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import train_tasks as TT, xvtask as XT
cond, seed, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]; STEPS = 4000; STOP = int(sys.argv[4]) if len(sys.argv) > 4 else STEPS
a, beta, nu = int(cond.split(":")[0]), float(cond.split(":")[1]), float(cond.split(":")[2]); BLIND = os.environ.get("BLIND") == "1"
snaps = set(range(25, 2001, 25)) | set(range(2250, STEPS + 1, 250)); os.makedirs(OUT, exist_ok=True)
p = TT.init("gate", seed)
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 200, STEPS), weight_decay=1e-4); st = opt.init(p)
@jax.jit
def step(p, st, X, Y, Tg):
    l, g = jax.value_and_grad(TT.loss)(p, X, Y, Tg, "gate"); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
rng = np.random.default_rng(41_000_000 + 1000 * seed); i0 = 0; hist = []
rp = os.path.join(OUT, "resume.pkl")
if os.path.exists(rp):
    R = pickle.load(open(rp, "rb")); p, st, i0, hist = R["p"], R["st"], R["i"], R["hist"]; rng.bit_generator.state = R["rng"]
init_hash = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(TT.init("gate", seed)))).hexdigest()
t0 = time.time(); Tg0 = jnp.zeros((256, 6, 3), jnp.int32)
for i in range(i0, STOP):
    d = XT.sample(256, rng, a, beta, nu); p, st, l = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), Tg0)
    if not BLIND and (i + 1) % 100 == 0: hist.append((i + 1, float(l)))
    if (i + 1) in snaps: np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(x) for x in jax.tree_util.tree_leaves(p)])
    if (i + 1) % 500 == 0 and i + 1 < STOP:
        pickle.dump(dict(p=p, st=st, i=i + 1, hist=hist, rng=rng.bit_generator.state), open(rp + ".tmp", "wb")); os.replace(rp + ".tmp", rp)
if not BLIND: json.dump(dict(hist=hist), open(os.path.join(OUT, "hist.json"), "w"))
json.dump(dict(cond=cond, seed=seed, stop=STOP, seconds=time.time() - t0, init_sha256=init_hash, blind=BLIND), open(os.path.join(OUT, "done.json"), "w"))
if os.path.exists(rp): os.remove(rp)
