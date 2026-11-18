"""XVI trainer: fresh or continued GATE-architecture models on a two-reliability task condition (src/xvitask.py), with dense INLINE
diagnostics (src/xvidiag.py) and optimizer logging.  Architecture and recipe as XV (train_tasks.init("gate", SEED): N.init "M",
3 layers, 4 heads, d 64; AdamW, weight decay 1e-4, batch 256 episodes x 6 questions, loss train_tasks.loss(..., "gate")).

Usage: xvi_train.py COND SEED OUTDIR SCHED STEPS DIAG_EVERY [INIT_NPZ]
  COND  a:b1:b0:nu  (b1 pair reliability, b0 single reliability; a:0.9:0.9:nu is XV's a:0.9:nu exactly)
  SCHED cosine : XV's schedule (warmup 0 -> 1e-3 over 200 steps, cosine decay to 0 at STEPS)
        const  : warmup 0 -> 1e-3 over 200 steps, then constant 1e-3
        cont   : parameters from INIT_NPZ (an XV step4000 snapshot), a NEW AdamW state (XV did not save optimizer state), warmup
                 0 -> 1e-3 over 200 steps then constant 1e-3 (a learning-rate restart).  The data stream continues exactly where the
                 XV run stopped (its 4000 batches are regenerated and discarded first: the generator's random consumption does not
                 depend on parameters).
Data rng: 41_000_000 + 1000 * SEED (XV's formula).  Diagnostic set: stream 99_700_000, 1500 episodes (fixed per condition).
Outputs (BLIND=1 prints nothing; every outcome goes to files that are not read before the frozen scorer runs):
  OUTDIR/traj.jsonl : one line per diagnostic point (step 0 and every DIAG_EVERY steps): xvidiag scores plus lr, cumulative lr,
                      mean training loss since the previous point, gradient norm, update norm, parameter norm, Adam moment norms.
  OUTDIR/step<k>.npz: parameters every 500 steps and at STEPS (same format as XV snapshots).
  OUTDIR/done.json  : configuration, init hash, timing, blind flag.   OUTDIR/resume.pkl: resume state every 500 steps (removed at end)."""
import json, os, sys, time, pickle, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "step1_foundational_atlas_continuation", "src")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import train_tasks as TT, xvitask as XT, xvidiag as XD, gatelib as GL
cond, seed, OUT, SCHED, STEPS, DEV = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
INIT = sys.argv[7] if len(sys.argv) > 7 else None
a, b1, b0, nu = XD.parse(cond); BLIND = os.environ.get("BLIND") == "1"; LR, WU = 1e-3, 200; STREAM, NDIAG = 99_700_000, 1500
assert SCHED in ("cosine", "const", "cont") and (SCHED == "cont") == (INIT is not None)
os.makedirs(OUT, exist_ok=True)
if SCHED == "cosine": sched = optax.warmup_cosine_decay_schedule(0.0, LR, WU, STEPS)
else: sched = lambda c: LR * jnp.minimum(1.0, c / WU)
lr_at = lambda i: float(sched(i))                       # learning rate applied at update i (0-indexed), as optax evaluates it
p = TT.init("gate", seed) if INIT is None else GL.load(INIT)
init_hash = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(p))).hexdigest()
opt = optax.adamw(sched, weight_decay=1e-4); st = opt.init(p)
gn = lambda t: jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree_util.tree_leaves(t)))
@jax.jit
def step(p, st, X, Y, Tg):
    l, g = jax.value_and_grad(TT.loss)(p, X, Y, Tg, "gate"); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l, gn(g), gn(up)
rng = np.random.default_rng(41_000_000 + 1000 * seed)
if SCHED == "cont":
    for _ in range(4000): XT.sample(256, rng, a, b1, b0, nu)          # continue the XV data stream where the XV run stopped
D = XD.Diag(cond, STREAM, NDIAG, os.path.join(HERE, "..", "logs"))
tp = os.path.join(OUT, "traj.jsonl"); rp = os.path.join(OUT, "resume.pkl"); i0, cum, lsum, ln = 0, 0.0, 0.0, 0
if os.path.exists(rp):
    R = pickle.load(open(rp, "rb")); p, st, i0, cum = R["p"], R["st"], R["i"], R["cum"]; rng.bit_generator.state = R["rng"]
    keep = [l for l in open(tp) if json.loads(l)["step"] <= i0] if os.path.exists(tp) else []; open(tp, "w").writelines(keep)
elif os.path.exists(tp): os.remove(tp)
def diag(i, gnorm, unorm):
    r = D.score(GL.probs(p, D.X), GL.probs(p, D.Xf), GL.probs(p, D.Xw))
    mu, nv = st[0].mu, st[0].nu
    r.update(step=i, lr=lr_at(max(i - 1, 0)) if i > 0 else 0.0, cum_lr=cum, train_loss=(lsum / ln if ln else None), grad_norm=gnorm, update_norm=unorm,
             param_norm=float(gn(p)), adam_mu_norm=float(gn(mu)), adam_nu_norm=float(gn(nv)))
    with open(tp, "a") as f: f.write(json.dumps(r) + "\n")
if i0 == 0: diag(0, None, None)
t0 = time.time(); Tg0 = jnp.zeros((256, 6, 3), jnp.int32)
for i in range(i0, STEPS):
    d = XT.sample(256, rng, a, b1, b0, nu); p, st, l, g_, u_ = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), Tg0)
    cum += lr_at(i); lsum += float(l); ln += 1
    if (i + 1) % DEV == 0: diag(i + 1, float(g_), float(u_)); lsum, ln = 0.0, 0
    if (i + 1) % 500 == 0 or i + 1 == STEPS: np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(x) for x in jax.tree_util.tree_leaves(p)])
    if (i + 1) % 500 == 0 and i + 1 < STEPS:
        pickle.dump(dict(p=p, st=st, i=i + 1, cum=cum, rng=rng.bit_generator.state), open(rp + ".tmp", "wb")); os.replace(rp + ".tmp", rp)
json.dump(dict(cond=cond, seed=seed, sched=SCHED, steps=STEPS, diag_every=DEV, init=INIT, init_sha256=init_hash, seconds=time.time() - t0,
               blind=BLIND, cum_lr=cum, diag_stream=STREAM, diag_n=NDIAG), open(os.path.join(OUT, "done.json"), "w"))
if os.path.exists(rp): os.remove(rp)
if not BLIND: print("done", cond, seed, SCHED, STEPS, round(time.time() - t0), "s")
