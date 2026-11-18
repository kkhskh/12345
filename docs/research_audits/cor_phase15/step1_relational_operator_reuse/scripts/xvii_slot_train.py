"""XVII slot-targeted training intervention (Experiment D; also the embedding-only rescue test of Experiment C).
Usage: xvii_slot_train.py COND CKPT OUTDIR STEPS MODE SLOTS LOSSSLOTS DATASEED [DIAG_EVERY=50] [NDIAG=600]
  MODE  all    : every parameter trainable
        eq     : only the rows SLOTS of E_q trainable (all shared weights frozen)
        noteq  : every parameter except E_q trainable
  SLOTS comma list (rows of E_q trainable in MODE eq; ignored otherwise; '-' for none)
  LOSSSLOTS comma list of question slots whose loss is used ('all' = all six), others weighted 0
Training: AdamW lr 1e-3 constant (warm-up 50), weight decay 1e-4 (on trainable parameters), batch 256, generator src/xvitask.py with data
stream 61_000_000 + DATASEED.  Per-slot F_G (src/xviislot.SlotDiag, stream 99_800_000, NDIAG episodes) at step 0 and every DIAG_EVERY steps,
plus the norm of the change of every E_q row and of the shared weights.  Writes OUTDIR/slot_traj.json and the final parameters."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_latent_reliability", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import gatelib as GL, xvitask as T, xvidiag as XD, xviislot as XS, nets as N
cond, ck, OUT, STEPS, MODE, SLOTS, LSL, DS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7], int(sys.argv[8])
DE = int(sys.argv[9]) if len(sys.argv) > 9 else 50; ND = int(sys.argv[10]) if len(sys.argv) > 10 else 600
a, b1, b0, nu = XD.parse(cond); os.makedirs(OUT, exist_ok=True)
p0 = GL.load(ck); rows = [] if SLOTS in ("-", "") else [int(x) for x in SLOTS.split(",")]
lw = jnp.ones(6) if LSL == "all" else jnp.zeros(6).at[jnp.array([int(x) for x in LSL.split(",")])].set(1.0)
def mask_tree(p):
    m = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) if MODE in ("all", "noteq") else jnp.zeros_like(x), p)
    if MODE == "eq": m["Eq"] = jnp.zeros_like(p["Eq"]).at[jnp.array(rows)].set(1.0)
    if MODE == "noteq": m["Eq"] = jnp.zeros_like(p["Eq"])
    return m
M = mask_tree(p0)
def loss(p, X, Y):
    lp = jax.nn.log_softmax(N.forward(p, X)["answer"], -1); nll = -jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]   # N,Q
    return (nll * lw).sum() / (lw.sum() * X.shape[0])
sched = lambda c: 1e-3 * jnp.minimum(1.0, c / 50)
opt = optax.chain(optax.masked(optax.adamw(sched, weight_decay=1e-4), jax.tree_util.tree_map(lambda m: bool(m.max() > 0), M)), )
@jax.jit
def step(p, st, X, Y):
    l, g = jax.value_and_grad(loss)(p, X, Y); g = jax.tree_util.tree_map(lambda gg, mm: gg * mm, g, M)
    up, st = opt.update(g, st, p); up = jax.tree_util.tree_map(lambda u, mm: u * mm, up, M); return optax.apply_updates(p, up), st, l
p = p0; st = opt.init(p); rng = np.random.default_rng(61_000_000 + DS); D = XS.SlotDiag(cond, n=ND)
shared = lambda q: jnp.sqrt(sum(jnp.sum(x ** 2) for k, v in q.items() if k != "Eq" for x in jax.tree_util.tree_leaves(v)))
res = dict(cond=cond, ckpt=ck, mode=MODE, slots=rows, loss_slots=LSL, data_seed=DS, steps=STEPS, traj=[]); t0 = time.time()
def rec(i):
    s = D.score(p); dq = np.linalg.norm(np.asarray(p["Eq"]) - np.asarray(p0["Eq"]), axis=1).tolist()
    dsh = float(shared(jax.tree_util.tree_map(lambda x, y: x - y, {k: v for k, v in p.items() if k != "Eq"}, {k: v for k, v in p0.items() if k != "Eq"})))
    res["traj"].append(dict(step=i, F_G=s["F_G"], KL=s["KL"], dEq=dq, dShared=dsh)); print(i, [round(x, 2) for x in s["F_G"]], "dEq", [round(x, 2) for x in dq], "dShared %.3f" % dsh, flush=True)
rec(0)
for i in range(STEPS):
    d = T.sample(256, rng, a, b1, b0, nu); p, st, l = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]))
    if (i + 1) % DE == 0: rec(i + 1)
res["seconds"] = time.time() - t0; json.dump(res, open(os.path.join(OUT, "slot_traj.json"), "w"), indent=1)
np.savez(os.path.join(OUT, f"step{STEPS}.npz"), *[np.asarray(x) for x in jax.tree_util.tree_leaves(p)])
