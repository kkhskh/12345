"""XX fresh-parent pretraining: the historical P-count recipe (XIII cohort13_pretrain.py -> train_tasks.train("pcount", seed, 3000)),
re-expressed as a RESUMABLE loop that is step-for-step the same computation (same init, same AdamW warmup-cosine schedule, same
data RNG 31_000_000 + 1000*seed + task index, same batch 256).  Every CKPT_EVERY steps the parameters, the AdamW state, the data
RNG bit-generator state and the loss history are written atomically; a restart resumes exactly (verified in tests/test_pretrain.py
against train_tasks.train itself on a short schedule).
Usage: python -m cor_research.training.pretrain SEED [STEPS=3000]  ->  $COR_PARENT_ROOT/zP<SEED>/{pre.npz, pre.json, lineage.json}
Eligibility of a parent for fine-tuning is NOT decided here (protocol: parent validity = aux_acc >= frozen threshold)."""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E
E.configure(); E.legacy_paths()
import numpy as np, jax, jax.numpy as jnp, optax
import train_tasks as TT, ckpt as C
CKPT_EVERY = int(os.environ.get("COR_PRETRAIN_CKPT", 250))


def run(seed, steps=3000, out=None, task="pcount", die_after=None):
    out = out or os.path.join(E.PARENT_ROOT, f"zP{seed}"); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "pre.json")): return json.load(open(os.path.join(out, "pre.json")))
    p = TT.init(task, seed, "M"); samp = TT.SAMPLERS[task]
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, steps // 10), steps), weight_decay=1e-4); st = opt.init(p)

    @jax.jit
    def step(p, st, X, Y, Tg):
        l, g = jax.value_and_grad(TT.loss)(p, X, Y, Tg, task); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    rng = np.random.default_rng(31_000_000 + 1000 * seed + list(TT.SAMPLERS).index(task)); hist = []; start = 0; t0 = time.time(); prev = 0.0
    rp = os.path.join(out, "resume_pretrain.pkl")
    if os.path.exists(rp):
        R = pickle.load(open(rp, "rb")); assert R["seed"] == seed and R["steps"] == steps and R["task"] == task, "resume file does not match this run"
        p = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(p), [jnp.asarray(a) for a in R["p"]])
        st = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(st), [jnp.asarray(a) for a in R["st"]])
        rng.bit_generator.state = R["rng"]; hist = R["hist"]; start = R["i"]; prev = R.get("seconds", 0.0)
        log = json.load(open(os.path.join(out, "resume_log.json"))) if os.path.exists(os.path.join(out, "resume_log.json")) else []
        log.append(dict(resumed_at=start)); E.atomic_json(log, os.path.join(out, "resume_log.json"))
    for i in range(start, steps):
        d = samp(256, rng); Tg = d.get("T", np.zeros((256, 6, 3), int))
        p, st, l = step(p, st, jnp.asarray(d["X"]), jnp.asarray(d["Y"]), jnp.asarray(Tg))
        if (i + 1) % max(steps // 10, 1) == 0: hist.append((i + 1, float(l)))
        if (i + 1) % CKPT_EVERY == 0 and i + 1 < steps:
            tmp = rp + ".tmp"
            pickle.dump(dict(seed=seed, steps=steps, task=task, i=i + 1, p=[np.asarray(a) for a in jax.tree_util.tree_leaves(p)], st=[np.asarray(a) for a in jax.tree_util.tree_leaves(st)],
                             rng=rng.bit_generator.state, hist=hist, seconds=prev + time.time() - t0), open(tmp, "wb")); os.replace(tmp, rp)
            if die_after and i + 1 >= die_after: sys.exit(3)
    p = jax.tree_util.tree_map(np.asarray, p); ev = TT.evaluate(p, task, seed=41_000_000)
    C.save(os.path.join(out, "pre.tmp.npz"), p); os.replace(os.path.join(out, "pre.tmp.npz"), os.path.join(out, "pre.npz"))
    rec = dict(arm="P", seed=seed, steps=steps, task=task, seconds=prev + time.time() - t0, eval=ev, hist=hist, recipe="train_tasks.train('pcount', seed, steps) (XIII cohort13_pretrain.py)",
               devices=[str(d_) for d_ in jax.devices()], xla_flags=os.environ.get("XLA_FLAGS", ""), pre_sha256=E.sha256(os.path.join(out, "pre.npz")))
    E.atomic_json(dict(parent=f"nP{seed}", init_seed=seed, data_rng_seed=31_000_000 + 1000 * seed + list(TT.SAMPLERS).index(task), architecture="nets/train_tasks size M (3 layers, d=64)",
                       config=dict(optimizer="adamw", lr=1e-3, warmup=min(200, steps // 10), schedule="warmup_cosine", weight_decay=1e-4, batch=256, steps=steps),
                       checkpoint=rec["pre_sha256"], code={f: E.sha256(os.path.join(E.LEGACY, "step1_foundational_atlas_continuation", "src", f)) for f in ("train_tasks.py",)}), os.path.join(out, "lineage.json"))
    E.atomic_json(rec, os.path.join(out, "pre.json"))
    if os.path.exists(rp): os.remove(rp)
    return rec


if __name__ == "__main__":
    s = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    print(json.dumps({k: v for k, v in run(s, n, die_after=int(os.environ.get("COR_DIE_AFTER", 0)) or None).items() if k != "hist"}, default=float))
