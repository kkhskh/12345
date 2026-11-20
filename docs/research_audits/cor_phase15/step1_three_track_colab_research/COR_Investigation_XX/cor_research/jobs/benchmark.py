"""Measured throughput on THIS runtime (mandate 8.4: GPU-hours from measured throughput, not from Colab compute units).
python -m cor_research.jobs.benchmark [--levels 1,2,4] [--steps 400]  ->  $COR_RUN_ROOT/benchmark.json
For each concurrency level k: k simultaneous harness runs (plain fine-tuning, development parent uP160, benchmark-only data orders
990001.., never used scientifically, outputs under $COR_RUN_ROOT/benchmark/ and deleted afterwards); per-run steps/s measured between
the step-100 and step-STEPS snapshots from the harness's own elapsed-time log (excludes JIT compilation); total throughput = sum over
runs.  Also: one pretraining step rate (P-count model, batch 256), one measurement and one Track-B battery wall time, per-snapshot
checkpoint size, peak GPU memory per process (nvidia-smi).  The chosen concurrency is the level with the highest total throughput whose
total GPU memory stays below 85% of the device (or the CPU count on CPU)."""
import os, sys, json, time, subprocess, re, shutil, threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E
DEV = E.configure()


def gpu_info():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        n, m, d = [x.strip() for x in o.split(",")]; return dict(name=n, memory_mb=float(m), driver=d)
    except Exception: return None


def harness_rate(k, steps, root):
    procs = []; mem = {}; stop = threading.Event()
    def poll():
        while not stop.is_set():
            try:
                for line in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines():
                    pid, m = [x.strip() for x in line.split(",")]; mem[int(pid)] = max(mem.get(int(pid), 0), float(m))
            except Exception: pass
            stop.wait(5)
    threading.Thread(target=poll, daemon=True).start(); t0 = time.time()
    for i in range(k):
        env = dict(os.environ, PYTHONPATH=E.PKG, COR_RUN_ROOT=root, FX_NOFALLBACK="1", FX_STOP=str(steps), FX_TAGSUF=f"_bench{k}_{i}")
        procs.append(subprocess.Popen([sys.executable, E.HARNESS, "uP160", "21", "21", str(990001 + i)], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate()[0] for p in procs]; wall = time.time() - t0; stop.set(); rates = []
    for o in outs:
        t = {int(a): float(b) for a, b in re.findall(r"step (\d+) loss \S+ (\d+(?:\.\d+)?)s", o)}
        if 100 in t and steps in t and t[steps] > t[100]: rates.append((steps - 100) / (t[steps] - t[100]))
    d = [x for x in os.listdir(os.path.join(root, "ft")) if f"_bench{k}_" in x]; sz = []
    for x in d:
        fs = [os.path.getsize(os.path.join(root, "ft", x, f)) for f in os.listdir(os.path.join(root, "ft", x)) if f.startswith("step") and f.endswith(".npz")]
        if fs: sz.append(sum(fs) / len(fs))
    return dict(k=k, per_run_steps_per_s=rates, total_steps_per_s=sum(rates) if len(rates) == k else None, wall_s=wall, snapshot_mb=(sum(sz) / len(sz) / 1e6 if sz else None),
                gpu_mem_mb_per_proc=max(mem.values()) if mem else None, ok=len(rates) == k, tail=[o[-400:] for o in outs if len(rates) < k])


def pretrain_rate(n=60):
    code = f"""
import sys, time; sys.path.insert(0, {E.PKG!r})
from cor_research import env as E; E.configure(); E.legacy_paths()
import numpy as np, jax, jax.numpy as jnp, optax, train_tasks as TT
p = TT.init('pcount', 1, 'M'); opt = optax.adamw(1e-3); st = opt.init(p); samp = TT.SAMPLERS['pcount']; rng = np.random.default_rng(0)
@jax.jit
def step(p, st, X, Y, Tg):
    l, g = jax.value_and_grad(TT.loss)(p, X, Y, Tg, 'pcount'); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
def one():
    global p, st
    d = samp(256, rng); p, st, l = step(p, st, jnp.asarray(d['X']), jnp.asarray(d['Y']), jnp.asarray(d.get('T', np.zeros((256, 6, 3), int)))); l.block_until_ready()
for _ in range(5): one()
t = time.time()
for _ in range({n}): one()
print({n} / (time.time() - t))"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=E.PKG))
    try: return float(r.stdout.strip().splitlines()[-1])
    except Exception: return None


def main(levels=(1, 2, 4), steps=400):
    root = os.path.join(E.RUN_ROOT, "benchmark"); os.makedirs(os.path.join(root, "ft"), exist_ok=True); g = gpu_info(); res = dict(device=DEV, gpu=g, cpu_count=os.cpu_count(), levels=[])
    import jax; res["jax_devices"] = [str(d) for d in jax.devices()]
    for k in levels:
        r = harness_rate(k, steps, root); res["levels"].append(r); print(json.dumps({x: r[x] for x in ("k", "total_steps_per_s", "gpu_mem_mb_per_proc", "ok")}), flush=True)
        if not r["ok"]: break
    res["pretrain_steps_per_s"] = pretrain_rate()
    d = [x for x in os.listdir(os.path.join(root, "ft")) if "_bench1_0" in x]
    if d:
        t = time.time(); subprocess.run([sys.executable, "-m", "cor_research.scoring.measure", os.path.join(root, "ft", d[0]), "uP160", str(steps), "1", os.path.join(root, "m.json")], env=dict(os.environ, PYTHONPATH=E.PKG), capture_output=True)
        res["measure_s"] = time.time() - t
        t = time.time(); subprocess.run([sys.executable, "-m", "cor_research.scoring.functional", os.path.join(root, "ft", d[0]), "uP160", str(steps), os.path.join(root, "f.json")], env=dict(os.environ, PYTHONPATH=E.PKG), capture_output=True)
        res["functional_battery_s"] = time.time() - t
    ok = [r for r in res["levels"] if r["ok"] and r["total_steps_per_s"]]
    cap = (0.85 * g["memory_mb"]) if g else None
    ok2 = [r for r in ok if cap is None or r["gpu_mem_mb_per_proc"] is None or r["k"] * r["gpu_mem_mb_per_proc"] < cap]
    best = max(ok2, key=lambda r: r["total_steps_per_s"]) if ok2 else None
    res["chosen_concurrency"] = best["k"] if best else 1; res["chosen_total_steps_per_s"] = best["total_steps_per_s"] if best else None; res["measured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    shutil.rmtree(os.path.join(root, "ft"), ignore_errors=True); E.atomic_json(res, os.path.join(E.RUN_ROOT, "benchmark.json")); return res


if __name__ == "__main__":
    lv = tuple(int(x) for x in sys.argv[sys.argv.index("--levels") + 1].split(",")) if "--levels" in sys.argv else (1, 2, 4)
    st = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 400
    r = main(lv, st); print(json.dumps({k: r[k] for k in ("device", "gpu", "chosen_concurrency", "chosen_total_steps_per_s", "pretrain_steps_per_s")}, indent=1))
