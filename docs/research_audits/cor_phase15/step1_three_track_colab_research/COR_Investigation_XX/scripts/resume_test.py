"""Runtime resume + determinism test (preflight; review C M5).  python scripts/resume_test.py  -> $COR_RUN_ROOT/preflight/RESUME_TEST.json
On THIS runtime (GPU if present): three short runs of the real harness (development parent uP160, benchmark-only data order 990201,
300 steps of the historical schedule): A uninterrupted; B killed after the step-150 snapshot and resumed; C uninterrupted again.
resume_bitwise_identical: every snapshot of B equals A's; deterministic: C equals A.  Outputs are deleted afterwards (hashes kept)."""
import os, sys, subprocess, shutil, json, time
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG)
from cor_research import env as E
E.configure(); root = os.path.join(E.RUN_ROOT, "preflight", "resume_test"); os.makedirs(os.path.join(root, "ft"), exist_ok=True)
def run(suf, die=None):
    env = dict(os.environ, PYTHONPATH=PKG, COR_RUN_ROOT=root, FX_NOFALLBACK="1", FX_STOP="300", FX_TAGSUF=suf, **({"FX_DIE_AFTER": str(die)} if die else {}))
    return subprocess.run([sys.executable, E.HARNESS, "uP160", "21", "21", "990201"], cwd=root, env=env, capture_output=True, text=True).returncode
t0 = time.time(); rc = dict(A=run("_rtA"), B1=run("_rtB", 150), B2=run("_rtB"), C=run("_rtC"))
d = lambda s: os.path.join(root, "ft", f"uP160_fx_L21_W21_D990201_stop300{s}")
snaps = sorted(f for f in os.listdir(d("_rtA")) if f.startswith("step") and f.endswith(".npz")) if os.path.isdir(d("_rtA")) else []
H = {s: {f: E.sha256(os.path.join(d(s), f)) for f in snaps if os.path.exists(os.path.join(d(s), f))} for s in ("_rtA", "_rtB", "_rtC")}
res = dict(device=os.environ.get("COR_DEVICE"), identity=E.identity(), exit_codes=rc, snapshots=snaps, resume_log=E.read_json(os.path.join(d("_rtB"), "resume_log.json")),
           resume_bitwise_identical=bool(snaps) and rc["B1"] == 3 and rc["B2"] == 0 and H["_rtA"] == H["_rtB"], deterministic_rerun=bool(snaps) and H["_rtA"] == H["_rtC"],
           hashes=H, seconds=time.time() - t0)
shutil.rmtree(os.path.join(root, "ft"), ignore_errors=True); E.atomic_json(res, os.path.join(E.RUN_ROOT, "preflight", "RESUME_TEST.json"))
print(json.dumps({k: res[k] for k in ("device", "exit_codes", "resume_bitwise_identical", "deterministic_rerun", "seconds")})); sys.exit(0 if res["resume_bitwise_identical"] and res["deterministic_rerun"] else 1)
