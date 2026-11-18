"""XVIII measurement of one fine-tuning run directory (frozen in FREEZE_XVIII).
Final diagnostics (src/xviiim.measure_v2) at steps 1000 and 1211 (and 2546 if present), route shares along the early trajectory (unmasked),
and the window logs (window_log.json, slog.json) if present.
Usage: measure18.py RUNDIR PARENT STREAM OUTJSONL.  Appends one JSON line; skips if the run is already in OUTJSONL (resumable)."""
import json, os, sys, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src"))
import numpy as np
import xviiim as XM
D, parent, stream, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; tag = os.path.basename(os.path.normpath(D))
if os.path.exists(out) and any(json.loads(l)["run"] == tag for l in open(out)): print("already", tag); sys.exit(0)
if not os.path.exists(os.path.join(D, "step1211.npz")): print("MISSING step1211", tag); sys.exit(2)
t0 = time.time(); r = dict(run=tag, parent=parent, stream=stream, final={})
for k in (1000, 1211, 2546, 4000):
    if os.path.exists(os.path.join(D, f"step{k}.npz")): r["final"][k] = XM.measure_v2(XM.load_dir(D, parent, k), stream)
Xs = XM.NV.Bm.sample_fast(XM.NV.ENV, 400, np.random.default_rng(stream + 11))["X"]; traj = {}
for k in (25, 50, 100, 150, 200, 300, 600):
    if os.path.exists(os.path.join(D, f"step{k}.npz")): traj[k] = XM.shares(XM.load_dir(D, parent, k), Xs)
r["traj"] = traj
for nm in ("window_log.json", "slog.json"):
    if os.path.exists(os.path.join(D, nm)): r[nm.split(".")[0]] = json.load(open(os.path.join(D, nm)))
r["sec"] = time.time() - t0; open(out, "a").write(json.dumps(r) + "\n")
f = r["final"][1211]; print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if k in ("lik_RO1", "mix_mass", "rhoL", "lapseL", "nec_E", "nec_M", "nec_rq1", "KL", "acc", "cls2")}, flush=True)
