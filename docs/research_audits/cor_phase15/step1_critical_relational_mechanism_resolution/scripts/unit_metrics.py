"""XIII-C: all continuous implementation measures (src/metrics.py) for units at given steps.
Usage: UM_STEPS=all|final|25,100,... UM_TAG=_x python3 scripts/unit_metrics.py SPEC...   (SPEC as in src/xc.py)
Appends to logs/unit_metrics<TAG>.jsonl one line per (unit, step); skips (unit, step) pairs already present (resumable)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import xc, metrics as MT
out = os.path.join(HERE, "..", "logs", "unit_metrics" + os.environ.get("UM_TAG", "") + ".jsonl"); done = set()
if os.path.exists(out):
    for l in open(out): r = json.loads(l); done.add((r["unit"], r["step"]))
S = MT.sets(int(os.environ.get("UM_STREAM", 86_100_000)))
for sp in sys.argv[1:]:
    st = os.environ.get("UM_STEPS", "final"); av = xc.steps_available(sp)
    steps = av if st == "all" else [max(av)] if st == "final" else [s for s in map(int, st.split(",")) if s in av]
    for k in steps:
        if (sp, k) in done: continue
        t0 = time.time(); M = xc.load(sp, k); r = MT.all_metrics(M, S, rho=os.environ.get("UM_RHO", "1") == "1")
        r.update(unit=sp, step=k, factors=xc.factors(sp), sec=time.time() - t0)
        with open(out, "a") as fh: fh.write(json.dumps(r) + "\n")
        print(sp, k, {a: round(b, 3) for a, b in r.items() if isinstance(b, float)}, flush=True)
