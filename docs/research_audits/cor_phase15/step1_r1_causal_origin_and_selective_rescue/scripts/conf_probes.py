"""Frozen P6 (confirmation): run the accessibility probes (scripts/accessibility_probes.py logic) on the frozen layers 0-1 of each
confirmation parent (pre.npz; identical to the transferred models' frozen layers) and write logs/conf_probes.json keyed yP{seed}."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); out = {}
for s in range(90, 98):
    f = os.path.join(HERE, "..", "checkpoints", f"yP{s}", "pre.npz"); tmp = f"conf_probe_yP{s}.json"
    env = dict(os.environ, PROBE_OUT=tmp); subprocess.run([sys.executable, os.path.join(HERE, "accessibility_probes.py"), f], env=env, check=True)
    out[f"yP{s}"] = list(json.load(open(os.path.join(HERE, "..", "logs", tmp))).values())[0]
json.dump(out, open(os.path.join(HERE, "..", "logs", "conf_probes.json"), "w"), indent=1)
