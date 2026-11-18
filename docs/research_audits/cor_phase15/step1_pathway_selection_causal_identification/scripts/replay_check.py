"""XVIII step 1: bitwise replay of XIV's E/D801 baseline and ro0 real block (steps 100 and 150) with the XVIII copy of the harness."""
import numpy as np, json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(H, "..", "..", "step1_cross_family_relational_foundations", "checkpoints", "ft")
pairs = [("uP160_fx_L21_W21_D801_stop150_replay", "uP160_fx_L21_W21_D801_stop1211_B_base"), ("uP160_fx_L21_W21_D801_ro0T100_stop150_replay", "uP160_fx_L21_W21_D801_ro0T100_stop1211_B_ro0100")]
out = {}
for mine, orig in pairs:
    for k in (100, 150):
        a = np.load(os.path.join(H, "..", "checkpoints", "ft", mine, f"step{k}.npz")); b = np.load(os.path.join(X, orig, f"step{k}.npz"))
        out[f"{orig}@{k}"] = max(float(np.abs(a[f] - b[f]).max()) for f in a.files)
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(H, "..", "logs", "replay_check.json"), "w"), indent=1)
