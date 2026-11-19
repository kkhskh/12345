"""Development table: every XIX development continuation measured by measure18 on stream 86 100 000 (logs/dev/measure_*.jsonl), with
phi_nec against the pair's fullB / fullI anchors (training arms: the same anchors -- fullB is the baseline continuation and fullI the
block continuation of the same matched step-100 states).  Writes logs/dev/dev_table19.json and prints it.  Development evidence only."""
import json, glob, os, math, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src")); import xix_score as XS
R = {}
for f in sorted(glob.glob(os.path.join(H, "..", "logs", "dev", "measure_*.jsonl"))):
    for l in open(f):
        r = json.loads(l); x = r["final"]["1211"]; R[r["run"]] = dict(nec_E=x["nec_E"], nec_M=x["nec_M"], KL=x["KL"], acc=x["acc"], cls=x["cls2_causal"])
out = {}
for base, L in (("E", 21), ("M", 23)):
    pre = f"uP160_fx_L{L}_W{L}_D801"; b = R.get(f"{pre}_stop1211_br100_{base}801_fullB"); i = R.get(f"{pre}_stop1211_br100_{base}801_fullI")
    if not (b and i): continue
    for k, v in R.items():
        if not k.startswith(pre): continue
        arm = k.replace(f"{pre}_stop1211_br100_{base}801_", "tp:").replace(f"{pre}_stop1211_br100_ctl_", "ctl:").replace(pre + "_", "train:")
        v = dict(v, phi=XS.phi_nec(v, b, i), competent=bool(v["KL"] <= 0.045 and v["acc"] >= 0.80)); out[f"{base}|{arm}"] = v
        print(f"{base} {arm[:34]:34s} nec_E {v['nec_E']:.3f} nec_M {v['nec_M']:.3f} KL {v['KL']:.3f} acc {v['acc']:.3f} {v['cls']:8s} phi {v['phi']:+.2f}")
json.dump(out, open(os.path.join(H, "..", "logs", "dev", "dev_table19.json"), "w"), indent=1)
