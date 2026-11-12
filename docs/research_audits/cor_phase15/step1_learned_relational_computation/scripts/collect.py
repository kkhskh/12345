"""Collect development results into logs/dev_summary.json and print compact tables."""
import json, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
def head(r):
    F, Q = r["families"], r["quality"]; g = lambda f, rg, k: F[f].get(rg, {}).get(k, float("nan"))
    return dict(KL=Q["KL"], acc=Q["acc"], E_R1=g("F45", "R1", "E"), absE_R1=g("F45", "R1", "absE"), dm_R1=g("F45", "R1", "dmM"), E_R2=g("F45", "R2", "E"), E_R3=g("F45", "R3", "E"),
                absE_R23=(g("F45", "R2", "absE") * g("F45", "R2", "n") + g("F45", "R3", "absE") * g("F45", "R3", "n")) / (g("F45", "R2", "n") + g("F45", "R3", "n")),
                E_F34R1=g("F34", "R1", "E"), E_S45R1=g("S45", "R1", "E"), F54o=g("F54o", "all", "dmM"), F0abs=g("F0", "all", "abs_dmM"), FLOCabs=g("FLOC", "all", "abs_dmM"))
out = {}
for sub in ("compose", "dev", "xtok"):
    for f in sorted(glob.glob(os.path.join(R, "checkpoints", sub, "*.json"))):
        j = json.load(open(f)); ev = j["evals"]; out[f"{sub}/{os.path.basename(f)[:-5]}"] = {k: head(v) for k, v in ev.items()} | {"_meta": {k: j.get(k) for k in ("params", "extra_params", "seconds")}}
json.dump(out, open(os.path.join(R, "logs", "dev_summary.json"), "w"), indent=1)
for k, v in out.items():
    last = max((s for s in v if s != "_meta"), key=int); h = v[last]
    print(f"{k:45s} @{last:>5s} KL {h['KL']:.4f} acc {h['acc']:.3f} E_R1 {h['E_R1']:+.3f} |E_R1| {h['absE_R1']:.3f} dm_R1 {h['dm_R1']:+.3f} |E|R23 {h['absE_R23']:.3f} F34 {h['E_F34R1']:+.3f} F0 {h['F0abs']:.3f}")
