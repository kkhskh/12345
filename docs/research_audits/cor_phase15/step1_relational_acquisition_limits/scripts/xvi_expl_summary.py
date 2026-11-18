"""EXPLORATORY summary tables of the cue-conditioned probes and the learning-signal estimates -> logs/expl/expl_summary.json (printed)."""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(HERE, "..", "logs", "expl"); out = dict(probes={}, gradsig={})
for f in sorted(glob.glob(os.path.join(E, "probe_*.json"))):
    d = json.load(open(f)); out["probes"][d["ckpt"]] = dict(follow=d["follow"], EVID=d["baselines"]["EVID"], nat_post2R=d["natural"]["resid_post2@R"], nat_pre2E=d["natural"]["resid_pre2@E"],
        decor_pre2E=d["decorrelated"]["resid_pre2@E"], decor_post2R=d["decorrelated"]["resid_post2@R"], decor_post2E=d["decorrelated"]["resid_post2@E"],
        use_pre2E=d["use"]["resid_pre2@E"], use_post2R=d["use"]["resid_post2@R"], use_base=d["use_baseline_no_patch"])
for f in sorted(glob.glob(os.path.join(E, "gs2_*.json"))):
    d = json.load(open(f)); out["gradsig"][os.path.basename(f)[4:-5]] = dict(gate_z=d["gate"]["z"], gate_norm=d["gate"]["norm"], null_z=d["null"]["z"], planted=d["planted"]["recovered_ratio"], label_z=d["label"]["z"], total_norm=d["total"]["norm"])
nz = [v["null_z"] for v in out["gradsig"].values()]; pr = [v["planted"] for v in out["gradsig"].values()]
out["gradsig_controls"] = dict(n=len(nz), null_z_range=[min(nz), max(nz)], null_abs_gt2=sum(abs(z) > 2 for z in nz), planted_range=[min(pr), max(pr)], label_z_min=min(v["label_z"] for v in out["gradsig"].values()))
json.dump(out, open(os.path.join(E, "expl_summary.json"), "w"), indent=1)
for k, v in out["probes"].items(): print("P", k.ljust(22), "follow %.2f EVID %.3f nat@R %.2f | decor E2 %.2f post2E %.2f post2R %.2f | use E2 %.2f R %.2f base %.2f" % (v["follow"], v["EVID"], v["nat_post2R"], v["decor_pre2E"], v["decor_post2E"], v["decor_post2R"], v["use_pre2E"], v["use_post2R"], v["use_base"]))
for k, v in out["gradsig"].items(): print("G", k.ljust(24), "gate z %7.2f |mu| %.4f null z %5.2f planted %.2f label z %6.1f |g| %.3f" % (v["gate_z"], v["gate_norm"], v["null_z"], v["planted"], v["label_z"], v["total_norm"]))
print(out["gradsig_controls"])
