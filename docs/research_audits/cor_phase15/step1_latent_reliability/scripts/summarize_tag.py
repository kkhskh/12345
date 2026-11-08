import json, sys
r = json.load(open(sys.argv[1])); t = r["tests"]
print(r["ckpt"], r["split"], "s_tag", r["s_tag"], "k", r["k_tag"])
print("  probes", {k: round(v, 3) for k, v in r["tag_probes"].items()})
for n in ["tag", "tag_D2", "full_src_q"]:
    print(f"  {n:11s}", " ".join(f"{k}:iia={v['iia']:.2f}/kl={v['kl_recovered']:.2f}/chg={v['change_rate']:.2f}" for k, v in t[n].items()))
print("  reuse q345", t["tag_fit_q012"], "D2:", t.get("tag_D2_fit_q012")); print("  random", t["random"])
print("  necessity", t["necessity"]); print("  content D2", r["content_D2_tag"], "overlap", round(r["overlap_tag_D2"], 3))
print("  routing shift", round(r["routing_shift_fraction_under_tag_patch"], 3), "D2", round(r.get("routing_shift_fraction_under_tag_D2_patch", float("nan")), 3), " pattern patch target pairs", r["pattern_patch_target_pairs"])
if "conf_head_tag" in t: print("  conf", t["conf_head_tag"], t.get("conf_head_tag_D2"))
