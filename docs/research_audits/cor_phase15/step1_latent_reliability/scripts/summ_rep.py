import json, sys
for f in sys.argv[1:]:
    r = json.load(open(f)); t = r["tests"]; j = t["joint"]
    print(f.split("/")[-1], r["criteria"])
    print("  heads route", r["route_heads"], "ctrl", r["control_heads"], "R2eval", [round(x, 2) for x in r["head_group_mass_r2_eval"]], "bayes", {k: round(v, 3) for k, v in r["bayes_comparison"].items()})
    print("  T1/T4 acc", round(r["behaviour"]["T1_stage"]["acc"], 4), round(r["behaviour"]["T4_stage"]["acc"], 4), "bayes", round(r["behaviour"]["T4_stage"]["bayes_acc"], 4))
    print("  joint cond iia %.3f kl %.3f iia_bayes %.3f shift %.3f | perm %.3f | countpres chg %.3f | target chg %.3f" % (j["cond"]["iia"], j["cond"]["kl_recovered"], j["cond"]["iia_bayes"], j["cond"]["routing_shift"], j["perm"]["iia"], j["countpres"]["change_rate"], j["target"]["change_rate"]))
    print("  key %.3f query %.3f full %.3f mismatch %.3f random %.3f | route-frozen %.3f ctrl-frozen %.3f all-frozen %.3f | pattern %.3f/%.3f" % (t["key_only"]["cond"]["iia"], t["query_only"]["cond"]["iia"], t["full_q_tokens"]["cond"]["iia"], t["joint_mismatch"]["iia"], t["joint_random"], t["joint_route_frozen"]["iia"], t["joint_control_frozen"]["iia"], t["joint_all_frozen"]["iia"], t["pattern_patch_all_heads"]["iia"], t["pattern_patch_all_heads"]["kl_recovered"]))
    R4 = r["R4"]; print("  R4 m", [round(x, 2) for x in R4["m"]], "drop", round(R4["drop"], 2), [round(x, 2) for x in R4["drop_ci"]], "rise", round(R4["rise"], 2), "r2 bayes/mono/lookup", round(R4["r2_bayes"], 3), round(R4["r2_best_monotone_count"], 3), round(R4["r2_flexible_count_lookup"], 3))
    print("  R5 lookup %.3f agg %.3f" % (r["R5"]["r2_tag_from_count_lookup"], r["R5"]["r2_tag_from_flexible_aggregate"]))
