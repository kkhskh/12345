import json, sys
r = json.load(open(sys.argv[1])); s = r["s_star"]; t = r["tests"]
b = r["behaviour"]; print(f"{r['ckpt']} {r['split']}  acc {b['acc']:.4f} bayes {b['bayes_acc']:.4f} own {b['own_question_bayes_acc']:.4f}  s*={s}")
pr = r["probes"]; pi = r["probes_random_init"]
for kind in ["readout", "episode", "src_q"]:
    for tg in ["pi_minus", "pi_episode", "naive", "counts", "hist", "agree_own"]:
        key = f"{s}|{kind}|{tg}"
        if key in pr: print(f"  probe {kind:8s} {tg:10s} trained {pr[key]:.3f} init {pi[key]:.3f}")
print("  site sweep (readout pi/counts):", {x: round(pr.get(f'{x}|readout|{r["selection_target"]}', float('nan')), 3) for x in r["sites"]})
for name in ["D1_probe", "D2_contrast", "D1_fit_q012", "hist_surrogate", "full_residual"]:
    if name in t:
        print(f"  {name:15s} " + " ".join(f"{k}:iia={v.get('iia', float('nan')):.2f}/kl={v.get('kl_recovered', float('nan')):.2f}/chg={v.get('change_rate', float('nan')):.2f}" for k, v in t[name].items() if isinstance(v, dict) and 'iia' in v))
print("  random", t["random"])
for kk, v in t["site_location_sweep_cond"].items(): print(f"  sweep {kk:28s} iia {v['iia']:.2f} kl {v['kl_recovered']:.2f}")
print("  necessity", {k: (round(v, 3) if isinstance(v, float) else {a: round(b, 2) for a, b in v.items()}) for k, v in t["necessity"].items()})
print("  restoration", t["restoration_kl_recovered"]); print("  conf head", t.get("conf_head_D1")); print("  content D2", r["content_D2"], "overlap D1/D2", round(r["overlap_D1_D2"], 3))
for k, v in r["attention"].items():
    print("  att", k, v if not isinstance(v, dict) else f"iia={v['iia']:.2f} kl={v['kl_recovered']:.2f}")
