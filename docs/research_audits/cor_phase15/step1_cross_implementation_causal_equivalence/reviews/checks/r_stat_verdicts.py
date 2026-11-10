"""Recompute every frozen-prediction verdict from logs/{cal,conf}/*.json and logs/site/site_eval.json (reviewer check)."""
import json, os, sys, numpy as np
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
def L(p): return json.load(open(os.path.join(B, p)))
for PH in ["cal", "conf"]:
    R = L(f"logs/{PH}/results.json"); N = L(f"logs/{PH}/necessity.json"); A = L(f"logs/{PH}/analysis.json")
    ms = R["models"]; P = [m for m in ms if m[0] == "P"]; T = [m for m in ms if m[0] == "T"]
    print(f"===== {PH} {ms}")
    W = {m: R["per_model"][m]["within"] for m in ms}
    for m in T:
        s = W[m]["single"]; a = s["enc|T_own|local_group_tokens_only"]["iia_adjusted"]; b = s["enc|T_own|one_token_of_pair_only"]["iia_adjusted"]
        ceil, fl = s["full_site_patch"]["iia"], s["enc|T_own|floor_mismatched"]["iia"]; n = s["full_site_patch"]["n_inf"]
        se = np.sqrt(0.25 / n) / (ceil - fl)
        print(f"I(a) {m} local {a:.3f} pass={a>=0.5} | I(b) one-token {b:.3f} delta {a-b:.3f} pass={a-b>=0.2} | denom {ceil-fl:.3f} n_inf {n} approx SE(adj) {se:.3f}")
    for m in P:
        s = W[m]["single"]; a = s["enc|P_full"]["iia_adjusted"]; ceil, fl = s["full_site_patch"]["iia"], s["enc|P_full|floor_mismatched"]["iia"]
        print(f"I(c) {m} {a:.3f} pass={a>=0.5} denom {ceil-fl:.3f}")
    for m in ms:
        c = W[m]["cond"]; k = "enc|T_own" if m[0] == "T" else "enc|P_full"
        print(f"  cond {m}: enc iia {c[k]['iia']:.3f} ceiling {c['full_site_patch']['iia']:.3f} floor {c[k+'|floor_mismatched']['iia']:.3f} adj {c[k]['iia_adjusted']:.3f}")
    Tr = R["transport"]
    for fam in ["cond", "single"]:
        for met in ["transport_fidelity_iia", "transport_fidelity_r"]:
            for d in ms:
                same = [Tr[f"{d}->{r}|{fam}"][met] for r in ms if r != d and r[0] == d[0]]
                cross = [Tr[f"{d}->{r}|{fam}"][met] for r in ms if r[0] != d[0]]
                dec = R["per_model"][d]["decoder_acc_per_component"]
                print(f"II {fam} {met[19:]} donor {d} dec {dec:.3f} same {np.mean(same):.3f} cross {np.mean(cross):.3f} |diff| {abs(np.mean(cross)-np.mean(same)):.3f} IIa_pass={abs(np.mean(cross)-np.mean(same))<=0.10}"
                      + (f" IIb min cross {min(cross):.3f} pass={min(cross)>=0.85}" if dec >= 0.95 and met.endswith("iia") else ""))
    for m in ms:
        f = R["per_model"][m]["within"]["relabel"] if "relabel" in R["per_model"][m]["within"] else None
        if f: print(f"III {m} count-channel {f['self_decoded_then_encoded']['iia']:.3f} full {f['full_site_patch']['iia']:.3f} pass={f['self_decoded_then_encoded']['iia']<=0.2 and f['full_site_patch']['iia']>=0.2}")
    for m in ms:
        o = N["per_model"][m]; k0 = o["clean"]["KL_own_to_out"]; red = {c: 1 - o[c]["KL_own_to_out"] / k0 for c in ["marg", "null_cross", "rotated"]}
        print(f"IV(a) {m} marg {red['marg']:.4f} null {red['null_cross']:.3f} rot {red['rotated']:.3f} pass={red['marg']>=0.7 and red['null_cross']<0.3 and red['rotated']<0.3} | IV(b) restore {o['restoration_fraction_of_lost_gain']:.3f}")
    pa = A["pi_preserving_abstract"]
    up = pa["cross_minus_within_T"]["ci95"][1]; upP = pa["cross_minus_within_P"]["ci95"][1]
    print(f"VI cross-wT {pa['cross_minus_within_T']['point']:.3f} upper {up:.3f} sep={up<-0.10} shared={up>=-0.10 and upP>=-0.10}")
    print("VI-bayes", {m: round(v, 3) for m, v in A["bayes_tracking_r"].items()}, "all>=0.5:", all(v >= 0.5 for v in A["bayes_tracking_r"].values()))
S = L("logs/site/site_eval.json"); print("===== site")
ok = 0
for m in ["S5", "S6", "S7"]:
    src, rd = S[m]["source|T_full"], S[m]["readout|P_full"]; p = src["adjusted"] >= 0.5 and src["adjusted"] >= rd["adjusted"] + 0.2; ok += p
    print(f"V {m} source adj {src['adjusted']:.3f} (ceil-floor {src['ceiling']-src['floor']:.3f}, n_inf {src['n_inf']}) readout adj {rd['adjusted']:.3f} (ceil-floor {rd['ceiling']-rd['floor']:.3f}) pass={p}")
    print(f"   raw IIA source {src['iia']:.3f} readout {rd['iia']:.3f}; raw iia-floor: source {src['iia']-src['floor']:.3f} readout {rd['iia']-rd['floor']:.3f}")
for m in ["P2", "P3", "P4"]:
    src, rd = S[m]["source|T_full"], S[m]["readout|P_full"]; d = rd["adjusted"] - src["adjusted"]
    print(f"V {m} readout {rd['adjusted']:.3f} source {src['adjusted']} pass={d>=0.2} raw readout-floor {rd['iia']-rd['floor']:.3f} source-floor {src['iia']-src['floor']:.3f}")
print("V S-part passes:", ok, "of 3")
