"""FROZEN scorer for Investigation XII predictions P1-P8 on the fresh cohort (FRESH_CONFIRMATION_AND_STATISTICS.md Part I).
Unit: complete fresh parent (aux_acc >= 0.99; Stage-0 rule).  Arm passes iff it holds in >= ceil(0.75 * n_complete) complete parents (or pairs,
for P5).  A prediction is confirmed iff every arm passes.  Incomplete parents are scored identically but reported separately.
Every individual check is listed in logs/score_xii.json (no aggregate count is reported without its exact denominator).
Effect sizes and 95% bootstrap CIs (scripts/stats_xii.py) are computed for the principal quantity of every arm."""
import json, os, glob, math, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, stats_xii as ST
L = os.path.join(HERE, "..", "logs"); CK = os.path.join(HERE, "..", "checkpoints")
J = lambda f: json.load(open(os.path.join(L, f)))
PARENTS = [p for p in ("P150", "P151", "P152", "P153", "P154") if os.path.exists(os.path.join(CK, f"z{p}", "pre.json"))]
complete = {p: json.load(open(os.path.join(CK, f"z{p}", "pre.json")))["eval"]["aux_acc"] >= 0.99 for p in PARENTS}
COMP = [p for p in PARENTS if complete[p]]; INC = [p for p in PARENTS if not complete[p]]
S = "_eval_conf12"; full = lambda p, s=10: f"v{p}:full_c1:{s}:{S}"; c1 = lambda p: f"v{p}:c1:10:{S}"
REP = COMP[:2]                                                                    # same-parent replicates (Stage-0 rule)
need = lambda n: math.ceil(0.75 * n)
out = dict(parents=PARENTS, complete=COMP, incomplete=INC, replicate_parents=REP, checks=[], effects={})
def arm(pred, name, unit_results, group="complete"):
    vals = list(unit_results.values()); n = len(vals); k = sum(bool(v) for v in vals)
    for u, v in unit_results.items(): out["checks"].append(dict(prediction=pred, arm=name, unit=u, group=group, passed=bool(v)))
    return dict(passed=(k >= need(n)) if n else None, k=k, n=n, need=need(n) if n else None)
def safe(f, *a):
    try: return f(*a)
    except Exception as ex: return None
res = {}
def score_group(units, group):
    R = {}
    fa = J("factorial_conf.json"); ab = J("alt_ablation_conf.json"); ci = J("circuit_l1_conf.json"); rs = J("restore_conf.json")
    # P1
    a = {}
    a["a_full_Eratio>=2_CIlo>=1.5"] = {p: (lambda T, C: (ST.ci_ratio(T["per_episode_E_norm"], C["per_episode_E_norm"])["est"] >= 2.0) and (ST.ci_ratio(T["per_episode_E_norm"], C["per_episode_E_norm"])["lo"] >= 1.5))(fa[full(p)]["TRUE"], fa[full(p)]["CTRL"]) for p in units}
    a["b_full_mid0<=0.05_L0share<=0.10"] = {p: fa[full(p)]["TRUE"]["stages"]["mid0"]["rel_I"] <= 0.05 and abs(sum(fa[full(p)]["TRUE"]["share_E"][f"h0.{h}"] for h in range(4)) + fa[full(p)]["TRUE"]["share_E"]["mlp0"]) <= 0.10 for p in units}
    a["c_full_fullspace_ratio>=1.75"] = {p: ST.ci_ratio(fa[full(p)]["TRUE"]["per_episode_full_norm"], fa[full(p)]["CTRL"]["per_episode_full_norm"])["est"] >= 1.75 for p in units}
    a["d_c1_Eratio<=1.75"] = {p: ST.ci_ratio(fa[c1(p)]["TRUE"]["per_episode_E_norm"], fa[c1(p)]["CTRL"]["per_episode_E_norm"])["est"] <= 1.75 for p in units}
    R["P1"] = {k: arm("P1", k, v, group) for k, v in a.items()}
    out["effects"][f"P1_{group}"] = {p: dict(E_ratio=ST.ci_ratio(fa[full(p)]["TRUE"]["per_episode_E_norm"], fa[full(p)]["CTRL"]["per_episode_E_norm"]), full_ratio=ST.ci_ratio(fa[full(p)]["TRUE"]["per_episode_full_norm"], fa[full(p)]["CTRL"]["per_episode_full_norm"]),
                                         c1_E_ratio=ST.ci_ratio(fa[c1(p)]["TRUE"]["per_episode_E_norm"], fa[c1(p)]["CTRL"]["per_episode_E_norm"])) for p in units}
    # P2
    b = lambda p, k: ab[full(p)][k]
    a = {}
    a["a_addcomp_KLe0<KLB_pref>=0.5"] = {p: b(p, "addcomp")["KL_e0"] < b(p, "addcomp")["KL_bayes"] and b(p, "addcomp")["pref_e0"] >= 0.5 for p in units}
    a["b_pathinA_KLe0<KLB_pref>=0.5_CIhi<0"] = {p: b(p, "path_inA")["KL_e0"] < b(p, "path_inA")["KL_bayes"] and b(p, "path_inA")["pref_e0"] >= 0.5 and ST.ci_mean(b(p, "path_inA")["per_episode_KLdiff"])["hi"] < 0 for p in units}
    a["c_pathinP_KLe0<KLB"] = {p: b(p, "path_inP")["KL_e0"] < b(p, "path_inP")["KL_bayes"] for p in units}
    a["d_scrub_match_KLB<=0.05_and_scrub_e0_KLe0<KLB"] = {p: b(p, "scrub_match")["KL_bayes"] <= 0.05 and b(p, "scrub_e0")["KL_e0"] < b(p, "scrub_e0")["KL_bayes"] for p in units}
    a["e_meanL1_nonspecific_KLe0>=0.3"] = {p: b(p, "mean_L1")["KL_e0"] >= 0.3 for p in units}
    R["P2"] = {k: arm("P2", k, v, group) for k, v in a.items()}
    out["effects"][f"P2_{group}"] = {p: {k: dict(KL_bayes=b(p, k)["KL_bayes"], KL_e0=b(p, k)["KL_e0"], pref_e0=b(p, k)["pref_e0"], diff_CI=ST.ci_mean(b(p, k)["per_episode_KLdiff"]))
                                         for k in ("addcomp", "path_inA", "path_inP", "scrub_match", "scrub_e0", "mean_L1")} for p in units}
    # P3
    a = {}
    a["a_queries_irrelevant_KLB<=0.03_keysonly~pathinA"] = {p: b(p, "path_inA_queries_only")["KL_bayes"] <= 0.03 and abs(b(p, "path_inA_keys_only")["KL_e0"] - b(p, "path_inA")["KL_e0"]) <= 0.02 for p in units}
    a["b_subhead_redundancy_patternA_and_valuesA_stay_Bayes_side"] = {p: b(p, "pattern_A")["KL_bayes"] < b(p, "pattern_A")["KL_e0"] and b(p, "values_A")["KL_bayes"] < b(p, "values_A")["KL_e0"] for p in units}
    a["c_MLP1_amplifies_total>=1.5x_direct"] = {p: b(p, "attn_addcomp_mlp_recomputed")["KL_bayes"] >= 1.5 * b(p, "attn_addcomp_mlp_frozen")["KL_bayes"] for p in units}
    a["d_smooth_ANDshare<0.5_LN2in[0.7,1.3]"] = {p: ci[full(p)]["mlp1"]["share_by_AND_neurons"] < 0.5 and 0.7 <= ci[full(p)]["mlp1"]["LN2_frozen_vs_actual_E"] <= 1.3 for p in units}
    R["P3"] = {k: arm("P3", k, v, group) for k, v in a.items()}
    out["effects"][f"P3_{group}"] = {p: dict(attn_share=ci[full(p)]["attn_sum_share"], mlp1=ci[full(p)]["mlp1"], pattern_A=b(p, "pattern_A")["KL_bayes"], values_A=b(p, "values_A")["KL_bayes"],
                                         total=b(p, "attn_addcomp_mlp_recomputed")["KL_bayes"], direct=b(p, "attn_addcomp_mlp_frozen")["KL_bayes"]) for p in units}
    # P4
    a = {}
    a["a_restoreE_KLB<=0.25_and<=0.5xablated"] = {p: rs[full(p)]["restore_E"] <= 0.25 and rs[full(p)]["restore_E"] <= 0.5 * rs[full(p)]["ablated"] for p in units}
    a["b_restore_random_rec<=0.1"] = {p: rs[full(p)]["rec_random"] <= 0.1 for p in units}
    R["P4"] = {k: arm("P4", k, v, group) for k, v in a.items()}
    out["effects"][f"P4_{group}"] = {p: rs[full(p)] for p in units}
    # P6
    di = J("disc_ident_conf.json"); vs = J("vs_linear_conf.json"); a = {}
    a["a_disc_pairs_IIA_E>=0.70_IIA_A1<=0.30"] = {p: di[full(p)]["IIA_E"] >= 0.70 and di[full(p)]["IIA_A1"] <= 0.30 for p in units}
    def vsok(p):
        r = vs[full(p)]["VS"]; mp = min(r[k]["KL"] for k in ("PROJ_1", "PROJ_2", "PROJ_3", "ID")); return r["LIN"]["KL"] <= 0.7 * mp and max(r["LIN_eigs"]) >= 1.2
    a["b_VS_linear_beats_projections_KL<=0.7x_eig>=1.2"] = {p: vsok(p) for p in units}
    R["P6"] = {k: arm("P6", k, v, group) for k, v in a.items()}
    out["effects"][f"P6_{group}"] = {p: dict(disc={k: v for k, v in di[full(p)].items() if not k.startswith("per_")}, disc_IIA_E_CI=ST.ci_frac(di[full(p)]["per_pair_hitE"]), disc_IIA_A1_CI=ST.ci_frac(di[full(p)]["per_pair_hitA1"]),
                                         VS={k: (v if not isinstance(v, dict) else dict(IIA=v["IIA"], KL=v["KL"])) for k, v in vs[full(p)]["VS"].items()}) for p in units}
    # P7
    dm = J("dyn_mech_conf.json"); tr = J("traj_alg_xii.json"); a = {}
    def p7(p):
        d = dm[full(p)]; steps = sorted(int(s) for s in d["traj"]); us, em = d["use_onset_step"], d["representation_emergence_step"]
        rows = tr[full(p)]; sw = next((r["step"] for r in rows if r["D2"]["closest"] == "B"), None); nonA1 = next((r["step"] for r in rows if r["D2"]["closest"] != "A1"), None)
        return us, em, sw, nonA1, steps
    a["a_|idx(use)-idx(emerge)|<=1"] = {}; a["b_use<=first_nonA1"] = {}; a["c_D2B_switch>use"] = {}
    for p in units:
        us, em, sw, nonA1, steps = p7(p); ix = lambda s: steps.index(s) if s in steps else None
        a["a_|idx(use)-idx(emerge)|<=1"][p] = us is not None and em is not None and abs(ix(us) - ix(em)) <= 1
        a["b_use<=first_nonA1"][p] = us is not None and nonA1 is not None and us <= nonA1
        a["c_D2B_switch>use"][p] = us is not None and sw is not None and sw > us
        out["effects"].setdefault(f"P7_{group}", {})[p] = dict(use_onset=us, emergence=em, D2B_switch=sw, first_nonA1=nonA1)
    R["P7"] = {k: arm("P7", k, v, group) for k, v in a.items()}
    # P8
    a = {}; ad = {p: json.load(open(glob.glob(os.path.join(L, "adapt_c2_conf", f"v{p}_full_c1_s4000_seed10{S}.json"))[0])) for p in units}
    a["a_native_acc<=0.40_KL_C1rule<=0.05"] = {p: ad[p]["native"]["acc"] <= 0.40 and ad[p]["native"]["KL_C1rule"] <= 0.05 for p in units}
    a["b_adapt_l2_acc>=bayes-0.03"] = {p: ad[p]["adapt_l2"]["acc"] >= ad[p]["refs"]["bayes_acc"] - 0.03 for p in units}
    a["c_adapt_l2_noE_acc>=bayes-0.03(bypass)"] = {p: ad[p]["adapt_l2_noE"]["acc"] >= ad[p]["refs"]["bayes_acc"] - 0.03 for p in units}
    a["d_adapt_l2_E_dependence>=0.1"] = {p: ad[p]["adapt_l2"]["E_dependence"] >= 0.1 for p in units}
    R["P8"] = {k: arm("P8", k, v, group) for k, v in a.items()}
    out["effects"][f"P8_{group}"] = {p: {k: ad[p][k] for k in ("refs", "native", "adapt_l2", "adapt_l2_noE", "fresh_l2")} for p in units}
    return R
res["complete"] = score_group(COMP, "complete")
if INC: res["incomplete"] = safe(score_group, INC, "incomplete")
# P5: pairs (complete parents only)
def cross(A, B):
    f = glob.glob(os.path.join(L, "cross_conf", f"{A}__{B}.json")); return json.load(open(f[0])) if f else None
def gain(d, k): return (d["mean"]["KL"] - d[k]["KL"]) / (d["mean"]["KL"] - d["within"]["KL"])
tag = lambda p, spec="full_c1", s=10: f"v{p}_{spec}_s4000_seed{s}{S}"
same = {}; crossp = {}; c1f = {}; infl = {}
for p in REP:
    for A, B in ((tag(p, s=10), tag(p, s=11)), (tag(p, s=11), tag(p, s=10))):
        d = cross(A, B); same[f"{A[:5]}s{A.split('seed')[1][:2]}->{B[:5]}s{B.split('seed')[1][:2]}"] = d
for i, p in enumerate(COMP):
    for p2 in COMP[i + 1:]:
        for A, B in ((tag(p), tag(p2)), (tag(p2), tag(p))): crossp[f"{A[:5]}->{B[:5]}"] = cross(A, B)
for p in COMP: c1f[p] = cross(tag(p, "c1"), tag(p))
a = {}
a["a_same_parent_gain>=0.85"] = {k: gain(d, "mapped") >= 0.85 for k, d in same.items() if d}
a["a_cross_parent_gain>=0.85"] = {k: gain(d, "mapped") >= 0.85 for k, d in crossp.items() if d}
gs, gc = [gain(d, "mapped") for d in same.values() if d], [gain(d, "mapped") for d in crossp.values() if d]
a["b_|mean_same-mean_cross|<=0.10"] = {"all": abs(np.mean(gs) - np.mean(gc)) <= 0.10}
pc = J("perm_compare_conf.json"); a["c_head_profile_same<cross(raw)"] = {"all": pc["mean_same_parent"]["raw"] < pc["mean_cross_parent"]["raw"]}
a["d_c1_to_full_gain<=0.85"] = {p: gain(d, "mapped") <= 0.85 for p, d in c1f.items() if d}
a["e_meanpatch_IIAdisc>=base+0.30"] = {k: d["mean"]["IIA_disc"] >= d["base"]["IIA_disc"] + 0.30 for k, d in {**same, **crossp}.items() if d}
res["P5"] = {k: arm("P5", k, v, "pairs") for k, v in a.items()}
out["effects"]["P5"] = dict(same={k: dict(gain=gain(d, "mapped"), shuffled=gain(d, "shuffled"), R2_insample=d["map_insample_R2"], R2_heldout=d["map_heldout_R2"], R2_evalstream=d["map_evalstream_R2"]) for k, d in same.items() if d},
                            cross={k: dict(gain=gain(d, "mapped"), shuffled=gain(d, "shuffled"), R2_insample=d["map_insample_R2"], R2_heldout=d["map_heldout_R2"], R2_evalstream=d["map_evalstream_R2"]) for k, d in crossp.items() if d},
                            c1_to_full={p: gain(d, "mapped") for p, d in c1f.items() if d}, mean_same=float(np.mean(gs)), mean_cross=float(np.mean(gc)), perm_compare={k: pc[k] for k in ("mean_same_parent", "mean_cross_parent")})
verd = {}
for P in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
    arms = res["P5"] if P == "P5" else res["complete"][P]; verd[P] = "confirmed" if all(v["passed"] for v in arms.values()) else "falsified"
out["arms"] = res; out["verdicts"] = verd
ch = [c for c in out["checks"] if c["group"] in ("complete", "pairs")]; out["check_count_complete_and_pairs"] = dict(passed=sum(c["passed"] for c in ch), total=len(ch))
json.dump(out, open(os.path.join(L, "score_xii.json"), "w"), indent=1, default=float)
for P, v in verd.items():
    arms = res["P5"] if P == "P5" else res["complete"][P]; print(P, v, {k: f"{x['k']}/{x['n']}" for k, x in arms.items()})
print("individual checks (complete parents + pairs):", out["check_count_complete_and_pairs"], "| complete:", COMP, "incomplete:", INC)
