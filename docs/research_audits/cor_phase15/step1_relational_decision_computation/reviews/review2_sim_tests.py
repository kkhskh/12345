"""Sandbox simulation of the frozen merge + scorer on synthetic confirmation outputs built from real dev outputs (read-only on the package)."""
import json, os, shutil, subprocess, sys, copy, math
PK = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation"
S = "/tmp/review_integrity/sim2"
AN = ("l2_scan", "l2_mix_causal", "l2_mediation", "score_contrast", "l1abl_mix", "decoy_disc")
dev = {a: json.load(open(f"{PK}/logs/{a}_pre.json")) for a in AN}
ad = {k: json.load(open(f"{PK}/logs/adapt_sel_pre/{k}.json")) for k in ("xP81_full_c1_s4000_seed0", "vP150_full_c1_s4000_seed10_eval_conf12")}
MIX = "xP81:full_c1:0:"; PRE = "vP150:full_c1:10:_eval_conf12"
P = ["P160", "P161", "P162", "P163", "P164", "P165"]
def setup(mod=None, complete=None, pre_units=()):
    shutil.rmtree(S, ignore_errors=True); os.makedirs(f"{S}/scripts"); os.makedirs(f"{S}/logs/adapt_sel_conf")
    for f in ("score_xiii.py", "stats_xii.py"): shutil.copy(f"{PK}/scripts/{f}", f"{S}/scripts/")
    shutil.copy(f"{PK}/logs/predictions_xiii.json", f"{S}/logs/")
    for p in P:
        sp = f"u{p}:full_c1:20:_eval_conf13"; tag = f"u{p}_full_c1_s4000_seed20_eval_conf13"; tmpl = PRE if p in pre_units else MIX
        os.makedirs(f"{S}/checkpoints/z{p}"); json.dump(dict(complete=(complete or {}).get(p, True)), open(f"{S}/checkpoints/z{p}/gate.json", "w"))
        for a in AN:
            src = dev[a].get(tmpl, dev[a].get(MIX)) if a != "decoy_disc" else dev[a].get(tmpl, dev[a][MIX])
            v = copy.deepcopy(src)
            if mod: v = mod(p, a, v)
            o = {sp: v} if v is not None else {}
            if a == "decoy_disc": o.update(n_pairs=2000)
            json.dump(o, open(f"{S}/logs/{a}_conf_{p}.json", "w"))
        a_ = copy.deepcopy(ad["vP150_full_c1_s4000_seed10_eval_conf12" if p in pre_units else "xP81_full_c1_s4000_seed0"])
        if mod: a_ = mod(p, "adapt", a_)
        json.dump(a_, open(f"{S}/logs/adapt_sel_conf/{tag}.json", "w"))
    # the merge step of conf_merge.sh, verbatim python
    merge = open(f"{PK}/scripts/conf_merge.sh").read().split("<<'PY'\n")[1].split("\nPY")[0]
    subprocess.run([sys.executable, "-c", merge], cwd=S, check=True)
    r = subprocess.run([sys.executable, f"{S}/scripts/score_xiii.py"], cwd=S, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-1500:]); return None
    return json.load(open(f"{S}/logs/score_xiii.json"))
def show(name, o):
    print("==", name)
    if o is None: print("   CRASH"); return
    print("   counts", o["counts"], "| mixture", sum(bool(v) for v in o["mixture_class"].values()), "| complete", len(o["complete"]))
    for pid, p in o["predictions"].items(): print("  ", pid, p["verdict"], {a: (x["status"], x["k"], x["n"]) for a, x in p["arms"].items()})
show("S1 all six = xP81 dev outputs", setup())
def m2(p, a, v):
    if p == "P163" and a == "l2_mix_causal": return dict(mixture_heads=[], mass=[0.2, 0.79, 0.2, 0.2])
    return v
show("S2 P163 mixture-class by l2_scan but l2_mix_causal found no head (mass 0.79 on its 500-episode sample)", setup(m2))
def m3(p, a, v):
    if p == "P162" and a == "l2_scan": return None
    return v
show("S3 l2_scan output missing for P162 (key absent)", setup(m3))
show("S4 two pre-assembled (vP150-like) complete parents", setup(pre_units=("P164", "P165")))
def m5(p, a, v):
    if p == "P160" and a == "l1abl_mix": v["layer1_mean_ablated"]["B"] = float("nan")
    return v
show("S5 NaN in l1abl layer1_mean_ablated.B for P160", setup(m5))
show("S6 only one complete parent", setup(complete={p: p == "P160" for p in P}))
def m7(p, a, v):
    if p == "P160" and a == "l2_mix_causal": v["donor"]["bayes_resp"]["R2"] = 0.1          # fail arm a
    if p == "P161" and a == "l2_mix_causal": del v["donor_EneA1"]                              # missing arm c
    return v
show("S7 P1: one fail on arm a + one missing on arm c", setup(m7))
def m8(p, a, v):
    if a == "l2_mix_causal": v["donor"]["shuffled_pred"] = None
    return v
show("S8 null quantity missing everywhere (P1a null)", setup(m8))
o = setup(m8); print("   P1a diagnostic flag with null missing:", o["predictions"]["P1"]["arms"]["a"]["diagnostic"])
