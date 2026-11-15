"""Map DEVELOPMENT outputs (XIII seed-20 models) into the file layout score_13b.py reads, under SFX _dev13b, so the frozen scorer can be
preflighted on development data; and build NULL units (SFX _null13b) whose values follow from null scenarios.  Writes logs/units_dev13b.txt
and logs/units_null13b.txt.  The seed-20 specs stand in for the seed-21 units (the scorer only needs the spec strings)."""
import json, os, glob, shutil, re
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
jl = lambda p: json.load(open(os.path.join(L, p))) if os.path.exists(os.path.join(L, p)) else {}
P = ["P160", "P161", "P163", "P164", "P165"]; sp = lambda p: f"u{p}:full_c1:20:_eval_conf13"; tag = lambda p: f"u{p}_full_c1_s4000_seed20_eval_conf13"
S = "_dev13b"; os.makedirs(os.path.join(L, f"das_bal{S}"), exist_ok=True); os.makedirs(os.path.join(L, f"adapt2{S}"), exist_ok=True)
cf = jl("class_features_dev.json"); ff = jl("fit_family_dev.json"); lc = {**jl("l2_components_dev.json"), **{k: v for p in P for k, v in jl(f"l2_components_dev_{p}.json").items()}}
for p in P:
    for nm, d in (("class_features", cf), ("fit_family", ff), ("l2_components", lc)):
        if sp(p) in d: json.dump({sp(p): d[sp(p)]}, open(os.path.join(L, f"{nm}{S}_{p}.json"), "w"))
    f = os.path.join(L, "das_bal_dev", tag(p) + ".json")
    if os.path.exists(f): shutil.copy(f, os.path.join(L, f"das_bal{S}", tag(p) + ".json"))
    a = {}
    for g in [os.path.join(L, "adapt2", tag(p) + "_b0.6.json"), os.path.join(L, "adapt2_dev2", tag(p) + "_b0.6.json"), os.path.join(L, "adapt2_p160early", tag(p) + "_b0.6.json")]:
        if os.path.exists(g): a.update({k: v for k, v in json.load(open(g)).items() if k not in a or k in ("early01", "early01ro", "full")})
    if p == "P160":   # full/attn/... of the OOM-killed run are only in its printed log (logs/run/8acd55afd4ff.out); parse them
        for line in open(os.path.join(L, "run", "8acd55afd4ff.out")):
            m = re.match(r"\s+(readout|attn|mlp|full|fresh)\s+(\{.*\})\s+\d+ s", line)
            if m: a.setdefault(m.group(1), eval(m.group(2)))
            if line.startswith("uP160_full") and "bayes" in line: a.setdefault("bayes_acc", float(line.split("bayes ")[1].split()[0])); a.setdefault("native", eval(line.split("native ")[1].split(" Jpairs")[0]))
    if a: json.dump(a, open(os.path.join(L, f"adapt2{S}", tag(p) + "_b0.6.json"), "w"))
ad = jl("algdisc_eval_conf13.json")
if ad: json.dump(ad, open(os.path.join(L, f"algdisc{S}.json"), "w"))
open(os.path.join(L, "units_dev13b.txt"), "w").write("\n".join(sp(p) for p in P) + "\n")
# ---------------- null units
N = "_null13b"; os.makedirs(os.path.join(L, f"das_bal{N}"), exist_ok=True); os.makedirs(os.path.join(L, f"adapt2{N}"), exist_ok=True)
fz = lambda R2=0.0, slope=0.0: dict(R2=R2, slope=slope)
def nunit(kind):
    """NOOP: no intervention changes anything; UNIFORM: uniform output; POE: model implementing log-linear pooling; A1: model implementing the
    perfect-only rule; SEEDSWITCH: replicate whose class switched; DECOYALL: every row-swap equally effective."""
    cfv = dict(lik_RO1=0.85 if kind == "SEEDSWITCH" else (0.0 if kind == "UNIFORM" else 0.02), mix_mass=0.2 if kind == "SEEDSWITCH" else (0.0 if kind == "UNIFORM" else 0.95), transfer=0.0 if kind in ("NOOP", "UNIFORM") else 0.9,
               route_Q=0.0 if kind in ("NOOP", "UNIFORM") else 0.6, route_R=0.0 if kind in ("NOOP", "UNIFORM") else 0.3)
    ffv = dict(MIX=dict(R2_test=0.84 if kind == "POE" else (0.0 if kind == "UNIFORM" else 0.99)), POE=dict(R2_test=0.99 if kind == "POE" else (0.0 if kind == "UNIFORM" else 0.83)))
    adv = {f: dict(B=1.1 if kind == "A1" else (1.0 if kind == "UNIFORM" else 0.15), A1=0.1 if kind == "A1" else (1.0 if kind == "UNIFORM" else 1.1)) for f in ("D1", "D2")}
    z = 0.0 if kind in ("NOOP", "UNIFORM") else None
    dec = lambda mixp, others: {"MIX": dict(proj=mixp, fid_E=fz(0.9 if mixp else 0.0), fid_A1=fz(0.9 if kind == "A1" else 0.02)), **{k: dict(proj=others, fid_E=fz(0.9 if kind == "DECOYALL" else 0.0), fid_A1=fz(0.0)) for k in ("LOW", "TOP", "MAG", "PERM", "MEAN")}}
    mixp = 0.0 if z is not None else 0.9; oth = 0.85 if kind == "DECOYALL" else 0.0
    rest = {"REMOVED_only": dict(proj=1.0 if z is not None else 0.3, fid_E=fz(), fid_A1=fz()), "removal_fraction_removed": 0.0 if z is not None else 0.7,
            **{k: dict(proj=(1.0 if z is not None else (0.9 if kind == "DECOYALL" else 0.3)), fid_E=fz(0.9 if kind == "DECOYALL" else 0.0), fid_A1=fz()) for k in ("RESTORE_LOW", "RESTORE_TOP", "RESTORE_MAG", "RESTORE_PERM", "RESTORE_MEAN")},
            "RESTORE_MIX": dict(proj=1.0 if z is not None else 0.95, fid_E=fz(0.0 if z is not None else 0.95), fid_A1=fz(0.95 if kind == "A1" else 0.02))}
    lcv = dict(mixture_heads=[1], EA=dict(shapley=dict(R=0.0 if z is not None else 0.7), decoys=dec(mixp, oth), restoration=rest, preservation_likelihood_edit=dict(fid_rr_to_intact=fz(1.0, 1.0))),
               P0=dict(decoys={"MIX": dict(fid_E=fz(0.02 if kind == "A1" else (0.0 if z is not None else 0.8)), fid_A1=fz(0.9 if kind == "A1" else 0.02))}, restoration={"RESTORE_MIX": dict(fid_E=fz(0.02 if kind == "A1" else (0.0 if z is not None else 0.8)), fid_A1=fz(0.9 if kind == "A1" else 0.02))}))
    dbv = {k: dict(fid_E=fz(0.0 if (z is not None or kind == "A1") else (0.7 if k == "trained_E" else 0.0)), fid_A1=fz(0.9 if kind == "A1" and k == "trained_A1" else (0.0 if z is not None else 0.7))) for k in ("trained_E", "trained_A1", "trained_RANDE", "sourcefree_E", "sourcefree_A1", "sourcefree_RANDE")}
    a2v = dict(bayes_acc=0.79, native=dict(acc=0.45, Jpair=fz()), full=dict(acc=0.45 if z is not None else 0.72, Jpair=fz()), early01=dict(acc=0.45 if z is not None else 0.72, Jpair=fz()))
    return cfv, ffv, adv, lcv, dbv, a2v
kinds = ["NOOP", "UNIFORM", "POE", "A1", "SEEDSWITCH", "DECOYALL"]; units = []; ad_n = dict(models={})
for i, k in enumerate(kinds):
    par = ["P160", "P161", "P163", "P164", "P165", "P160"][i]; u = f"u{par}:NULL{k}:21:_x"; units.append(u); cfv, ffv, adv, lcv, dbv, a2v = nunit(k)
    for nm, v in (("class_features", cfv), ("fit_family", ffv), ("l2_components", lcv)):
        fn = os.path.join(L, f"{nm}{N}_{par}.json"); d = json.load(open(fn)) if os.path.exists(fn) else {}; d[u] = v; json.dump(d, open(fn, "w"))
    t = f"u{par}_NULL{k}_s4000_seed21_x"; json.dump(dbv, open(os.path.join(L, f"das_bal{N}", t + ".json"), "w")); json.dump(a2v, open(os.path.join(L, f"adapt2{N}", t + "_b0.6.json"), "w")); ad_n["models"][u] = adv
json.dump(ad_n, open(os.path.join(L, f"algdisc{N}.json"), "w")); open(os.path.join(L, "units_null13b.txt"), "w").write("\n".join(units) + "\n"); print("dev + null units written")
