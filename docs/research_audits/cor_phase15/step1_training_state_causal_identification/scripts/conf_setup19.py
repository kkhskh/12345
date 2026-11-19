"""XIX confirmation setup.  Usage: conf_setup19.py SPEC.json
SPEC: {"parent", "stream", "Ds": [...], "transplant_arms": [{"name","base":"B"|"I","groups","opt","perturb","alpha"}], "training_arms": ["vsg",...],
       "hypotheses": [...], "min_pairs": int}
Writes:
  logs/conf/queue_c1.txt   step-100 matched states (B, I and every training arm's window state), FX_SAVEOPT at 100
  logs/conf/queue_c2.txt   continuations 100 -> 1211 (plain training): transplant arms via compose19 + branch; training arms as branch
                           continuations of their own step-100 window state (identical computation to an uninterrupted windowed run)
                           each followed by measure19 on the frozen confirmation stream into logs/conf/measure_c19.jsonl;
                           training arms first record the sha256 of the step-100 source state (logs/conf/branch_src/<run>.json)
  logs/conf/design19.json  the scorer's design (pairs, families, arms, run-name templates, provenance expectations, hypotheses)
Per-base window interventions (all for steps < 100, then plain):
  I (block)   E: gof own0 B=1e9          M: gof own2 B=1e9
  vsg/vrs/vrg/gsg on own0 (E) / own2 (M)
  frz         E: L0_QK                   M: L2_QK          (target-layer routing parameters frozen, values and Adam moments)
  ctl         E: msk oq0                 M: gof oq2 B=1.0  (non-targeted, matched in development on step-100 state distance)"""
import os, sys, json, hashlib
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, ".."); S = json.load(open(sys.argv[1])); P = S["parent"]
os.makedirs(os.path.join(R, "logs", "conf", "branch_src"), exist_ok=True)
BASE = dict(E=dict(L=21, T="own0", frz="L0_QK"), M=dict(L=23, T="own2", frz="L2_QK"))
def wenv(base, arm, D=None):
    """the window environment (FX_* keys the scorer checks) of each step-100 state"""
    b = BASE[base]
    if arm == "B": return {}
    if arm == "I": return {"FX_INTERV": "gof:100", "FX_EDGES": b["T"], "FX_B": "1e9"}
    if arm in ("vsg", "vsgo", "vrs", "vrg", "gsg"): return {"FX_INTERV": f"{arm}:100", "FX_EDGES": b["T"]}
    if arm in ("frz", "frzp"): return {"FX_INTERV": f"{arm}:100", "FX_FREEZE": b["frz"]}
    if arm == "pnz": return {"FX_INTERV": "pnz:100", "FX_PNZ_SEED": str(9100 + D)}
    if arm == "lbl": return {"FX_INTERV": "lbl:100", "FX_LBL_P": "0.5", "FX_LBL_SEED": str(9200 + D)}
    raise ValueError(arm)
def win(base, arm, D=None):
    e = wenv(base, arm, D); b = BASE[base]
    if arm == "B": return "", ""
    suf = (f"_gof{b['T']}b1e9T100" if arm == "I" else f"_frz{b['frz']}T100" if arm == "frz" else f"_frzp{b['frz']}T100" if arm == "frzp"
           else f"_pnzs{9100 + D}T100" if arm == "pnz" else f"_lblp0.5s{9200 + D}T100" if arm == "lbl" else f"_{arm}{b['T']}T100")
    return " ".join(f"{k}={v}" for k, v in e.items()), suf
pairs, q1, q2 = [], [], []
M19 = "python3 scripts/measure19.py"; MOUT = "logs/conf/measure_c19.jsonl"
for base in ("E", "M"):
    for D in S["Ds"]:
        L = BASE[base]["L"]; pn = f"{base}{D}"; pre = f"{P}_fx_L{L}_W{L}_D{D}"
        pairs.append(dict(name=pn, base=base, L=L, W=L, D=D, Isuf=win(base, "I")[1], win={a: win(base, a, D)[1] for a in S["training_arms"]}, wenv={a: wenv(base, a, D) for a in ["I"] + S["training_arms"]}))
        sB, sI = f"checkpoints/ft/{pre}_stop100_s100c19", f"checkpoints/ft/{pre}{win(base, 'I')[1]}_stop100_s100c19"
        for arm in ["B", "I"] + S["training_arms"]:
            env, suf = win(base, arm, D); wait = ""
            if arm == "pnz": env += f" FX_PNZ_REF={sB}:{sI}"; wait = f"t0=$(date +%s); until [ -f {sB}/opt_step100.pkl ] && [ -f {sI}/opt_step100.pkl ]; do sleep 15; [ $(( $(date +%s) - t0 )) -gt 2400 ] && exit 7; done; "
            q1.append(f"{wait}{env} FX_NOFALLBACK=1 FX_WLOSS=1 FX_STOP=100 FX_SAVEOPT=1 FX_SAVEOPT_AT=100 FX_TAGSUF=_s100c19 python3 scripts/xix_train.py {P} {L} {L} {D}".replace("  ", " ").strip())
        for a in S["transplant_arms"]:
            if base not in a.get("bases", "EM"): continue
            a = dict(a, perturb=(a["perturb"] + D + (500 if base == "M" else 0)) if a.get("perturb") is not None else None)
            src, oth = (sB, sI) if a["base"] == "B" else (sI, sB); tag = f"tp_{pn}_{a['name']}"; rd = f"checkpoints/ft/{pre}_stop1211_br100_{pn}_{a['name']}"
            comp = f"python3 scripts/compose19.py {src} {oth} {tag} --params-from-I {a['groups']} --opt {a['opt']}" + (f" --perturb {a['perturb']}" if a.get("perturb") is not None else "") + (f" --alpha {a['alpha']}" if a.get("alpha") is not None else "")
            q2.append(f"{comp} && FX_NOFALLBACK=1 FX_BRANCH={tag}:100 FX_STOP=1211 FX_TAGSUF=_{pn}_{a['name']} python3 scripts/xix_train.py {P} {L} {L} {D} && {M19} {rd} {P} {S['stream']} {MOUT}")
        for arm in S["training_arms"]:
            s100 = f"{pre}{win(base, arm, D)[1]}_stop100_s100c19"; rn = f"{pre}_stop1211_br100_{pn}_{arm}"
            q2.append(f"FX_NOFALLBACK=1 FX_BRANCH={s100}:100 FX_STOP=1211 FX_TAGSUF=_{pn}_{arm} python3 scripts/xix_train.py {P} {L} {L} {D} && {M19} checkpoints/ft/{rn} {P} {S['stream']} {MOUT}")
open(os.path.join(R, "logs", "conf", "queue_c1.txt"), "w").write("\n".join(q1) + "\n"); open(os.path.join(R, "logs", "conf", "queue_c2.txt"), "w").write("\n".join(q2) + "\n")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
names = json.load(open(os.path.join(R, "logs", "param_names19.json")))
exec(open(os.path.join(H, "compose19.py")).read().split("def sel")[0].split("names = json.load")[1].split("\n", 1)[1])   # group()
ALLG = sorted({group(n) for n in names})
def expect(spec): return [] if spec == "none" else (ALLG if spec == "all" else sorted({g for g in ALLG for x in spec.split(",") if g == x or g.startswith(x + "_")}))
arms = {}
for a in S["transplant_arms"]:
    pv = a.get("perturb")
    arms[a["name"]] = dict(family="T", run="{parent}_fx_L{L}_W{W}_D{D}_stop1211_br100_{pair}_" + a["name"], bases=a.get("bases", "EM"),
                           transplant=dict(tp="tp_{pair}_" + a["name"], params_from_I=expect(a["groups"]), opt=a["opt"], perturb=pv, alpha=a.get("alpha"),
                                           src_dirs=dict(B="{parent}_fx_L{L}_W{W}_D{D}_stop100_s100c19" if a["base"] == "B" else "{parent}_fx_L{L}_W{W}_D{D}{Isuf}_stop100_s100c19",
                                                         I="{parent}_fx_L{L}_W{W}_D{D}{Isuf}_stop100_s100c19" if a["base"] == "B" else "{parent}_fx_L{L}_W{W}_D{D}_stop100_s100c19")))
for arm in S["training_arms"]:
    arms[arm] = dict(family="T", run="{parent}_fx_L{L}_W{W}_D{D}_stop1211_br100_{pair}_" + arm, transplant=None, branch_from=arm)
DEPM = {os.path.basename(f): sha(f) for f in [os.path.join(R, "src", "xix_score.py"), os.path.join(H, "measure19.py"), os.path.join(R, "..", "step1_pathway_selection_causal_identification", "src", "xviiim.py")] + [os.path.join(R, "..", "step1_cross_family_relational_foundations", "src", f) for f in ("native.py", "ftlib.py", "metrics.py", "maskfwd.py", "relpairs.py", "genlib.py")]}
DEPS = {f: sha(os.path.join(R, f)) for f in ("src/xix_score.py", "scripts/measure19.py", "scripts/score19.py", "scripts/compose19.py", "scripts/xix_train.py", "scripts/stats19.py")}
D = dict(stream=S["stream"], parent=P, score_code_sha256=sha(os.path.join(R, "src", "xix_score.py")), harness_sha256=sha(os.path.join(H, "xix_train.py")), dep_sha256=DEPS, dep_sha256_measure=DEPM, refs_sha256=S["refs_sha256"], min_pairs=S["min_pairs"],
         matching=S.get("matching"), pairs=pairs, families=dict(T=dict(anchors=dict(B="fullB", I="fullI"))), arms=arms, hypotheses=S["hypotheses"], window=BASE)
json.dump(D, open(os.path.join(R, "logs", "conf", "design19.json"), "w"), indent=1)
print(len(q1), "step-100 runs;", len(q2), "continuations;", len(pairs), "pairs")
