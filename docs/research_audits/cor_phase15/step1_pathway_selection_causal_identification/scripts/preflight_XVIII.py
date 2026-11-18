"""XVIII preflight of score_XVIII.py on synthetic measurements with known answers (3 scenarios) and 5 refusal cases.  Writes logs/preflight_XVIII.json."""
import json, os, sys, subprocess, tempfile, copy
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, ".."); d = json.load(open(os.path.join(P, "logs", "XVIII_design.json")))
def fin(nE, nM, cls, KL=0.02, acc=0.85): return dict(nec_E=nE, nec_M=nM, KL=KL, acc=acc, cls2=cls, mix_mass=0.5, lik_RO1=0.5, rhoL=0.6, lapseL=0.01)
EARLY = (1.2, 0.005, "EARLY"); MIX = (0.01, 0.75, "MIXTURE"); HYB = (0.15, 0.30, "HYBRID")
def state(base, band):
    b, k = (EARLY, MIX) if base == "E" else (MIX, EARLY)
    return {"N": b, "F": k, "P": HYB}[band]
def rec(spec, band, share100, wl=None, slog=None):
    s = state(spec["base"], band); r = dict(run=spec["tag"], parent=d["parent"], stream=d["stream"], final={"1000": fin(*s), "1211": fin(*s)},
                                           traj={"100": {"own0_mean": share100, "own2_mean": share100}})
    if wl is not None: r["window_log"] = wl
    if slog is not None: r["slog"] = slog
    return r
SLOG = dict(shares=[[i, [0.15, 0.15, 0.15, 0.15], 1.0] for i in range(100)])
WL = [[i, 1.0, 1.02] + [0.1] * 5 for i in range(100)]
WLS = [[i, 1.0, 1.02] + [0.1] * 5 + [0.0] * 4 + [0.15] * 4 for i in range(100)]
def scenario(bands, med_share):
    out = []
    for key, spec in d["runs"].items():
        arm = spec["arm"]
        if arm == "baseline": out.append(rec(spec, "N", 0.24, slog=SLOG))
        elif arm == "block": out.append(rec(spec, "F", 0.13, wl=WL))
        else: out.append(rec(spec, bands[arm], med_share.get(arm, 0.2), wl=WLS if arm.startswith("sha") else WL))
    return out
SC = {"H1d": (dict(vrg="F", gsg="N", gsv="N", sha_hi_rolled="F", sha_lo_real="N", gof23="F"), dict(gsg=0.23, gsv=0.235), "H1d"),
      "Hm": (dict(vrg="F", gsg="F", gsv="P", sha_hi_rolled="N", sha_lo_real="F", gof23="F"), dict(gsg=0.13, gsv=0.185), "Hm"),
      "additive": (dict(vrg="F", gsg="P", gsv="P", sha_hi_rolled="P", sha_lo_real="P", gof23="P"), dict(gsg=0.2, gsv=0.2), "H1d_plus_Hm")}
res = dict(scenarios={}, refusals={}); ok = True
def run_scorer(recs):
    t = tempfile.mkdtemp(); m = os.path.join(t, "m.jsonl"); o = os.path.join(t, "o.json")
    open(m, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    p = subprocess.run([sys.executable, os.path.join(H, "score_XVIII.py"), os.path.join(P, "logs", "XVIII_design.json"), m, o], capture_output=True, text=True)
    return p.returncode, (json.load(open(o)) if os.path.exists(o) else None)
for name, (bands, ms, expect) in SC.items():
    rc, o = run_scorer(scenario(bands, ms)); v = {h: x["verdict"] for h, x in o["hypotheses"].items()}
    good = v[expect] == "CONSISTENT" and sum(x == "CONSISTENT" for x in v.values()) == 1
    res["scenarios"][name] = dict(rc=rc, verdicts=v, expected_unique_consistent=expect, pass_=good); ok &= good; print(name, v, good)
base = scenario(*SC["H1d"][:2])
cases = {"missing_run": base[1:], "duplicate": base + [base[0]], "wrong_stream": [dict(base[0], stream=1)] + base[1:],
         "missing_field": [dict(base[0], final={"1000": base[0]["final"]["1000"], "1211": {k: v for k, v in base[0]["final"]["1211"].items() if k != "nec_M"}})] + base[1:],
         "missing_step": [dict(base[0], final={"1211": base[0]["final"]["1211"]})] + base[1:]}
for name, recs in cases.items():
    rc, o = run_scorer(copy.deepcopy(recs)); good = rc == 2 and o is None; res["refusals"][name] = dict(rc=rc, refused=good); ok &= good; print("refusal", name, rc, good)
res["all_pass"] = ok; json.dump(res, open(os.path.join(P, "logs", "preflight_XVIII.json"), "w"), indent=1); print("ALL_PASS", ok)
