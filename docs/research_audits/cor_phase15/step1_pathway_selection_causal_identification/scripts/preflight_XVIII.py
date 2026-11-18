"""XVIII preflight of score_XVIII.py (final form, after the pre-review addenda) on synthetic measurements with known answers and refusal cases.
Scenarios: each designed so that exactly one hypothesis is CONSISTENT.  Symmetry check (pre-review A addendum, A2): the same perturbation
(2 of 6 gsg pairs turned to P) contradicts a fixed-band hypothesis and a mediator hypothesis alike.  Writes logs/preflight_XVIII.json."""
import json, os, sys, subprocess, tempfile, copy
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, ".."); DES = os.path.join(P, "logs", "XVIII_design.json"); d = json.load(open(DES))
def fin(nE, nM, cls, KL=0.02, acc=0.85): return dict(nec_E=nE, nec_M=nM, KL=KL, acc=acc, cls2=cls, cls2_struct=cls if cls != "HYBRID" else "HYBRID", cls2_causal=cls if cls != "HYBRID" else "HYBRID", mix_mass=0.5, lik_RO1=0.5, rhoL=0.6, lapseL=0.01)
EARLY = (1.2, 0.005, "EARLY"); MIX = (0.01, 0.75, "MIXTURE"); HYB = (0.15, 0.30, "HYBRID")
def state(base, band): b, k = (EARLY, MIX) if base == "E" else (MIX, EARLY); return {"N": b, "F": k, "P": HYB}[band]
SLOG = dict(shares=[[i, [0.15] * 4, 1.0] for i in range(100)]); WL = [[i, 1.0, 1.02] + [0.1] * 5 for i in range(100)]; WLS = [[i, 1.0, 1.02] + [0.1] * 5 + [0.0] * 4 + [0.15] * 4 for i in range(100)]
def rec(spec, band, share):
    s = state(spec["base"], band); r = dict(run=spec["tag"], parent=d["parent"], stream=d["stream"], final={"1000": fin(*s), "1211": fin(*s)}, traj={"100": {"own0_mean": share, "own2_mean": share}})
    if spec["arm"] == "baseline": r["slog"] = SLOG
    else: r["window_log"] = WLS if spec["arm"].startswith("sha") else WL
    return r
def scenario(spec_bands, perturb=None):
    out = []
    for key, spec in d["runs"].items():
        arm = spec["arm"]
        if arm == "baseline": out.append(rec(spec, "N", 0.24))
        elif arm == "block": out.append(rec(spec, "F", 0.13))
        else:
            bnd, sh = spec_bands[arm]
            if perturb and arm == perturb[0] and key.split("|")[0] + "|" + key.split("|")[1] in perturb[1]: bnd = perturb[2]
            out.append(rec(spec, bnd, sh))
    return out
# latent shares: 0.13 -> phi_share 1 (F); 0.24 -> 0 (N); 0.185/0.2 -> P
SC = {"H1d": dict(vrg=("F", 0.13), gsg=("N", 0.23), gsv=("N", 0.235), sha_hi_rolled=("F", 0.24), sha_lo_real=("N", 0.24), gof23=("F", 0.13)),
      "Hm_window": dict(vrg=("F", 0.13), gsg=("F", 0.13), gsv=("P", 0.185), sha_hi_rolled=("N", 0.13), sha_lo_real=("F", 0.24), gof23=("F", 0.13)),
      "Hm_release": dict(vrg=("F", 0.13), gsg=("F", 0.13), gsv=("P", 0.185), sha_hi_rolled=("F", 0.13), sha_lo_real=("N", 0.24), gof23=("F", 0.13)),
      "H1d_plus_Hm": dict(vrg=("F", 0.13), gsg=("P", 0.2), gsv=("P", 0.2), sha_hi_rolled=("P", 0.24), sha_lo_real=("P", 0.24), gof23=("P", 0.2)),
      "H1d_plus_H2g": dict(vrg=("F", 0.13), gsg=("P", 0.2), gsv=("F", 0.2), sha_hi_rolled=("F", 0.24), sha_lo_real=("N", 0.24), gof23=("F", 0.13))}
H3 = {f"{b}|{D}": dict(vrg=0.12, sha_hi_rolled=0.12) for b in d["bases"] for D in d["orders"]}
def run_scorer(recs, h3=True):
    t = tempfile.mkdtemp(); m = os.path.join(t, "m.jsonl"); o = os.path.join(t, "o.json"); open(m, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    if h3: json.dump(H3, open(os.path.join(t, "h3_immediate_c18.json"), "w"))
    p = subprocess.run([sys.executable, os.path.join(H, "score_XVIII.py"), DES, m, o], capture_output=True, text=True); return p.returncode, (json.load(open(o)) if os.path.exists(o) else None)
res = dict(scenarios={}, symmetry={}, refusals={}); ok = True
for name, sb in SC.items():
    rc, o = run_scorer(scenario(sb)); v = {h: x["verdict"] for h, x in o["hypotheses"].items()}
    good = v[name] == "CONSISTENT" and sum(x == "CONSISTENT" for x in v.values()) == 1; res["scenarios"][name] = dict(verdicts=v, pass_=good, h3_E=o["h3_E"]["excluded"]); ok &= good; print(name, v, good)
pert = ("gsg", ["E|811", "M|812"], "P")
rc, o1 = run_scorer(scenario(SC["H1d"], pert)); rc, o2 = run_scorer(scenario(SC["Hm_window"], pert))
s1, s2 = o1["hypotheses"]["H1d"]["verdict"], o2["hypotheses"]["Hm_window"]["verdict"]; good = s1 == s2 == "CONTRADICTED"
res["symmetry"] = dict(H1d_after_2_gsg_P=s1, Hm_window_after_2_gsg_P=s2, pass_=good); ok &= good; print("symmetry", s1, s2, good)
base = scenario(SC["H1d"])
cases = {"missing_run": (base[1:], True), "duplicate": (base + [base[0]], True), "wrong_stream": ([dict(base[0], stream=1)] + base[1:], True),
         "missing_field": ([dict(base[0], final={"1000": base[0]["final"]["1000"], "1211": {k: v for k, v in base[0]["final"]["1211"].items() if k != "nec_M"}})] + base[1:], True),
         "missing_step": ([dict(base[0], final={"1211": base[0]["final"]["1211"]})] + base[1:], True), "missing_h3": (base, False)}
for name, (recs, h3) in cases.items():
    rc, o = run_scorer(copy.deepcopy(recs), h3); good = rc == 2 and o is None; res["refusals"][name] = dict(rc=rc, refused=good); ok &= good; print("refusal", name, rc, good)
res["all_pass"] = ok; json.dump(res, open(os.path.join(P, "logs", "preflight_XVIII.json"), "w"), indent=1); print("ALL_PASS", ok)
