"""XVI frozen scorer (FAIL-CLOSED).  Usage: score_XVI.py [PKG=package dir] [OUT=logs/score_XVI.json]
Refuses (exit 2, writes nothing) unless: every file in logs/FREEZE_XVI_HASHES.txt matches; logs/xvi_math_verified.json has all_pass;
every design run has done.json matching the design (cond, seed, sched, steps, diag_every, init path), blind = true, the init hash
recomputed (train_tasks.init for fresh runs, the XV snapshot for continuations); traj.jsonl holds exactly one record per diagnostic step
0, DE, 2 DE, ..., STEPS with every required field finite; every 500-step snapshot and the final snapshot exist.

Per model (frozen definitions; diagnostics from src/xvidiag.py; KL_oa, KL_o1 = exact KL of the order-a and order-1 optima on the
diagnostic set, A_ref = A_conf of the order-a optimum):
  t_first  first diagnostic step with F_G >= 0.2 (first measurable gate response).
  t_A      SUSTAINED ACQUISITION: first step t such that t and the next two diagnostic points all have F_G >= 0.5 and KL <= 0.75 KL_oa.
           No such t: right-censored at STEPS (never assigned an acquisition time).
  E_A      cumulative learning-rate exposure at t_A (sum of applied learning rates, units of steps x lr); for continuations the XV
           cosine run's exact total exposure is added (E_total).
  width    steps from the last point with F_G <= 0.1 before t_A to the first point with F_G >= 0.9 at or after t_A (None if never).
  regressions  number of diagnostic points after t_A with F_G < 0.5.
  near_gatefree_steps  diagnostic interval x number of points with F_G < 0.2 and |KL - KL_oa| <= 0.1 KL_oa.
  competent_end  mean of the last 5 points: KL <= KL_COMP and F_G >= 0.9 (KL_COMP = 0.03 for constant-rate runs, 0.02 for cosine).
  stage    floor-conditioned intermediate stage (mandate D2): >= 4 consecutive points before t_A with F_G <= 0.1, A_conf >= 0.5 A_ref and
           KL <= KL_o1 - 0.5 (KL_o1 - KL_oa) (at least half the agreement benefit realised).  Evaluated only for models with t_A.
Frozen predictions and verdict rules:
  P1 (Workstream A, fresh constant rate, 4 independent models, 15000 steps): k = number with t_A.
     BROAD (k >= 3)  |  SELECTIVE (k = 2)  |  RARE (k <= 1).            Frozen expectation: BROAD.
  P2 (Workstream A, 6 learning-rate-restart continuations of the 6 censored XV arity-3 models, 4000 steps): kc = number with t_A.
     CONT_BROAD (kc >= 4)  |  CONT_SELECTIVE (kc = 2, 3)  |  CONT_RARE (kc <= 1).   Frozen expectation: CONT_BROAD.
     Secondary (frozen descriptors): FAST_UNLOCK if kc >= 2 and the median post-restart t_A <= 1000 steps; matched-exposure comparison at
     E* = the smallest E_total reached by any continuation: kf* = fresh models with E_A <= E*; COMPARABLE if |kc*/6 - kf*/4| <= 0.25
     (kc* = continuations with E_total(t_A) <= E*), else CONT_LOWER / CONT_HIGHER.  (Continuations are selected XV failures: a difference
     is not attributable to the schedule alone.)
  P3 (Workstream B, value at arity 2 with the agreement law identical; 4 seed pairs 501-504): r = ln(t_A(low) / t_A(high)), a censored
     t_A entering as 4000 (a lower bound).  BENEFIT_SPEEDS: r >= ln 1.25 in >= 3/4 pairs;  BENEFIT_SLOWS: r <= -ln 1.25 in >= 3/4;
     NO_EFFECT: |r| < ln 1.25 in >= 3/4;  otherwise INCONCLUSIVE.   Frozen expectation: BENEFIT_SPEEDS.
  P5 (stage validity, among acquirers only): per cohort (WA_fresh_const, WB_value_low, WB_value_high) PRESENT if >= 75% of acquirers show
     the stage (>= 2 acquirers), ABSENT if <= 25%, MIXED otherwise, NOT_EVALUABLE if < 2 acquirers.  STAGE_ROBUST if every evaluable cohort is
     PRESENT and >= 2 are evaluable; STAGE_NOT_ROBUST if any is ABSENT; otherwise MIXED.   Frozen expectation: STAGE_ROBUST.
Statistics reported (not verdicts): Kaplan-Meier acquisition curves by step and by exposure, exact Clopper-Pearson 95% intervals for
acquisition proportions, Fisher exact test fresh vs continuation at matched exposure, threshold sensitivity of t_A and the stage."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."); OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, "logs", "score_XVI.json")
def refuse(m): print("REFUSED:", m); sys.exit(2)
H = os.path.join(PKG, "logs", "FREEZE_XVI_HASHES.txt")
if not os.path.exists(H): refuse("freeze hash list missing")
for l in open(H):
    h, p = l.split()
    fp = os.path.join(PKG, p)
    if not os.path.exists(fp) or hashlib.sha256(open(fp, "rb").read()).hexdigest() != h: refuse(f"frozen file changed or missing: {p}")
V = os.path.join(PKG, "logs", "xvi_math_verified.json")
if not os.path.exists(V) or not json.load(open(V)).get("all_pass"): refuse("mathematical verification missing or failed")
D = json.load(open(os.path.join(PKG, "logs", "XVI_design.json")))
NEED = ("step", "KL", "F_G", "A_conf", "cum_lr", "G", "A_p", "conf", "sB")
TR = {}
for r in D["runs"]:
    d = os.path.join(PKG, r["dir"]); f = os.path.join(d, "done.json")
    if not os.path.exists(f): refuse(f"run not finished: {r['dir']}")
    j = json.load(open(f))
    if (j.get("cond"), j.get("seed"), j.get("sched"), j.get("steps"), j.get("diag_every"), j.get("init")) != (r["cond"], r["seed"], r["sched"], r["steps"], r["diag_every"], r["init"]): refuse(f"config mismatch: {r['dir']}")
    if j.get("blind") is not True: refuse(f"not blind: {r['dir']}")
    if os.path.exists(os.path.join(d, "resume.pkl")): refuse(f"resume state left (incomplete?): {r['dir']}")
    snaps = list(range(500, r["steps"] + 1, 500)) + [r["steps"]]
    if [k for k in snaps if not os.path.exists(os.path.join(d, f"step{k}.npz"))]: refuse(f"snapshots missing: {r['dir']}")
    t = os.path.join(d, "traj.jsonl")
    if not os.path.exists(t): refuse(f"trajectory missing: {r['dir']}")
    L = [json.loads(x) for x in open(t)]
    if [x["step"] for x in L] != list(range(0, r["steps"] + 1, r["diag_every"])): refuse(f"trajectory steps mismatch: {r['dir']}")
    for x in L:
        for k in NEED:
            if k == "A_conf" and x.get("sB", 0) <= 0.05: continue
            if x.get(k) is None or x[k] != x[k]: refuse(f"trajectory field missing or NaN ({k} at step {x.get('step')}): {r['dir']}")
    TR[r["dir"]] = L
sys.path.insert(0, os.path.join(PKG, "..", "step1_foundational_atlas_continuation", "src")); sys.path.insert(0, os.path.join(PKG, "..", "step1_cross_family_relational_foundations", "src")); sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, optax
from scipy.stats import beta as Beta, fisher_exact
import train_tasks as TT, gatelib as GL, xvidiag as XD
hsh = lambda p: hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(p))).hexdigest()
for r in D["runs"]:
    want = hsh(TT.init("gate", r["seed"])) if r["init"] is None else hsh(GL.load(os.path.normpath(os.path.join(PKG, r["init"]))))
    if json.load(open(os.path.join(PKG, r["dir"], "done.json")))["init_sha256"] != want: refuse(f"init hash mismatch: {r['dir']}")
xv_sched = optax.warmup_cosine_decay_schedule(0.0, 1e-3, 200, 4000); XV_EXPOSURE = float(sum(float(xv_sched(i)) for i in range(4000)))
REF = {}
for c in sorted({r["cond"] for r in D["runs"]}):
    R = XD.Diag(c, D["diag_stream"], D["diag_n"], os.path.join(PKG, "logs")).ref_scores(); REF[c] = dict(KL_oa=R["ordera"]["KL"], KL_o1=R["order1"]["KL"], A_ref=R["ordera"]["A_conf"])
def model(r, L, fg=0.5, klf=0.75, st_n=4, st_a=0.5, st_k=0.5):
    ref = REF[r["cond"]]; n = len(L); de = r["diag_every"]; KLoa, KLo1, Aref = ref["KL_oa"], ref["KL_o1"], ref["A_ref"]
    acq = lambda x: x["F_G"] >= fg and x["KL"] <= klf * KLoa
    iA = next((i for i in range(n - 2) if acq(L[i]) and acq(L[i + 1]) and acq(L[i + 2])), None)
    tA = L[iA]["step"] if iA is not None else None; pre = XV_EXPOSURE if r["sched"] == "cont" else 0.0
    out = dict(t_first=next((x["step"] for x in L if x["F_G"] >= 0.2), None), t_A=tA, E_A=(L[iA]["cum_lr"] if iA is not None else None),
               E_total_A=(L[iA]["cum_lr"] + pre if iA is not None else None), E_total_end=L[-1]["cum_lr"] + pre, censored=tA is None, steps=r["steps"])
    if iA is not None:
        lo = max([L[j]["step"] for j in range(iA + 1) if L[j]["F_G"] <= 0.1], default=0); hi = next((L[j]["step"] for j in range(iA, n) if L[j]["F_G"] >= 0.9), None)
        out["width"] = (hi - lo) if hi is not None else None; out["regressions"] = sum(x["F_G"] < 0.5 for x in L[iA + 1:])
        ok = [x["F_G"] <= 0.1 and x.get("A_conf") is not None and x["A_conf"] >= st_a * Aref and x["KL"] <= KLo1 - st_k * (KLo1 - KLoa) for x in L[:iA]]
        run = best = 0
        for v in ok: run = run + 1 if v else 0; best = max(best, run)
        out["stage"] = best >= st_n; out["stage_longest_run_points"] = best
    else: out.update(width=None, regressions=None, stage=None, stage_longest_run_points=None)
    out["near_gatefree_steps"] = de * sum(x["F_G"] < 0.2 and abs(x["KL"] - KLoa) <= 0.1 * KLoa for x in L)
    last = L[-5:]; kc = 0.03 if r["sched"] in ("const", "cont") else 0.02
    out["final_mean"] = dict(KL=float(np.mean([x["KL"] for x in last])), F_G=float(np.mean([x["F_G"] for x in last])), G=float(np.mean([x["G"] for x in last])))
    out["competent_end"] = bool(out["final_mean"]["KL"] <= kc and out["final_mean"]["F_G"] >= 0.9)
    return out
M = {r["dir"]: model(r, TR[r["dir"]]) for r in D["runs"]}
coh = lambda c: [r for r in D["runs"] if r["cohort"] == c]
cp = lambda k, n: [float(Beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(Beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]
def km(ms, key):
    ev = sorted((m[key], 1) if not m["censored"] else (m["E_total_end"] if key.startswith("E") else m["steps"], 0) for m in ms); S, n, curve = 1.0, len(ev), []
    for i, (t, e) in enumerate(ev):
        if e: S *= 1 - 1 / (n - i); curve.append((t, 1 - S))
    return curve
out = dict(models=M, refs=REF, xv_cosine_total_exposure=XV_EXPOSURE, verdicts={}, stats={})
F = [M[r["dir"]] for r in coh("WA_fresh_const")]; k = sum(not m["censored"] for m in F)
out["verdicts"]["P1"] = "BROAD" if k >= 3 else ("SELECTIVE" if k == 2 else "RARE")
out["stats"]["P1"] = dict(acquired=k, n=len(F), cp95=cp(k, len(F)), km_steps=km(F, "t_A"), km_exposure=km(F, "E_total_A"), t_A=[m["t_A"] for m in F], E_A=[m["E_A"] for m in F])
C = [M[r["dir"]] for r in coh("WA_continuation_restart")]; kc = sum(not m["censored"] for m in C)
out["verdicts"]["P2"] = "CONT_BROAD" if kc >= 4 else ("CONT_SELECTIVE" if kc >= 2 else "CONT_RARE")
Es = min(m["E_total_end"] for m in C); kcs = sum((not m["censored"]) and m["E_total_A"] <= Es for m in C); kfs = sum((not m["censored"]) and m["E_A"] <= Es for m in F)
diff = kcs / len(C) - kfs / len(F)
out["stats"]["P2"] = dict(acquired=kc, n=len(C), cp95=cp(kc, len(C)), t_A_post_restart=[m["t_A"] for m in C], E_total_A=[m["E_total_A"] for m in C],
                          fast_unlock=bool(kc >= 2 and float(np.median([m["t_A"] for m in C if not m["censored"]])) <= 1000),
                          matched_exposure=dict(E_star=Es, cont=kcs, fresh=kfs, call=("COMPARABLE" if abs(diff) <= 0.25 else ("CONT_LOWER" if diff < 0 else "CONT_HIGHER")),
                                                fisher_p=float(fisher_exact([[kcs, len(C) - kcs], [kfs, len(F) - kfs]])[1])),
                          km_exposure=km(C, "E_total_A"), runs=[r["xv_run"] for r in coh("WA_continuation_restart")])
L125 = float(np.log(1.25)); pr = []
for s in (501, 502, 503, 504):
    lo, hi = M[f"confirm_outputs/WB_low_s{s}"], M[f"confirm_outputs/WB_high_s{s}"]
    tl, th = lo["t_A"] if lo["t_A"] is not None else 4000, hi["t_A"] if hi["t_A"] is not None else 4000; pr.append(float(np.log(tl / th)))
out["verdicts"]["P3"] = "BENEFIT_SPEEDS" if sum(x >= L125 for x in pr) >= 3 else ("BENEFIT_SLOWS" if sum(x <= -L125 for x in pr) >= 3 else ("NO_EFFECT" if sum(abs(x) < L125 for x in pr) >= 3 else "INCONCLUSIVE"))
out["stats"]["P3"] = dict(log_ratios=pr, median=float(np.median(pr)), range=[min(pr), max(pr)], t_low=[M[f"confirm_outputs/WB_low_s{s}"]["t_A"] for s in (501, 502, 503, 504)],
                          t_high=[M[f"confirm_outputs/WB_high_s{s}"]["t_A"] for s in (501, 502, 503, 504)])
calls = {}
for c in ("WA_fresh_const", "WB_value_low", "WB_value_high"):
    acq = [M[r["dir"]] for r in coh(c) if not M[r["dir"]]["censored"]]; ks = sum(m["stage"] for m in acq)
    calls[c] = dict(acquirers=len(acq), with_stage=ks, call=("NOT_EVALUABLE" if len(acq) < 2 else ("PRESENT" if ks >= 0.75 * len(acq) else ("ABSENT" if ks <= 0.25 * len(acq) else "MIXED"))))
ev = [v["call"] for v in calls.values() if v["call"] != "NOT_EVALUABLE"]
out["verdicts"]["P5"] = "STAGE_NOT_ROBUST" if "ABSENT" in ev else ("STAGE_ROBUST" if len(ev) >= 2 and all(v == "PRESENT" for v in ev) else "MIXED")
out["stats"]["P5"] = calls
sens = {}
for fg in (0.3, 0.5, 0.7):
    for klf in (0.5, 0.75, 0.9):
        mm = {r["dir"]: model(r, TR[r["dir"]], fg=fg, klf=klf) for r in D["runs"]}
        sens[f"FG{fg}_KL{klf}"] = {c: sum(not mm[r["dir"]]["censored"] for r in coh(c)) for c in ("WA_fresh_const", "WA_continuation_restart", "WB_value_low", "WB_value_high")}
out["stats"]["threshold_sensitivity_acquired_counts"] = sens
st = {}
for n_, a_, k_ in ((3, 0.5, 0.5), (6, 0.5, 0.5), (4, 0.3, 0.5), (4, 0.7, 0.5), (4, 0.5, 0.25), (4, 0.5, 0.75)):
    mm = {r["dir"]: model(r, TR[r["dir"]], st_n=n_, st_a=a_, st_k=k_) for r in D["runs"]}
    st[f"n{n_}_A{a_}_K{k_}"] = {c: sum(bool(mm[r["dir"]]["stage"]) for r in coh(c) if not mm[r["dir"]]["censored"]) for c in ("WA_fresh_const", "WB_value_low", "WB_value_high")}
out["stats"]["stage_threshold_sensitivity"] = st
json.dump(out, open(OUT, "w"), indent=1, default=float); print(json.dumps(out["verdicts"]))
