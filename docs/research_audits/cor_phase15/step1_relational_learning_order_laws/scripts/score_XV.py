"""XV frozen scorer (FAIL-CLOSED).  Usage: score_XV.py [ROOT=package dir] [OUT=logs/score_XV.json]
Refuses (exit 2, writes nothing) unless: frozen files match logs/FREEZE_XV_HASHES.txt; every design run has done.json with the design's
cond/seed, blind = true, stop = 4000 and the init hash recomputed from train_tasks.init; every snapshot file exists; every trajectory JSON
exists with the design's stream / n / cond and one entry per snapshot; the exact-reference cache passes the math verification
(logs/xv_math_verified.json all_pass = true).
Per model (frozen definitions):
  t_G : first snapshot s with G >= 0.5 at s and at the next snapshot (censored at 4000 if none; reported as '>4000').
  t_I : first snapshot s with A_p >= 0.5 and G <= 0.1 at s and at the next snapshot (intermediate stage); absent if none.
  t_S : first snapshot with acc >= acc(order-1 optimum) - 0.03 and G <= 0.1 (source stage).
  width_G : steps from the last snapshot with G <= 0.1 before t_G to the first snapshot with G >= 0.9 at or after t_G.
  competent : final KL(Bayes || model) <= 0.02 and final G >= 0.9.
P1 complexity (8 seed-paired comparisons: B vs C and A vs D):  r = ln(t_G(arity 3) / t_G(arity 2)), censored t_G = 4000 (lower bound).
   COMPLEXITY_DELAYS: r >= ln 1.5 in >= 7/8;  NO_EFFECT: |r| < ln 1.25 in >= 6/8;  otherwise INCONCLUSIVE.
P2 usefulness (8 seed-paired comparisons: A vs B and D vs C; the gate's exact gain is ~2x larger in B, C):
   r = ln(t_G(lower gain) / t_G(higher gain)).  BENEFIT_SPEEDS: r >= ln 1.25 in >= 6/8;  BENEFIT_SLOWS: r <= -ln 1.25 in >= 6/8;
   NO_EFFECT: |r| < ln 1.25 in >= 6/8;  otherwise INCONCLUSIVE.
P3 intermediate stage pattern: a model shows the stage if t_I exists and precedes t_G (or t_G is censored).  Counted over the models
   that reached the source stage (t_S exists), whether or not they later became competent (a gate never acquired is censored, the stage
   before it is still observed); per condition PRESENT if >= 3 of 4 show it, ABSENT if <= 1, else MIXED (also MIXED if < 3 models reached
   the source stage).  A competent-only version is reported as a sensitivity analysis.  Verdict = the hypothesis whose pattern (A,B,C,D) matches all four calls:
   H1 complexity-first (+,+,+,+);  H2 payoff (+,-,-,+);  RACE (+,-,+,+);  else NONE_MATCH.
Censoring: a model's unacquired gate enters P1/P2 as censored t_G = 4000 (a lower bound; reported as not acquired within budget).  All per-model times are reported."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."); OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, "logs", "score_XV.json")
def refuse(m): print("REFUSED:", m); sys.exit(2)
for l in open(os.path.join(PKG, "logs", "FREEZE_XV_HASHES.txt")):
    h, p = l.split()
    if hashlib.sha256(open(os.path.join(PKG, p), "rb").read()).hexdigest() != h: refuse(f"frozen file changed: {p}")
V = os.path.join(PKG, "logs", "xv_math_verified.json")
if not os.path.exists(V) or not json.load(open(V)).get("all_pass"): refuse("mathematical verification missing or failed")
D = json.load(open(os.path.join(PKG, "logs", "XV_design.json")))
SN = list(range(25, 2001, 25)) + list(range(2250, 4001, 250))
sys.path.insert(0, os.path.join(PKG, "..", "step1_foundational_atlas_continuation", "src"))
import numpy as np
for r in D["runs"]:
    d = os.path.join(PKG, r["dir"]); f = os.path.join(d, "done.json")
    if not os.path.exists(f): refuse(f"run not finished: {r['dir']}")
    j = json.load(open(f))
    if j["cond"] != r["spec"] or j["seed"] != r["seed"] or not j["blind"] or j["stop"] != 4000: refuse(f"config mismatch: {r['dir']}")
    if [k for k in SN if not os.path.exists(os.path.join(d, f"step{k}.npz"))]: refuse(f"snapshots missing: {r['dir']}")
    t = os.path.join(PKG, r["traj"])
    if not os.path.exists(t): refuse(f"trajectory missing: {r['traj']}")
    T = json.load(open(t))
    if T["stream"] != D["stream"] or T["n"] != D["n_episodes"] or T["cond"] != r["spec"] or [x["step"] for x in T["traj"]] != SN: refuse(f"trajectory mismatch: {r['traj']}")
import jax, train_tasks as TT
inith = {}
for r in D["runs"]:
    if r["seed"] not in inith: inith[r["seed"]] = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(TT.init("gate", r["seed"])))).hexdigest()
    if json.load(open(os.path.join(PKG, r["dir"], "done.json")))["init_sha256"] != inith[r["seed"]]: refuse(f"init hash mismatch: {r['dir']}")
def times(T, acc1):
    tr = T["traj"]; n = len(tr); out = {}
    tG = next((tr[i]["step"] for i in range(n - 1) if tr[i]["G"] >= 0.5 and tr[i + 1]["G"] >= 0.5), None)
    tI = next((tr[i]["step"] for i in range(n - 1) if tr[i]["A_p"] >= 0.5 and tr[i]["G"] <= 0.1 and tr[i + 1]["A_p"] >= 0.5 and tr[i + 1]["G"] <= 0.1), None)
    tS = next((x["step"] for x in tr if x["acc"] >= acc1 - 0.03 and x["G"] <= 0.1), None)
    w = None
    if tG is not None:
        i = [x["step"] for x in tr].index(tG); lo = max([tr[j]["step"] for j in range(i + 1) if tr[j]["G"] <= 0.1], default=0)
        hi = next((tr[j]["step"] for j in range(i, n) if tr[j]["G"] >= 0.9), None); w = (hi - lo) if hi is not None else None
    fin = tr[-1]; return dict(t_G=tG, t_I=tI, t_S=tS, width_G=w, competent=bool(fin["KL_bayes"] <= 0.02 and fin["G"] >= 0.9), final=fin,
                              max_A_p_before_gate=max([x["A_p"] for x in tr if x["G"] <= 0.1], default=None))
M = {}
for r in D["runs"]:
    T = json.load(open(os.path.join(PKG, r["traj"]))); M[(r["cond"], r["seed"])] = times(T, T["refs"]["order1"]["acc"])
tg = lambda c, s: M[(c, s)]["t_G"] if M[(c, s)]["t_G"] is not None else 4000
out = dict(models={f"{c}_s{s}": v for (c, s), v in M.items()}, verdicts={}, comparisons={})
L15, L125 = np.log(1.5), np.log(1.25); S = D["seeds"]
r1 = [float(np.log(tg(hi, s) / tg(lo, s))) for lo, hi in (("B", "C"), ("A", "D")) for s in S]
out["comparisons"]["P1_log_ratios"] = r1; out["comparisons"]["P1_censored_arity3"] = sum(M[(c, s)]["t_G"] is None for c in "CD" for s in S)
out["verdicts"]["P1"] = "COMPLEXITY_DELAYS" if sum(x >= L15 for x in r1) >= 7 else ("NO_EFFECT" if sum(abs(x) < L125 for x in r1) >= 6 else "INCONCLUSIVE")
r2 = [float(np.log(tg(lo, s) / tg(hi, s))) for lo, hi in (("A", "B"), ("D", "C")) for s in S]
out["comparisons"]["P2_log_ratios"] = r2
out["verdicts"]["P2"] = "BENEFIT_SPEEDS" if sum(x >= L125 for x in r2) >= 6 else ("BENEFIT_SLOWS" if sum(x <= -L125 for x in r2) >= 6 else ("NO_EFFECT" if sum(abs(x) < L125 for x in r2) >= 6 else "INCONCLUSIVE"))
calls, calls_comp = {}, {}
for c in "ABCD":
    allm = [M[(c, s)] for s in S]; srcok = [m for m in allm if m["t_S"] is not None]
    k = sum(m["t_I"] is not None and (m["t_G"] is None or m["t_I"] < m["t_G"]) for m in srcok)
    calls[c] = "MIXED" if len(srcok) < 3 else ("PRESENT" if k >= 3 else ("ABSENT" if k <= 1 else "MIXED"))
    comp = [m for m in allm if m["competent"]]; kc = sum(m["t_I"] is not None and m["t_I"] < m["t_G"] for m in comp)
    calls_comp[c] = "MIXED" if len(comp) < 3 else ("PRESENT" if kc >= 3 else ("ABSENT" if kc <= 1 else "MIXED"))
out["comparisons"]["P3_calls_competent_only_sensitivity"] = calls_comp
out["comparisons"]["P3_calls"] = calls
pat = tuple("+" if calls[c] == "PRESENT" else ("-" if calls[c] == "ABSENT" else "?") for c in "ABCD")
H = {"H1_COMPLEXITY_FIRST": ("+", "+", "+", "+"), "H2_PAYOFF": ("+", "-", "-", "+"), "RACE": ("+", "-", "+", "+")}
out["verdicts"]["P3"] = next((k for k, v in H.items() if v == pat), "NONE_MATCH")
rng = np.random.default_rng(0)
for nm, r in (("P1", r1), ("P2", r2)):
    v = np.array(r); b = [float(np.median(v[rng.integers(0, len(v), len(v))])) for _ in range(5000)]; out["comparisons"][f"{nm}_median_log_ratio"] = float(np.median(v)); out["comparisons"][f"{nm}_median_ci95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out["n_competent"] = {c: sum(M[(c, s)]["competent"] for s in S) for c in "ABCD"}
json.dump(out, open(OUT, "w"), indent=1, default=float); print(json.dumps(out["verdicts"]), out["n_competent"])
