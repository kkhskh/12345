"""XVI verification step 16: key numbers and verdicts stated in the reports are checked against the primary result files
(logs/score_XVI.json, logs/xvi_stats.json, logs/xvi_math_verified.json, logs/expl/expl_summary.json).  Prints each check and ALL_PASS."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, "..")
S = json.load(open(os.path.join(P, "logs", "score_XVI.json"))); T = json.load(open(os.path.join(P, "logs", "xvi_stats.json")))
V = json.load(open(os.path.join(P, "logs", "xvi_math_verified.json"))); E = json.load(open(os.path.join(P, "logs", "expl", "expl_summary.json")))
doc = {f: open(os.path.join(P, f)).read() for f in ("MAIN_RESEARCH_SYNTHESIS.md", "EXTENDED_RELATIONAL_LEARNING.md", "FROZEN_CONFIRMATION_AND_STATISTICS.md", "DECISION_VALUE_AND_COMPETING_CUES.md", "LEARNING_SIGNAL_AND_STAGE_VALIDATION.md", "HIGHER_ORDER_COMPLEXITY_AND_SCALING.md")}
M = {k.split("/")[-1]: v for k, v in S["models"].items()}; checks = []
def chk(name, ok): checks.append((name, bool(ok))); print(("PASS " if ok else "FAIL ") + name)
for p, v in S["verdicts"].items(): chk(f"verdict {p}={v} stated in MAIN and FROZEN_CONFIRMATION", v in doc["MAIN_RESEARCH_SYNTHESIS.md"] and v in doc["FROZEN_CONFIRMATION_AND_STATISTICS.md"])
tf = sorted(M[f"WA_const_s{s}"]["t_A"] for s in (401, 402, 403, 404)); chk(f"fresh t_A {tf} in MAIN", f"{tf[0]}–{tf[-1]}" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
for s in (401, 402, 403, 404): chk(f"fresh s{s} t_A {M[f'WA_const_s{s}']['t_A']} in EXTENDED", f"| s{s} | {M[f'WA_const_s{s}']['t_first']} | {M[f'WA_const_s{s}']['t_A']} |" in doc["EXTENDED_RELATIONAL_LEARNING.md"])
cens = sorted(k for k, v in M.items() if v["censored"]); chk(f"censored runs {cens}", cens == ["WA_cont_D_s201", "WA_cont_D_s202"] and "D_s201" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
for r in ("C_s201", "C_s202", "C_s203", "D_s204"): chk(f"restart {r} t_A {M['WA_cont_' + r]['t_A']} in EXTENDED", f"| {r} | {M['WA_cont_' + r]['t_A']} |" in doc["EXTENDED_RELATIONAL_LEARNING.md"])
for s, lo, hi in zip((501, 502, 503, 504), T["P3"]["t_low"], T["P3"]["t_high"]): chk(f"WB pair {s} ({lo}, {hi}) in DECISION_VALUE", f"| {s} | {lo} | {hi} |" in doc["DECISION_VALUE_AND_COMPETING_CUES.md"])
chk("P3 ratios 1.62, 0.71, 0.88, 2.03", [round(x, 2) for x in T["P3"]["ratios"]] == [1.62, 0.71, 0.88, 2.03] and "1.62, 0.71, 0.88, 2.03" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
chk("math verification all_pass and 7 checks", V["all_pass"] and len(V["checks"]) == 7)
v4 = V["checks"]["V4_value_contrast"]["detail"]; chk(f"value ratio {v4['ratio']:.2f} and agreement-benefit diff {100*v4['int_rel_diff']:.2f}% stated", f"{v4['ratio']:.2f}" in doc["DECISION_VALUE_AND_COMPETING_CUES.md"] and f"{100*v4['int_rel_diff']:.2f}%" in doc["DECISION_VALUE_AND_COMPETING_CUES.md"])
c = E["gradsig_controls"]; chk(f"estimator controls (null |z| <= {max(abs(x) for x in c['null_z_range']):.2f}, planted {c['planted_range'][0]:.2f}-{c['planted_range'][1]:.2f}) stated", f"{c['planted_range'][0]:.2f}" in doc["LEARNING_SIGNAL_AND_STAGE_VALIDATION.md"] and c["null_abs_gt2"] == 0)
chk("P5 12/12 acquirers with stage", sum(v["with_stage"] for v in S["stats"]["P5"].values()) == 12 and "12/12" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
chk("no 'human peer review' claim", all("not human peer review" in doc[f] or "peer review" not in doc[f] for f in doc))
chk("archive not claimed standalone", "not standalone" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
print("ALL_PASS", all(ok for _, ok in checks), f"({sum(ok for _, ok in checks)}/{len(checks)})")
