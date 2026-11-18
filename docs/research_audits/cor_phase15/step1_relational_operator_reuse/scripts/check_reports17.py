"""XVII verification step 17: reports checked against primary results (logs/score_XVII.json, logs/analysis/*.json, learning/*, logs/xvii_od_check.json)
and against the adjudication (every accepted correction ID listed in reviews/ADJUDICATION.md must be referenced in MAIN_RESEARCH_SYNTHESIS.md)."""
import json, os, re, glob
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); S = json.load(open(os.path.join(P, "logs", "score_XVII.json")))
doc = {f: open(os.path.join(P, f)).read() for f in ("MAIN_RESEARCH_SYNTHESIS.md", "FROZEN_CONFIRMATION_AND_GENERALIZATION.md", "CROSS_SLOT_CAUSAL_TRANSFER.md", "ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md")}
ok = []
def chk(n, c): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n)
for k, v in S["verdicts"]["pooled"].items(): chk(f"pooled {k}={v['call']} ({v['k']}/{v['n']}) in MAIN and FROZEN_CONFIRMATION", v["call"] in doc["MAIN_RESEARCH_SYNTHESIS.md"] and v["call"] in doc["FROZEN_CONFIRMATION_AND_GENERALIZATION.md"] and f"{v['k']}/{v['n']}" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
for n, t in S["models"].items():
    if t.get("P2_values"): chk(f"{n} [0,2) transfer {t['P2_values']['transfer']:.2f} in FROZEN_CONFIRMATION", f"{t['P2_values']['transfer']:.2f}" in doc["FROZEN_CONFIRMATION_AND_GENERALIZATION.md"])
ev = [n for n, t in S["models"].items() if t.get("P1") is not None]; chk(f"evaluable models {len(ev)} = 6 stated", len(ev) == 6 and "6 evaluable" in doc["FROZEN_CONFIRMATION_AND_GENERALIZATION.md"])
od = json.load(open(os.path.join(P, "logs", "xvii_od_check.json"))); chk("evidence-only gate-flip statistic (0.000, vacuous) reproduced and marked withdrawn W1; S' stated", all(v["fraction_inputs_where_gate_flip_changes_minD"] == 0.0 for v in od.values()) and "W1" in doc["ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md"] and "S′" in doc["ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md"] and "S′" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
chk("H4 mixture verdict not asserted in MAIN", "a mixture (H4)" not in doc["MAIN_RESEARCH_SYNTHESIS.md"])
chk("O/D not presented as validated", "not validated" in doc["MAIN_RESEARCH_SYNTHESIS.md"] and "silent" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
chk("agent reviews not called human peer review", "not human peer review" in doc["MAIN_RESEARCH_SYNTHESIS.md"]); chk("archive not standalone", "not standalone" in doc["MAIN_RESEARCH_SYNTHESIS.md"])
adj = os.path.join(P, "reviews", "ADJUDICATION.md")
if os.path.exists(adj):
    ids = sorted(set(re.findall(r"\*\*(W\d+)", open(adj).read()))); chk(f"adjudication withdrawals {ids} referenced in MAIN", all(i in doc["MAIN_RESEARCH_SYNTHESIS.md"] for i in ids))
print("ALL_PASS", all(ok), f"({sum(ok)}/{len(ok)})")
