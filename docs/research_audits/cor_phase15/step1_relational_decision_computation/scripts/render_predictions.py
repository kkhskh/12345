"""Render the prediction table of FROZEN_PROTOCOL_AND_CONFIRMATION.md from logs/predictions_xiii.json (single source of truth for text and scorer)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); S = json.load(open(os.path.join(HERE, "..", "logs", "predictions_xiii.json")))
o = [f"Mixture-class rule (frozen): a parent is mixture-class iff some layer-2 head has readout-token attention mass on the target question's six source "
     f"tokens >= {S['mixture_class_rule']['mass']} AND direct-logit variance share >= {S['mixture_class_rule']['share']} (l2_scan).", "",
     f"Pass rule: an arm passes if its test holds in >= ceil(0.75 n) units (n >= min_units, default 2; otherwise INCONCLUSIVE). A prediction is CONFIRMED iff all its arms pass. "
     f"An arm with a null is DIAGNOSTIC only if the null fails on every unit; a passing arm whose null passes anywhere is NON-DIAGNOSTIC.", ""]
for P in S["predictions"]:
    o.append(f"### {P['id']}: {P['title']}")
    o.append(f"Units: {P['unit']} parents" + (", mixture-class only" if P.get("mixture_only") else "") + (f"; status: {P['status']}" if P.get("status") else "") + "\n")
    o.append(P["rationale"] + "\n")
    o.append("| arm | statement | test (scorer expression) | null (matched control substituted) |"); o.append("|---|---|---|---|")
    for A in P["arms"]: o.append(f"| {A['id']} | {A['text']} | `{A['test']}` | " + (f"`{A['null']}`" if A.get("null") else "none (validity/descriptive arm)") + " |")
    o.append("")
open(os.path.join(HERE, "..", "logs", "predictions_xiii_rendered.md"), "w").write("\n".join(o)); print("\n".join(o)[:1500])
