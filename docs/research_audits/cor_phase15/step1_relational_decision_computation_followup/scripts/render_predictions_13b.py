"""Render the frozen prediction table of FROZEN_CONFIRMATION_AND_STATISTICS.md from logs/predictions_13b.json (single source of truth)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); S = json.load(open(os.path.join(HERE, "..", "logs", "predictions_13b.json")))
o = [S["note"], "", "Pass rule: an arm passes if its test holds on >= ceil(0.75 n) of its units (n >= min_units, else INCONCLUSIVE); a prediction is CONFIRMED iff all arms pass; "
     "an arm is DIAGNOSTIC only if its null was computed on every unit and failed on every unit.", ""]
for P in S["predictions"]:
    o += [f"### {P['id']}: {P['title']}", "", f"- **Competing explanations:** {P['competing']}", f"- **Procedure:** {P['procedure']}", f"- **Unit:** {P['unit']} (min_units {P.get('min_units', 2)})" + (f"; unit filter `{P['unit_filter']}`" if P.get("unit_filter") else "") + (f"; parents {P['parents']}" if P.get("parents") else ""),
          f"- **Interpretation:** {P['interpretation']}", "", "| arm | statement | test | null |", "|---|---|---|---|"]
    for A in P["arms"]: o.append(f"| {A['id']} | {A['text']}" + (f" (units: `{A['unit_filter']}`)" if A.get("unit_filter") else "") + f" | `{A['test']}` | " + (f"`{A['null']}`" if A.get("null") else "none") + " |")
    o.append("")
open(os.path.join(HERE, "..", "logs", "predictions_13b_rendered.md"), "w").write("\n".join(o)); print(len(o), "lines")
