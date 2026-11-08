"""Tally REPLICATION_PROTOCOL_T4.md criteria over seeds 10-13. Writes logs/replication/tally.json and prints a markdown table."""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "logs", "replication")
rows = {s: json.load(open(os.path.join(R, f"eval_s{s}.json"))) for s in [10, 11, 12, 13]}
keys = ["R0", "R1", "R2", "R3", "R4", "R5"]
tally = {k: sum(bool(r["criteria"][k]) for r in rows.values()) for k in keys}
primary = all(tally[k] >= 3 for k in ["R0", "R1", "R2", "R3", "R4"])
out = dict(tally=tally, n=4, replicated=primary, localisation={s: r["criteria"]["localisation"] for s, r in rows.items()})
json.dump(out, open(os.path.join(R, "tally.json"), "w"), indent=1)
print("| seed | " + " | ".join(keys) + " | localisation |"); print("|" + "---|" * (len(keys) + 2))
for s, r in rows.items(): print(f"| {s} | " + " | ".join("pass" if r["criteria"][k] else "**FAIL**" for k in keys) + f" | {r['criteria']['localisation']} |")
print("| **k/4** | " + " | ".join(f"{tally[k]}/4" for k in keys) + " | |")
print("\nreplicated (R0-R4 each >= 3/4):", primary)
