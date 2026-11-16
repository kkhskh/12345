"""Non-confirmation progress monitor: counts finished re-roll runs and continuations ONLY (never reads metrics or classes)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "..", "logs", "A_design.json"))); CK = os.path.join(HERE, "..", "checkpoints", "ft")
done = sum(os.path.exists(os.path.join(CK, r["tag"], "stop_done.json")) for r in D["runs"]); print(f"re-roll runs finished: {done}/{len(D['runs'])}")
