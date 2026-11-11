"""Frozen success/failure classification of the Investigation VI cohort -> logs/cohort_status.json.
success = step4000.json exists, its eval acc is finite and >= 0.6595 (transfer threshold used since Investigation II), and the final
training loss is finite.  Everything else is a failure (kept, reported, enters ITT by worst-case imputation)."""
import json, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "checkpoints"); st = {}
for a, S in (("T1", range(70, 78)), ("P", range(80, 88))):
    for s in S:
        d = os.path.join(C, f"x{a}{s}"); f = os.path.join(d, "step4000.json"); key = f"x{a[0]}{s}"
        if not os.path.exists(f): st[key] = dict(success=False, reason="no step4000 checkpoint", have=sorted(os.listdir(d)) if os.path.isdir(d) else []); continue
        j = json.load(open(f)); acc = j["eval"]["acc"]; loss = j["hist"][-1][1] if j["hist"] else float("nan")
        pre = json.load(open(os.path.join(d, "pre.json")))
        ok = math.isfinite(acc) and acc >= 0.6595 and math.isfinite(loss)
        st[key] = dict(success=ok, reason="" if ok else f"acc {acc:.4f} / loss {loss}", acc=acc, final_loss=loss, pre_eval=pre["eval"], pre_seconds=pre["seconds"],
                       acc_by_step={n: json.load(open(os.path.join(d, f"step{n}.json")))["eval"]["acc"] for n in (250, 500, 1000, 2000, 4000)})
json.dump(st, open(os.path.join(HERE, "..", "logs", "cohort_status.json"), "w"), indent=1, default=float)
print({k: (v["success"], round(v.get("acc", float("nan")), 4)) for k, v in st.items()})
