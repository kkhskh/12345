"""Retrospective (development) analysis: P-count pretraining completeness (pretraining count accuracy, read from pre.json) vs the original
transfer's outcomes on the Investigation VIII lineages zP100-107 (Investigation VIII confirmation evals, eval_conf8) and Investigation VI
development parents xP80/xP81 (eval_dev; Investigation VIII dev orig run exists only for xP80).  -> logs/completeness_retro.json"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.join(HERE, "..", "..", "step1_learned_relational_computation")
rows = []
for p in range(100, 108):
    pre = json.load(open(f"{V8}/checkpoints/zP{p}/pre.json"))["eval"]["aux_acc"]
    for spec in ("orig", "radapt"):
        e = json.load(open(f"{V8}/checkpoints/conf/zP{p}_{spec}.json"))["evals"]["4000"]; F = e["families"]["F45"]
        r23 = (F["R2"]["absE"] * F["R2"]["n"] + F["R3"]["absE"] * F["R3"]["n"]) / (F["R2"]["n"] + F["R3"]["n"])
        rows.append(dict(parent=f"zP{p}", spec=spec, pretrain_count_acc=pre, complete=pre > 0.99, KL=e["quality"]["KL"], absE_R1=F["R1"]["absE"], E_R1=F["R1"]["E"],
                         absE_R23=r23, absE_F34=e["families"]["F34"]["R1"]["absE"]))
for r in rows: print(r["parent"], r["spec"], round(r["pretrain_count_acc"], 3), " ".join(f"{k} {r[k]:+.3f}" for k in ("KL", "absE_R1", "E_R1", "absE_R23", "absE_F34")))
o = [r for r in rows if r["spec"] == "orig"]
import statistics as st
summ = {g: {k: st.mean(r[k] for r in o if r["complete"] == g) for k in ("KL", "absE_R1", "E_R1", "absE_R23", "absE_F34")} for g in (True, False)}
print("complete", summ[True]); print("incomplete", summ[False])
json.dump(dict(rows=rows, orig_means={"complete": summ[True], "incomplete": summ[False]}), open(os.path.join(HERE, "..", "logs", "completeness_retro.json"), "w"), indent=1)
