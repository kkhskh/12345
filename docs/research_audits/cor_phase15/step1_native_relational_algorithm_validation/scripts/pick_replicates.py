"""Stage-0 rule: same-parent replicate fine-tunes (seed 11) go to the FIRST TWO COMPLETE parents (aux_acc >= 0.99) in seed order.
Reads only pre.json.  Waits until the decision is determined (enough parents finished in seed order).  Writes logs/replicates.txt."""
import json, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); CK = os.path.join(HERE, "..", "checkpoints"); P = ["P150", "P151", "P152", "P153", "P154", "P155", "P156"]   # Amendment 1
while True:
    chosen = []; undetermined = False
    for p in P:
        f = os.path.join(CK, f"z{p}", "pre.json")
        if not os.path.exists(f): undetermined = True; break
        if json.load(open(f))["eval"]["aux_acc"] >= 0.99: chosen.append(p)
        if len(chosen) == 2: break
    if len(chosen) == 2 or not undetermined: break
    time.sleep(30)
if len(chosen) < 2:                                   # Amendment 1: fill with the highest-aux_acc incomplete parent(s)
    rest = sorted([p for p in P if p not in chosen], key=lambda p: -json.load(open(os.path.join(CK, f"z{p}", "pre.json")))["eval"]["aux_acc"])
    chosen += rest[: 2 - len(chosen)]
open(os.path.join(HERE, "..", "logs", "replicates.txt"), "w").write(" ".join(chosen) + "\n"); print(chosen)
