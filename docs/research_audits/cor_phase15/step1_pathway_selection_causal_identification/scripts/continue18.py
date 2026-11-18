"""XVIII frozen stability readout (design v2 §10.4): every P-band or unstable run listed in a score JSON ('stability_continuations') is
continued bitwise from its step-1211 state to step 2546 on the unchanged LR schedule, then measured at step 2546 on the same stream.
Usage: continue18.py SCORE.json DESIGN.json MEASURE_OUT.jsonl  -> writes logs/queue_cont18.txt (copies resume state into *_cont2546 dirs)."""
import json, os, sys, shutil
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
S = json.load(open(sys.argv[1])); D = json.load(open(sys.argv[2])); mout = sys.argv[3]; lines = []
specs = {v["tag"]: v for v in D["runs"].values()} if "runs" in D and isinstance(D["runs"], dict) and all(isinstance(v, dict) and "tag" in v for v in D["runs"].values()) else {}
for tag in S["stability_continuations"]:
    sp = specs[tag]; src = os.path.join(P, sp["dir"]); new = tag.replace("_stop1211", "_stop2546") + "_cont"; dst = os.path.join(P, "checkpoints", "ft", new)
    os.makedirs(dst, exist_ok=True)
    for f in ("resume.pkl", "step1211.npz"): shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    b = D["bases"][sp["base"]]; suf = tag.split("_stop1211")[1]
    lines.append(f"{sp['env']} FX_STOP=2546 FX_EXTRA_SNAPS=2546 FX_TAGSUF={suf}_cont python3 scripts/xviii_train.py {D['parent']} {b['L']} {b['W']} {sp['D']} && python3 scripts/measure18.py checkpoints/ft/{new} {D['parent']} {D['stream']} {mout}")
open(os.path.join(P, "logs", "queue_cont18.txt"), "w").write("".join(l + "\n" for l in lines)); print(len(lines), "continuations")
