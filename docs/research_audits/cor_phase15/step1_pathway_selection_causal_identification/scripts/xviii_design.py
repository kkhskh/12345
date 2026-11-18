"""XVIII frozen design (design v2, PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §10): within-parent confirmation on uP160.
Writes logs/XVIII_design.json and logs/queue_c18.txt.  Every training line is followed (same lane) by the frozen measurement on the
confirmation stream 95_400_000 (never used before in COR).  Baselines come first; clamp arms wait for their paired baseline's slog.json."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, ".."); PARENT = "uP160"; STREAM = 95_400_000; SUF = "_c18"
BASES = dict(E=dict(L=21, W=21, edges="own0"), M=dict(L=23, W=23, edges="own2")); ORDERS = [811, 812, 813]
def tag(b, D, mid): B = BASES[b]; return f"{PARENT}_fx_L{B['L']}_W{B['W']}_D{D}{mid}_stop1211{SUF}"
def arms(b, D):
    e = BASES[b]["edges"]; base_dir = f"checkpoints/ft/{tag(b, D, '')}"
    return {"baseline": (f"FX_SLOG={e} FX_SLOGT=100", ""),
            "block": (f"FX_INTERV=gof:100 FX_EDGES={e} FX_B=1e9 FX_GLOG=1", f"_gof{e}b1e9T100"),
            "vrg": (f"FX_INTERV=vrg:100 FX_EDGES={e} FX_GLOG=1", f"_vrg{e}T100"),
            "gsg": (f"FX_INTERV=gsg:100 FX_EDGES={e} FX_GLOG=1", f"_gsg{e}T100"),
            "gsv": (f"FX_INTERV=gsv:100 FX_EDGES={e} FX_GLOG=1", f"_gsv{e}T100"),
            "sha_hi_rolled": (f"FX_INTERV=sha:100 FX_EDGES={e} FX_SHA_SCHED={base_dir}/slog.json FX_SHA_LO=0 FX_SHA_CONTENT=rolled FX_GLOG=1", f"_sha{e}hirolledT100"),
            "sha_lo_real": (f"FX_INTERV=sha:100 FX_EDGES={e} FX_SHA_SCHED={base_dir}/slog.json FX_SHA_LO=1 FX_SHA_CONTENT=real FX_GLOG=1", f"_sha{e}lorealT100"),
            "gof23": (f"FX_INTERV=gof:100 FX_EDGES={e} FX_B=2.3 FX_GLOG=1", f"_gof{e}b2.3T100")}
ORDER_OF_ARMS = ["baseline", "block", "vrg", "gsg", "gsv", "sha_hi_rolled", "sha_lo_real", "gof23"]
design = dict(parent=PARENT, stream=STREAM, bases=BASES, orders=ORDERS, arms=ORDER_OF_ARMS, decisive_arms=["vrg", "gsg", "gsv", "sha_hi_rolled", "sha_lo_real"], secondary_arms=["gof23"],
              runs={}, queue="logs/queue_c18.txt")
lines = []
for arm in ORDER_OF_ARMS:
    for b in ("E", "M"):
        for D in ORDERS:
            env, mid = arms(b, D)[arm]; t = tag(b, D, mid); B = BASES[b]
            design["runs"][f"{b}|{D}|{arm}"] = dict(base=b, D=D, arm=arm, tag=t, dir=f"checkpoints/ft/{t}", env=env)
            cmd = f"{env} FX_STOP=1211 FX_TAGSUF={SUF} python3 scripts/xviii_train.py {PARENT} {B['L']} {B['W']} {D} && python3 scripts/measure18.py checkpoints/ft/{t} {PARENT} {STREAM} logs/measure_c18.jsonl"
            if arm.startswith("sha"): cmd = f"bash scripts/when_file.sh checkpoints/ft/{tag(b, D, '')}/slog.json -- " + cmd
            lines.append(cmd)
json.dump(design, open(os.path.join(P, "logs", "XVIII_design.json"), "w"), indent=1)
open(os.path.join(P, "logs", "queue_c18.txt"), "w").write("\n".join(lines) + "\n"); print(len(lines), "runs")
