"""XVIII frozen independent-parent rule (design v2 §10.5).  Reads logs/measure_scr.jsonl (screening baselines, stream 95_500_000) and writes
logs/XVIII_parents.json and logs/queue_p18.txt.  Rule: for each of uP162..uP165, seeds s = 20..23 (D = 811) in order; the E base is the first
competent (KL <= 0.045) seed with cls2 == EARLY, the M base the first competent seed with cls2 == MIXTURE.  If a class is missing after 20..23,
seeds 24 and 25 are screened (written to logs/queue_scr2.txt) and the rule is re-applied; a parent still lacking a class is non-bistable.
uP161 is non-bistable by the XIII-E record (0/32 early).  Bistable parents get, for both bases and D = 812: baseline, block, vrg, gsv,
sha_hi_rolled, sha_lo_real, measured on stream 95_600_000 (logs/measure_p18.jsonl)."""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
M = {}
for l in open(os.path.join(P, "logs", "measure_scr.jsonl")):
    r = json.loads(l); M[r["run"]] = r
out = dict(parents={}, uP161="non-bistable (XIII-E: 0/32 early draws; not screened)"); lines = []; extra = []
ARMS = ["baseline", "block", "vrg", "gsv", "sha_hi_rolled", "sha_lo_real"]
for p in ("uP162", "uP163", "uP164", "uP165"):
    cls = {}
    for s in (20, 21, 22, 23, 24, 25):
        t = f"{p}_fx_L{s}_W{s}_D811_stop1211_scr"
        if t in M: f = M[t]["final"]["1211"]; cls[s] = f["cls2"] if f["KL"] <= 0.045 else "INCOMPETENT"
    pick = {c: next((s for s in sorted(cls) if cls[s] == c), None) for c in ("EARLY", "MIXTURE")}
    screened = sorted(cls); need_more = any(v is None for v in pick.values()) and max(screened, default=0) < 25
    out["parents"][p] = dict(screen=cls, E_seed=pick["EARLY"], M_seed=pick["MIXTURE"], bistable=all(v is not None for v in pick.values()), pending_extra_seeds=need_more)
    if need_more:
        for s in (24, 25):
            if s not in cls: t = f"{p}_fx_L{s}_W{s}_D811_stop1211_scr"; extra.append(f"FX_STOP=1211 FX_TAGSUF=_scr python3 scripts/xviii_train.py {p} {s} {s} 811 && python3 scripts/measure18.py checkpoints/ft/{t} {p} 95500000 logs/measure_scr.jsonl")
    if out["parents"][p]["bistable"]:
        for b, s, e in (("E", pick["EARLY"], "own0"), ("M", pick["MIXTURE"], "own2")):
            bt = f"{p}_fx_L{s}_W{s}_D812_stop1211_p18"; bd = f"checkpoints/ft/{bt}"
            env = {"baseline": (f"FX_SLOG={e} FX_SLOGT=100", ""), "block": (f"FX_INTERV=gof:100 FX_EDGES={e} FX_B=1e9 FX_GLOG=1", f"_gof{e}b1e9T100"),
                   "vrg": (f"FX_INTERV=vrg:100 FX_EDGES={e} FX_GLOG=1", f"_vrg{e}T100"), "gsv": (f"FX_INTERV=gsv:100 FX_EDGES={e} FX_GLOG=1", f"_gsv{e}T100"),
                   "sha_hi_rolled": (f"FX_INTERV=sha:100 FX_EDGES={e} FX_SHA_SCHED={bd}/slog.json FX_SHA_LO=0 FX_SHA_CONTENT=rolled FX_GLOG=1", f"_sha{e}hirolledT100"),
                   "sha_lo_real": (f"FX_INTERV=sha:100 FX_EDGES={e} FX_SHA_SCHED={bd}/slog.json FX_SHA_LO=1 FX_SHA_CONTENT=real FX_GLOG=1", f"_sha{e}lorealT100")}
            for arm in ARMS:
                en, mid = env[arm]; t = f"{p}_fx_L{s}_W{s}_D812{mid}_stop1211_p18"
                c = f"{en} FX_STOP=1211 FX_TAGSUF=_p18 python3 scripts/xviii_train.py {p} {s} {s} 812 && python3 scripts/measure18.py checkpoints/ft/{t} {p} 95600000 logs/measure_p18.jsonl"
                if arm.startswith("sha"): c = f"bash scripts/when_file.sh {bd}/slog.json -- " + c
                lines.append((ARMS.index(arm), c)); out["parents"][p].setdefault("runs", {})[f"{b}|{arm}"] = t
json.dump(out, open(os.path.join(P, "logs", "XVIII_parents.json"), "w"), indent=1)
open(os.path.join(P, "logs", "queue_p18.txt"), "w").write("".join(c + "\n" for _, c in sorted(lines, key=lambda x: x[0])))
open(os.path.join(P, "logs", "queue_scr2.txt"), "w").write("".join(c + "\n" for c in extra))
print(json.dumps({p: (v["E_seed"], v["M_seed"], v["bistable"], v["pending_extra_seeds"]) for p, v in out["parents"].items()}), len(lines), "parent runs", len(extra), "extra screening runs")
