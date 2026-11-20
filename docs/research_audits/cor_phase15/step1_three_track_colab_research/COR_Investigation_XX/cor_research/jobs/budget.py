"""Compute budget per preset from the frozen design and the MEASURED benchmark of this runtime (v2: counts are taken from the REAL
planner run on a synthetic eligibility file, so they cannot drift from the plan).  python -m cor_research.jobs.budget [PRESET]
Scenarios: max design (every preregistered parent valid except uP162, both bases eligible) and half units (the historical-eligibility
scenario: XVIII found 1 of 5 historical parents bistable).  The realised counts are known only after screening (registry).
GPU-hours = training steps / measured total steps-per-second at the chosen concurrency (+ measurement and Track B wall time / concurrency
+ a per-job start-up overhead).  Colab compute units are NOT converted."""
import os, sys, json, tempfile, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E, protocol as PR


def counts(preset, unit_fraction=1.0):
    from cor_research.jobs import plan as PL
    C = PR.effective(preset); pre = C["PRESETS"][preset]; tmp = tempfile.mkdtemp()
    valid = [p for p in pre["parents"] if p != "uP162"]; EL = {}
    units = [(p, b) for p in valid for b in "EM"]; keep = set(units[: max(1, round(len(units) * unit_fraction))]) if not pre.get("forced_units") else set()
    for p in valid: EL[p] = dict(status="SCREENED", units={b: dict(eligible=(p, b) in keep, seed=20 if (p, b) in keep else None, qualifying_seeds=[]) for b in "EM"})
    E.atomic_json(EL, os.path.join(tmp, "eligibility.json"))
    _pe = PL.eligibility; PL.eligibility = lambda preset, root=None: EL                # count only: never touches a real run root
    try: J = PL.plan(preset, tmp, tracks="ABC")
    finally: PL.eligibility = _pe; shutil.rmtree(tmp, ignore_errors=True)
    by = {}
    for j in J:
        if not j["id"].startswith(("scr:", "mscr:")): by[j["phase"]] = by.get(j["phase"], 0) + 1      # screening counted analytically below
    n_fresh = len([p for p in pre["parents"] if p.startswith("n")]); n_scr = len(valid) * len(pre["screen_seeds"]) * 2 if not pre.get("forced_units") else 0
    ST = C["STEPS"]; steps = dict(pretrain=n_fresh * C["PRETRAIN"]["steps"], screening=n_scr * C["SCREEN"]["stop"], windows=by.get("window", 0) * ST["window"],
                                  continuations=by.get("continue", 0) * (ST["primary"] - ST["window"]), stability=by.get("stability", 0) * (ST["stability_end"] - ST["primary"]))
    snaps = (n_scr + by.get("stability", 0)) * 26 + by.get("window", 0) * 8 + sum(26 if j["phase"] == "continue" and not j.get("prune") else 4 for j in J if j["phase"] == "continue")
    return dict(parents_preregistered=len(pre["parents"]), fresh_pretraining_runs=n_fresh, parents_valid_assumed=len(valid), screening_runs=n_scr, units=len(keep) if keep else len(pre.get("forced_units", [])),
                window_runs=by.get("window", 0), compositions=by.get("compose", 0), continuation_runs=by.get("continue", 0), stability_runs=by.get("stability", 0),
                measurements=by.get("measure", 0) + n_scr, trackb_jobs=by.get("functional", 0), training_steps=steps, snapshots_estimate=snaps)


def estimate(preset, bench=None):
    bench = bench or E.read_json(os.path.join(E.RUN_ROOT, "benchmark.json")); out = {}
    for scen, frac in (("max_design", 1.0), ("historical_eligibility_half_units", 0.5)):
        c = counts(preset, frac); r = dict(counts=c)
        if bench and bench.get("chosen_total_steps_per_s"):
            k = bench["chosen_concurrency"]; T = bench["chosen_total_steps_per_s"]; pre_rate = bench.get("pretrain_steps_per_s") or T / k
            ft = sum(v for kk, v in c["training_steps"].items() if kk != "pretrain"); n_train_jobs = c["screening_runs"] + c["window_runs"] + c["continuation_runs"] + c["stability_runs"]
            h = dict(pretrain=c["training_steps"]["pretrain"] / pre_rate / 3600 / max(1, min(k, c["fresh_pretraining_runs"] or 1)), finetune=ft / T / 3600,
                     measurement=c["measurements"] * (bench.get("measure_s") or 30) / k / 3600, trackb=c["trackb_jobs"] * (bench.get("functional_battery_s") or 120) / k / 3600,
                     job_overhead=n_train_jobs * 45 / k / 3600)
            h["total"] = sum(h.values()); r["hours"] = h; r["device"] = (bench.get("gpu") or {}).get("name") or bench.get("device"); r["concurrency"] = k
            mb = max([l.get("snapshot_mb") or 0 for l in bench["levels"]] + [0]); r["storage_gb"] = 1.5 * c["snapshots_estimate"] * mb / 1000 if mb else None   # 1.5x: optimizer pickles, resume files, logs (review C r2 N8)
            r["gpu_memory_mb_total"] = (max([l.get("gpu_mem_mb_per_proc") or 0 for l in bench["levels"]]) * k) or None
        else: r["hours"] = "run the benchmark first (notebook 00, step 6): hours are only reported from measured throughput"
        out[scen] = r
    pre = PR.full()["PRESETS"][preset]; out["claims_can"] = pre["claims_can"]; out["claims_cannot"] = pre["claims_cannot"]; out["label"] = pre["label"]; return out


if __name__ == "__main__":
    ps = [sys.argv[1]] if len(sys.argv) > 1 else ["pilot", "standard", "extended", "high"]
    print(json.dumps({p: estimate(p) for p in ps}, indent=1, default=str))
