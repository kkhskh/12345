"""High-level steps shared by ALL notebooks (00-04).  The notebooks contain configuration and calls to these functions only.
  setup()        storage (Drive or local), device, run-root layout, environment record
  verify_package() package manifest + vendored checkpoint hashes + dependency pins (integrity checks 01-03)
  track_c()      exact mathematics (CPU; independent of training)
  preflight()    freeze verification, intervention graph audit, planted scorer tests, measured benchmark, budget table
  run()          plan -> registry -> runner (resumable; stops safely at the time budget)
  score()        frozen scorers (Track A at 4000 and 1211, Track B)
  report()       RESULTS/ (11 files) + integrity checks (20)
  export()       verifiable results archive + SHA256 + post-run review bundle"""
import os, sys, json, time, subprocess, glob, zipfile, shutil, platform


def setup(preset, run_root=None, use_drive=False, drive_dir="/content/drive/MyDrive/COR_XX_RUNS", device="auto", allow_identity_change=False):
    if use_drive:
        try:
            from google.colab import drive                                          # noqa
            if not os.path.isdir("/content/drive/MyDrive"): drive.mount("/content/drive")
        except Exception as e: raise SystemExit(f"Drive requested but not mountable: {e}")
        run_root = run_root or os.path.join(drive_dir, preset)
    run_root = os.path.abspath(run_root or os.path.join("/content" if os.path.isdir("/content") else os.getcwd(), "COR_XX_RUNS", preset))
    os.makedirs(run_root, exist_ok=True); os.environ["COR_RUN_ROOT"] = run_root; os.environ["COR_PARENT_ROOT"] = os.path.join(run_root, "parents"); os.environ["COR_DEVICE"] = device
    from cor_research import env as E
    dev = E.configure(device); E.RUN_ROOT = run_root; E.PARENT_ROOT = os.environ["COR_PARENT_ROOT"]
    import jax
    info = dict(preset=preset, run_root=run_root, device=dev, jax_devices=[str(d) for d in jax.devices()], python=sys.version, platform=platform.platform(), xla_flags=os.environ.get("XLA_FLAGS"),
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else None,
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    hist = E.read_json(os.path.join(run_root, "environment_history.json"), []) or []; hist.append(info); E.atomic_json(hist, os.path.join(run_root, "environment_history.json"))
    E.atomic_json(info, os.path.join(run_root, "environment_runtime.json"))
    if dev == "cpu": print("WARNING: no GPU visible. CPU is usable for Track C and smoke tests only; scientific presets would take days on CPU.")
    info["identity"] = check_identity(run_root, allow_change=allow_identity_change)
    return info


def check_identity(run_root, allow_change=False):
    """review C B2: the first setup of a run root records its numerical identity; every later setup must match it (device kind, GPU model,
    package versions, XLA flags, matmul precision, freeze).  A deliberate change (allow_change=True) is recorded and every later job is
    labelled with the identity it ran under (registry attempts record the device)."""
    from cor_research import env as E
    cur = E.identity(); f = os.path.join(run_root, "RUN_IDENTITY.json"); rec = E.read_json(f)
    if rec is None: E.atomic_json(dict(identity=cur, changes=[]), f); return dict(status="recorded", identity=cur)
    if rec["identity"] == cur: return dict(status="match", identity=cur)
    diff = {k: (rec["identity"].get(k), cur.get(k)) for k in set(rec["identity"]) | set(cur) if rec["identity"].get(k) != cur.get(k)}
    if diff.get("freeze_sha256"): raise SystemExit(f"FREEZE CHANGED for an existing run root ({diff['freeze_sha256']}): a different protocol/code may never continue a run root (review C r2 N2). Use a new DRIVE_DIR.")
    if not allow_change: raise SystemExit(f"RUN IDENTITY MISMATCH for {run_root}: {json.dumps(diff)[:1500]}\nUse the same GPU type/software stack, or a new DRIVE_DIR, "
                                          "or set ALLOW_IDENTITY_CHANGE=True (recorded; results are then labelled as mixed-identity).")
    rec["changes"].append(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), diff=diff)); rec["identity"] = cur; rec["mixed_identity"] = True; E.atomic_json(rec, f)
    pf = os.path.join(run_root, "preflight")                                            # the preflight must be redone on the new identity
    for x in ("PREFLIGHT.json", "RESUME_TEST.json", "graph_audit.json"):
        if os.path.exists(os.path.join(pf, x)): os.replace(os.path.join(pf, x), os.path.join(pf, x + f".superseded_{int(time.time())}"))
    if os.path.exists(os.path.join(run_root, "benchmark.json")): os.replace(os.path.join(run_root, "benchmark.json"), os.path.join(run_root, f"benchmark.json.superseded_{int(time.time())}"))
    return dict(status="changed (recorded; results will be labelled MIXED-IDENTITY; preflight must be rerun)", diff=diff)


def verify_package(preset="standard"):
    from cor_research.verification import integrity as IN
    r = IN.run(preset); keep = {k: v for k, v in r["checks"].items() if k[:2] in ("01", "02", "03", "05")}
    for k, v in keep.items(): print(f"{k:40s} {v['status']}" + ("" if v["status"] == "PASS" else f"  {json.dumps({x: y for x, y in v.items() if x != 'status'})[:600]}"))
    return keep


def _py(*args, env=None):
    from cor_research import env as E
    r = subprocess.run([sys.executable, *args], cwd=E.RUN_ROOT, env=dict(os.environ, PYTHONPATH=E.PKG, **(env or {})), capture_output=True, text=True)
    return r


def track_c():
    from cor_research import env as E
    out = os.path.join(E.RUN_ROOT, "track_c", "track_c_math.json")
    if not os.path.exists(out):
        r = _py("-m", "cor_research.mathematics.run_track_c", out)
        if r.returncode: raise SystemExit(r.stderr[-3000:])
    R = json.load(open(out)); print(json.dumps(R["classification"], indent=1)); return R


def preflight(levels=(1, 2, 4, 8), bench_steps=400, preset="standard"):
    """freeze, dependencies/Python, intervention graph audit, planted scorer tests, verdict operating characteristics, runtime resume +
    determinism test, measured benchmark, budget and storage check.  Writes preflight/PREFLIGHT.json with 'pass' (required by run())."""
    from cor_research import env as E
    from cor_research.jobs import runner as RN, budget as BU
    from cor_research.verification import integrity as IN
    pf = os.path.join(E.RUN_ROOT, "preflight"); os.makedirs(pf, exist_ok=True); res = {}
    ok, bad = RN.verify_freeze(); res["freeze"] = dict(ok=ok, mismatched=bad[:20]); print("freeze verified" if ok else f"FREEZE MISMATCH: {bad[:10]}")
    ic = IN.run(preset); res["dependencies"] = ic["checks"]["02_dependencies"]; res["package"] = ic["checks"]["01_code_version"]["status"]; res["checkpoints"] = ic["checks"]["03_checkpoint_provenance"]["status"]
    print("dependencies/Python:", res["dependencies"]["status"], "| package manifest:", res["package"], "| vendored checkpoints:", res["checkpoints"])
    if not os.path.exists(os.path.join(pf, "graph_audit.json")): _py("-m", "cor_research.interventions.graph_audit", os.path.join(pf, "graph_audit.json"))
    ga = E.read_json(os.path.join(pf, "graph_audit.json")); res["graph_audit_all_pass"] = bool(ga and ga["all_pass"]); print("graph audit all pass:", res["graph_audit_all_pass"])
    for name, script in (("planted_tests", "test_scorers_planted.py"), ("verdict_oc", "test_verdict_oc.py")):
        r = _py(os.path.join(E.PKG, "tests", script)); res[name] = r.returncode == 0; print(name, "PASS" if r.returncode == 0 else "FAIL\n" + r.stdout[-1500:])
    rt0 = E.read_json(os.path.join(pf, "RESUME_TEST.json")) or {}
    if rt0.get("identity") != E.identity(): _py(os.path.join(E.PKG, "scripts", "resume_test.py"))     # per identity (review C r2 N7)
    rt = E.read_json(os.path.join(pf, "RESUME_TEST.json")) or {}; res["resume_test"] = dict(resume=rt.get("resume_bitwise_identical"), deterministic=rt.get("deterministic_rerun"), device=rt.get("device"))
    print("runtime resume test:", res["resume_test"])
    if not os.path.exists(os.path.join(E.RUN_ROOT, "benchmark.json")):
        r = _py("-m", "cor_research.jobs.benchmark", "--levels", ",".join(map(str, levels)), "--steps", str(bench_steps)); print(r.stdout[-1500:])
    b = E.read_json(os.path.join(E.RUN_ROOT, "benchmark.json")); res["benchmark"] = {k: b.get(k) for k in ("device", "gpu", "chosen_concurrency", "chosen_total_steps_per_s", "pretrain_steps_per_s")} if b else None
    res["budget"] = {p: BU.estimate(p, b) for p in ("pilot", "standard", "extended", "high")}
    free_gb = shutil.disk_usage(E.RUN_ROOT).free / 1e9; need = (res["budget"].get(preset, {}).get("max_design", {}) or {}).get("storage_gb")
    res["storage"] = dict(free_gb=free_gb, estimated_gb_max_design=need, ok=(need is not None and need < 0.9 * free_gb), note="estimate includes a 1.5x margin (optimizer/resume files); Drive moves deleted files to its trash: empty it"); print(f"storage: {free_gb:.1f} GB free; estimated {need} GB for {preset} (max design)")
    for p, v in res["budget"].items():
        h = v["max_design"].get("hours"); h2 = v["historical_eligibility_half_units"].get("hours")
        print(f"{p:9s} max design: {v['max_design']['counts']['continuation_runs']:5d} continuations, " + (f"{h['total']:.1f} h" if isinstance(h, dict) else str(h)) +
              (f";  half-units scenario {h2['total']:.1f} h" if isinstance(h2, dict) else "") + f"  [{v['label']}]")
    res["pass"] = bool(res["freeze"]["ok"] and res["dependencies"]["status"] == "PASS" and res["package"] == "PASS" and res["checkpoints"] == "PASS" and res["graph_audit_all_pass"] and res["planted_tests"] and res["verdict_oc"]
                       and res["resume_test"]["resume"] and res["resume_test"]["deterministic"] and res["storage"]["ok"])
    res["preset"] = preset; res["identity"] = E.identity(); E.atomic_json(res, os.path.join(pf, "PREFLIGHT.json")); print("PREFLIGHT", "PASS" if res["pass"] else "FAIL (run() will refuse scientific presets)"); return res


def run(preset, tracks="ABC", budget_hours=None, concurrency="auto", retry_failed=False, require_freeze=True, phases=None):
    from cor_research import env as E
    from cor_research.jobs import plan as PL, runner as RN
    if concurrency == "auto":
        b = E.read_json(os.path.join(E.RUN_ROOT, "benchmark.json")); concurrency = (b or {}).get("chosen_concurrency") or 1
    print(f"preset {preset}, tracks {tracks}, concurrency {concurrency}, budget {budget_hours} h; re-running this cell resumes exactly where it stopped")
    if preset != "smoke" and require_freeze:                                          # review C M4: no confirmation without a passing preflight on this runtime
        pf = E.read_json(os.path.join(E.RUN_ROOT, "preflight", "PREFLIGHT.json")) or {}
        if not pf.get("pass") or pf.get("identity") != E.identity(): raise SystemExit("PREFLIGHT has not passed on this run root for this runtime identity: run PL.preflight(preset=...) first and fix every FAIL")
        check_identity(E.RUN_ROOT)
    R = RN.Runner(E.RUN_ROOT, int(concurrency)); s = R.run(lambda: PL.plan(preset, E.RUN_ROOT, tracks=tracks), budget_hours=budget_hours, retry_failed=retry_failed, require_freeze=require_freeze, phases=phases)
    print(json.dumps(s, indent=1)[:3000]); return s


def score(preset):
    from cor_research import env as E
    os.makedirs(os.path.join(E.RUN_ROOT, "scores"), exist_ok=True); out = {}
    for name, args in (("track_a", ["-m", "cor_research.scoring.track_a", preset, f"scores/track_a_{preset}.json"]),   # score_all: primary + fresh-only + sensitivity
                       ("track_a_1211", ["-m", "cor_research.scoring.track_a", preset, f"scores/track_a_{preset}_1211.json", "--step", str(__import__("cor_research.protocol", fromlist=["x"]).effective(preset)["STEPS"]["horizon_hist"])]),
                       ("track_b", ["-m", "cor_research.scoring.track_b", preset, f"scores/track_b_{preset}.json"])):
        r = _py(*args); out[name] = r.returncode; print(name, "rc", r.returncode, r.stdout[-1500:])
    return out


def report(preset):
    from cor_research.reporting import report as RP
    from cor_research.verification import integrity as IN
    from cor_research import env as E
    RP.generate(preset); ic = IN.run(preset); E.atomic_json(ic, os.path.join(E.RUN_ROOT, "RESULTS", "INTEGRITY_CHECKS.json")); r = RP.generate(preset)   # 2nd pass: banner shows integrity failures
    for k, v in ic["checks"].items(): print(f"{k:42s} {v['status']}")
    print(json.dumps(r)); return r, ic


def export(preset, dest=None):
    """results archive: RESULTS/, scores/, measure/, trackB/, track_c/, registry/, eligibility, benchmark, preflight, logs, parents' lineage
    (no checkpoints: they stay in the run root; their hashes are in the registry and the evidence manifest).  + SHA256 + review bundle."""
    from cor_research import env as E
    root = E.RUN_ROOT; dest = dest or os.path.join(root, "EXPORT"); os.makedirs(dest, exist_ok=True); stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rb = _py(os.path.join(E.PKG, "scripts", "make_review_bundle.py"), preset); print(rb.stdout[-800:])
    z = os.path.join(dest, f"cor_xx_results_{preset}_{stamp}.zip"); pats = ["RESULTS/*", "scores/*", "measure/*/*", "trackB/*", "track_c/*", "registry/jobs/*", "eligibility.json", "benchmark.json", "preflight/*",
                                                                             "environment_*.json", "logs/*", "parents/*/lineage.json", "parents/*/pre.json", "REVIEW_BUNDLE/*", "ft/*/run_config.json", "ft/*/final.json", "ft/*/provenance.json", "ft/*/resume_log.json"]
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(root, pat))):
                if os.path.isfile(f): zf.write(f, os.path.relpath(f, root))
    with zipfile.ZipFile(z) as zf: bad = zf.testzip()
    h = E.sha256(z); open(z + ".sha256", "w").write(f"{h}  {os.path.basename(z)}\n"); print("results archive", z, "sha256", h, "zip test:", "OK" if bad is None else bad); return z, h
