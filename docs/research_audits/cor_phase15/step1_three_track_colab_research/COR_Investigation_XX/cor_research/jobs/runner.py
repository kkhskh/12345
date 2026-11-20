"""Job registry and runner (mandate 7.4 / 7.5).  v2 after pre-run review C (B3, M2, M3, M11).

Registry: $COR_RUN_ROOT/registry/jobs/<safe id>.json, one file per job, written atomically; the runner keeps an in-memory copy
(write-through) and re-plans only when a job finishes (no rescans of Drive every few seconds).  Fields: id, track, phase, parent, base,
seed, data_order, arm, stream, start_ckpt, target_steps, argv, argv_norm, env, outputs, deps, status (pending | running | done |
failed | blocked), attempts [{start, end, rc, seconds, user_cpu_s, max_rss_mb, gpu_mem_mb_max, device, pid, boot_id, log, note}],
output_hashes, snapshot_hashes (every step*.npz of a training job at completion), pruned {file: sha256}, completed_steps, latest_ckpt.
Rules
  * a failed job is NEVER deleted or silently replaced; a retry (retry_failed) appends an attempt and resumes from resume.pkl;
  * a job whose output directory already contains files but has no registry record is 'blocked' (never adopted, never resumed);
  * a job definition that changes after registration is an integrity failure (raises);
  * outputs are validated (npz / pkl / json must load) before a job is marked done;
  * interrupting the runner (KeyboardInterrupt in a notebook) terminates its child processes and records the attempts as
    'interrupted' (status back to pending: training resumes exactly); a 'running' record whose process is still alive on this machine
    (same boot id and pid) is never reset; a stale one (dead process or another machine) is reset to pending with the attempt kept;
  * one runner per run root (heartbeat lock);
  * before any job: freeze verification (fail-closed); run identity and preflight are enforced by cor_research.pipeline.run."""
import os, sys, json, time, subprocess, threading, re, glob, pickle, signal
from cor_research import env as E

REG = lambda root: os.path.join(root, "registry", "jobs")
SAFE = lambda i: re.sub(r"[^A-Za-z0-9_.@:-]", "_", i).replace(":", "__")
TRAIN_PHASES = ("pretrain", "screen", "window", "continue", "stability")


def norm(argv): return [a.replace(E.PKG, "<PKG>") for a in argv[1:]]          # location-independent job definition
def rec_path(root, jid): return os.path.join(REG(root), SAFE(jid) + ".json")


def boot_id():
    try: return open("/proc/sys/kernel/random/boot_id").read().strip()
    except Exception: return None


def load_all(root):
    return {r["id"]: r for r in (E.read_json(f) for f in sorted(glob.glob(os.path.join(REG(root), "*.json")))) if r}


def verify_freeze(pkg=None):
    pkg = pkg or E.PKG; fz = E.read_json(os.path.join(pkg, "protocols", "FREEZE.json"))
    if not fz: return False, ["protocols/FREEZE.json missing"]
    bad = [f for f, h in fz["files"].items() if not os.path.exists(os.path.join(pkg, f)) or E.sha256(os.path.join(pkg, f)) != h]
    return (not bad), bad


def validate(path):
    try:
        if path.endswith(".npz"):
            import numpy as np
            with np.load(path) as z: _ = [z[k].shape for k in z.files]
        elif path.endswith(".pkl"): pickle.load(open(path, "rb"))
        elif path.endswith(".json"): json.load(open(path))
        return True
    except Exception: return False


def progress(root, j):
    if j["phase"] not in TRAIN_PHASES or not j["outputs"]: return None, None
    d = os.path.join(root, os.path.dirname(j["outputs"][0]))
    if j["phase"] == "pretrain": return (j.get("target_steps"), "pre.npz") if E.read_json(os.path.join(d, "pre.json")) else (None, None)
    st = sorted(int(m.group(1)) for m in (re.match(r"step(\d+)\.npz$", f) for f in (os.listdir(d) if os.path.isdir(d) else [])) if m)
    return (st[-1], f"step{st[-1]}.npz") if st else (None, None)


def alive(att):
    if not att or att.get("boot_id") != boot_id() or not att.get("pid"): return False
    try: os.kill(att["pid"], 0); return True
    except Exception: return False


class Runner:
    def __init__(self, root=None, concurrency=1, log=print):
        self.root = os.path.abspath(root or E.RUN_ROOT); self.conc = concurrency; self.log = log; os.makedirs(REG(self.root), exist_ok=True)
        self.lock = threading.Lock(); self.gpu_mem = {}; self.R = load_all(self.root); self.procs = {}; self.interrupting = False; self.finished = threading.Event()
        import hashlib; self.identity_sha = hashlib.sha256(json.dumps(E.identity(), sort_keys=True).encode()).hexdigest()   # review C r2 N2: identity of every attempt

    def _save(self, rec): self.R[rec["id"]] = rec; E.atomic_json(rec, rec_path(self.root, rec["id"]))

    def register(self, jobs):
        new = 0
        for j in jobs:
            old = self.R.get(j["id"])
            if old is not None:
                if [old["argv_norm"], old["env"], old["outputs"]] != [norm(j["argv"]), j["env"], j["outputs"]]:
                    raise RuntimeError(f"job {j['id']} changed definition after registration (integrity failure)")
                if old["argv"] != j["argv"]: old = dict(old, argv=j["argv"]); self._save(old)     # package relocated: same job
                continue
            rec = dict(j, argv_norm=norm(j["argv"]), status="pending", attempts=[], output_hashes={}, completed_steps=None, latest_ckpt=None, registered=time.time())
            if j["phase"] != "math" and j["outputs"]:
                d = os.path.join(self.root, os.path.dirname(j["outputs"][0]))
                pre = [f for f in (os.listdir(d) if os.path.isdir(d) and j["phase"] in TRAIN_PHASES + ("compose",) else []) if not f.startswith(".")]
                if pre or all(os.path.exists(os.path.join(self.root, o)) for o in j["outputs"]):
                    rec["status"] = "blocked"; rec["reason"] = "unregistered outputs present (not adopted): " + ",".join(pre[:5])
            self._save(rec); new += 1
        return new

    def _gpu_poll(self, stop):
        while not stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                for line in o.strip().splitlines():
                    pid, mem = [x.strip() for x in line.split(",")]; self.gpu_mem[int(pid)] = max(self.gpu_mem.get(int(pid), 0), float(mem))
            except Exception: pass
            stop.wait(15)

    def _run(self, rec):
        os.makedirs(os.path.join(self.root, "logs"), exist_ok=True); lg = os.path.join("logs", SAFE(rec["id"]) + f".{len(rec['attempts'])}.log")
        env = dict(os.environ, PYTHONPATH=E.PKG + os.pathsep + os.environ.get("PYTHONPATH", ""), COR_RUN_ROOT=self.root, **rec["env"])
        t0 = time.time(); att = dict(start=t0, device=os.environ.get("COR_DEVICE"), log=lg, boot_id=boot_id(), identity_sha256=self.identity_sha); rc, ru, p = None, None, None
        try:
            with open(os.path.join(self.root, lg), "w") as fo:
                p = subprocess.Popen(rec["argv"], cwd=self.root, env=env, stdout=fo, stderr=subprocess.STDOUT, start_new_session=True)
                att["pid"] = p.pid; self.procs[rec["id"]] = p
                with self.lock: r0 = dict(self.R[rec["id"]]); r0["attempts"] = r0["attempts"] + [dict(att, note="in progress")]; self._save(r0)
                _, status, ru = os.wait4(p.pid, 0); rc = os.waitstatus_to_exitcode(status)
        finally: self.procs.pop(rec["id"], None)
        att.update(end=time.time(), rc=rc, seconds=time.time() - t0, user_cpu_s=getattr(ru, "ru_utime", None), max_rss_mb=(ru.ru_maxrss / 1024 if ru else None), gpu_mem_mb_max=self.gpu_mem.get(att.get("pid")))
        with self.lock:
            rec = dict(self.R[rec["id"]]); rec["attempts"] = [a for a in rec["attempts"] if a.get("note") != "in progress"] + [att]
            outs_ok = all(os.path.exists(os.path.join(self.root, o)) and validate(os.path.join(self.root, o)) for o in rec["outputs"])
            if self.interrupting and rc != 0:
                att["note"] = "interrupted by the user (runner stopped); resumes from the saved state"; rec["status"] = "pending"
            elif rc == 0 and outs_ok:
                rec["status"] = "done"; rec["output_hashes"] = {o: E.sha256(os.path.join(self.root, o)) for o in rec["outputs"]}
                if rec["phase"] in TRAIN_PHASES and rec["phase"] != "pretrain":
                    d = os.path.join(self.root, os.path.dirname(rec["outputs"][0]))
                    rec["snapshot_hashes"] = {f: E.sha256(os.path.join(d, f)) for f in sorted(os.listdir(d)) if re.match(r"step\d+\.npz$", f)}
                pr = rec.get("prune")                      # storage: intermediate snapshots of non-anchor arms (hashes kept above)
                if pr:
                    d = os.path.join(self.root, pr["dir"]); rec["pruned"] = {}
                    for f in sorted(os.listdir(d)):
                        if re.match(r"step\d+\.npz$", f) and f not in pr["keep"]: rec["pruned"][f] = rec["snapshot_hashes"][f]; os.remove(os.path.join(d, f))
            else:
                rec["status"] = "failed"; rec["reason"] = f"rc={rc}" + ("" if outs_ok else "; missing or unreadable outputs: " + ",".join(o for o in rec["outputs"] if not (os.path.exists(os.path.join(self.root, o)) and validate(os.path.join(self.root, o)))))
            rec["completed_steps"], rec["latest_ckpt"] = progress(self.root, rec); self._save(rec)
        self.finished.set(); self.log(f"[{time.strftime('%H:%M:%S')}] {rec['status']:7s} {rec['id']} ({att['seconds']:.0f}s)")

    def reset_stale_running(self):
        n = 0
        for r in list(self.R.values()):
            if r["status"] == "running":
                live = [a for a in r["attempts"] if a.get("note") == "in progress" and alive(a)]
                if live: raise SystemExit(f"job {r['id']} is still running (pid {live[0]['pid']}) from an earlier runner on this machine: wait for it or restart the runtime")
                r = dict(r); r["attempts"] = [dict(a, note="interrupted (runtime restart or lost process)", rc=None) if a.get("note") == "in progress" else a for a in r["attempts"]]
                if not any(a.get("note", "").startswith("interrupted") for a in r["attempts"]): r["attempts"].append(dict(end=time.time(), rc=None, note="interrupted before the attempt was recorded"))
                r["status"] = "pending"; self._save(r); n += 1
        return n

    def run(self, planner, budget_hours=None, retry_failed=False, require_freeze=True, max_jobs=None, phases=None):
        if require_freeze:
            ok, bad = verify_freeze()
            if not ok: raise SystemExit("FREEZE VERIFICATION FAILED (refusing to run confirmation jobs): " + ", ".join(bad[:10]))
        lockf = os.path.join(self.root, "registry", "runner.lock"); L = E.read_json(lockf)
        if L and time.time() - L.get("heartbeat", 0) < 120: raise SystemExit(f"another runner is active on this run root (heartbeat {time.time() - L['heartbeat']:.0f}s ago)")
        n = self.reset_stale_running()
        if n: self.log(f"{n} interrupted job(s) reset to pending (training resumes from its exact saved state)")
        t_end = time.time() + 3600 * budget_hours if budget_hours else None; threads = {}; launched = 0; hb = 0; replan = True; self.interrupting = False
        stop = threading.Event(); threading.Thread(target=self._gpu_poll, args=(stop,), daemon=True).start()
        try:
            while True:
                if replan: self.register(planner()); replan = False
                for jid, th in list(threads.items()):
                    if not th.is_alive(): threads.pop(jid); replan = True
                ready = []
                for r in self.R.values():
                    if r["id"] in threads or (phases and r["phase"] not in phases): continue
                    if r["status"] == "pending" or (retry_failed and r["status"] == "failed" and len(r["attempts"]) < 3):
                        if all(self.R.get(d, {}).get("status") == "done" for d in r["deps"]): ready.append(r)
                ready.sort(key=lambda r: (r["priority"], r["id"]))
                stopping = (t_end and time.time() > t_end) or (max_jobs is not None and launched >= max_jobs)
                while ready and len(threads) < self.conc and not stopping:
                    r = dict(ready.pop(0)); r["status"] = "running"; self._save(r)
                    th = threading.Thread(target=self._run, args=(r,), daemon=True); th.start(); threads[r["id"]] = th; launched += 1
                    stopping = max_jobs is not None and launched >= max_jobs
                if not threads:
                    if replan: continue
                    if stopping: self.log("budget reached: stopped safely; all checkpoints preserved; remaining jobs stay pending")
                    break
                if time.time() - hb > 30: E.atomic_json(dict(heartbeat=time.time(), pid=os.getpid(), boot_id=boot_id()), lockf); hb = time.time()
                self.finished.wait(10); self.finished.clear()
        except BaseException:
            self.interrupting = True; self.log("interrupted: terminating running jobs (attempts recorded; they resume exactly next time)")
            for p in list(self.procs.values()):
                try: os.killpg(p.pid, signal.SIGTERM)
                except Exception: pass
            for th in list(threads.values()): th.join(timeout=120)
            raise
        finally:
            stop.set()
            if os.path.exists(lockf): os.remove(lockf)
        return summary(self.root)


def summary(root):
    R = load_all(root); out = {}
    for r in R.values():
        k = f"{r['track']}:{r['phase']}"; d = out.setdefault(k, {}); d[r["status"]] = d.get(r["status"], 0) + 1
    return dict(by_phase=out, n=len(R), failed=[r["id"] for r in R.values() if r["status"] == "failed"], blocked=[r["id"] for r in R.values() if r["status"] == "blocked"])
