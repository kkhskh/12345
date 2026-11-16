"""XIV Stage I preflight (synthetic data only; no confirmation output is read).  Checks that score_I.py
 P0 refuses on the real (incomplete) state; P1 confirms a synthetic route-specific + persistent scenario; P2 falsifies when the controls flip;
 P3 refuses when one measurement is missing; P4 reports SHAM INVALID when a sham fails validity; P5 falsifies R-P when late = early.
Writes logs/preflight_I.json; exit 0 only if every check passes."""
import json, os, sys, subprocess, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); D = json.load(open(os.path.join(PKG, "logs", "I_design.json")))
def run(ld, ck):
    p = subprocess.run([sys.executable, os.path.join(HERE, "score_I.py"), ld], capture_output=True, text=True, env=dict(os.environ, SCORE_CKPT=ck))
    return p.returncode, (json.load(open(os.path.join(ld, "score_I.json"))) if os.path.exists(os.path.join(ld, "score_I.json")) else None)
def synth(ctrl_flip=False, late_persist=False, valid=True, drop=False):
    ld, ck = tempfile.mkdtemp(), tempfile.mkdtemp(); shutil.copy(os.path.join(PKG, "logs", "I_design.json"), ld)
    for r in D["runs"]: os.makedirs(os.path.join(ck, r["tag"])); open(os.path.join(ck, r["tag"], "stop_done.json" if r["stop"] else "final.json"), "w").write("{}")
    with open(os.path.join(ld, "unit_metrics_I.jsonl"), "w") as f:
        for i, m in enumerate(D["measurements"]):
            if drop and i == 7: continue
            u = m["unit"]; E = ":21:21:" in u; flipped = False
            if "ro0T100" in u or "l2qT100" in u or "XIVearly" in u or "B_ro0100" in u or "B_l2q100" in u: flipped = m["step"] >= 150 or m["step"] == 100
            if "msk" in u or "lnz" in u or "XIVsham" in u or "XIVlnoise" in u: flipped = ctrl_flip
            frac = 0.0
            if "w546-646" in u: flipped = True; frac = 1.0 if (late_persist or m["step"] <= 1146) else 0.25
            elif flipped: frac = 1.0
            mix = (0.17 + 0.78 * frac) if E else (0.95 - 0.8 * frac); lik = (0.8 - 0.8 * frac) if E else (0.05 + 0.7 * frac)
            f.write(json.dumps(dict(unit=u, step=m["step"], mix_mass=mix, lik_RO1=lik, KL=0.01, acc=0.88)) + "\n")
    json.dump(dict(all_valid=valid), open(os.path.join(ld, "i_sham_validity.json"), "w")); return ld, ck
res = {}
rc, _ = run(os.path.join(PKG, "logs"), os.path.join(PKG, "checkpoints", "ft")); res["P0_refuse_real_state"] = rc == 2
rc, o = run(*synth()); res["P1_confirm"] = rc == 0 and o["verdicts"] == {"R-S": "CONFIRMED", "R-N": "CONFIRMED", "R-P": "CONFIRMED"}
rc, o = run(*synth(ctrl_flip=True)); res["P2_controls_flip_falsify"] = rc == 0 and o["verdicts"]["R-S"] == "FALSIFIED" and o["verdicts"]["R-N"] == "FALSIFIED"
rc, o = run(*synth(drop=True)); res["P3_missing_refuse"] = rc == 2 and o is None
rc, o = run(*synth(valid=False)); res["P4_sham_invalid"] = rc == 0 and o["verdicts"]["R-S"] == "SHAM INVALID"
rc, o = run(*synth(late_persist=True)); res["P5_equal_persistence_falsify"] = rc == 0 and o["verdicts"]["R-P"] == "FALSIFIED"
json.dump(res, open(os.path.join(PKG, "logs", "preflight_I.json"), "w"), indent=1); print(res); sys.exit(0 if all(res.values()) else 1)
