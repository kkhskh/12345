"""XIII-D section 11: full-length (4000-step) independent replays compared with the original runs, at every common checkpoint.
Pairs: (replay dir, original dir).  Compares every parameter leaf (max |diff|, bitwise flag), the logged training-loss history, the final
evaluation summary (acc, KL), and -- through unit_metrics -- the mechanistic classification measures (lik_RO1, mix_mass, route_Q) on the
confirmation stream used for both.  -> logs/determinism_check.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import xc, native as NV, metrics as MT, ftlib as F
PKG = os.path.join(HERE, ".."); X13C = NV.X13C
PAIRS = [("uP163_fx_L21_W22_D21_rep", os.path.join(X13C, "checkpoints", "ft", "uP163_fx_L21_W22_D21"), "uP163:fx:21:22:21:rep", "uP163:fx:21:22:21"),
         ("uP160_fx_L23_W23_D23_base", F.model_dir("uP160_full_c1_s4000_seed23_eval_conf13"), "uP160:fx:23:23:23:base", "uP160:s:23"),
         ("uP160_fx_L21_W21_D21_base", F.model_dir("uP160_full_c1_s4000_seed21_eval_conf13"), "uP160:fx:21:21:21:base", "uP160:s:21")]
out = {}; S = MT.sets(88_100_000)
for rep, orig, sr, so in PAIRS:
    rd = os.path.join(PKG, "checkpoints", "ft", rep)
    if not os.path.exists(os.path.join(rd, "final.json")): out[rep] = "replay not finished"; continue
    steps = sorted(set(int(f[4:-4]) for f in os.listdir(rd) if f.startswith("step")) & set(int(f[4:-4]) for f in os.listdir(orig) if f.startswith("step")))
    per = {}
    for s in steps:
        a, b = np.load(os.path.join(rd, f"step{s}.npz")), np.load(os.path.join(orig, f"step{s}.npz"))
        per[s] = max(float(np.abs(a[k] - b[k]).max()) for k in a.files)
    fa, fb = json.load(open(os.path.join(rd, "final.json")))["summary"], json.load(open(os.path.join(orig, "final.json")))["summary"]
    ca = json.load(open(os.path.join(rd, "curve.json"))); cb_p = os.path.join(orig, "curve.json"); cb = json.load(open(cb_p)) if os.path.exists(cb_p) else {}
    ha = {h[0]: h[1] for h in ca.get("hist", [])}; hb = {h[0]: h[1] for h in (cb.get("hist") or cb.get("curve_hist") or [])} if cb else {}
    common = sorted(set(ha) & set(hb)); loss_diff = max((abs(ha[k] - hb[k]) for k in common), default=None)
    ma, mb = MT.all_metrics(xc.load(sr), S, rho=False, l2=False), MT.all_metrics(xc.load(so), S, rho=False, l2=False)
    out[rep] = dict(common_steps=steps, max_abs_param_diff_per_step=per, bitwise_all_steps=all(v == 0.0 for v in per.values()),
                    final_acc=(fa["acc"], fb["acc"]), final_KL=(fa["KL"], fb["KL"]), loss_hist_points_compared=len(common), loss_hist_max_diff=loss_diff,
                    mechanistic={k: (ma[k], mb[k]) for k in ("lik_RO1", "mix_mass", "route_Q", "KL", "acc")})
    print(rep, "bitwise all", out[rep]["bitwise_all_steps"], "n steps", len(steps), "final acc", out[rep]["final_acc"], "loss pts", len(common), loss_diff, flush=True)
json.dump(out, open(os.path.join(PKG, "logs", "determinism_check.json"), "w"), indent=1, default=float)
