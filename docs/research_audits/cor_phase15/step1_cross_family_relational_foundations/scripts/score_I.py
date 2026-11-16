"""XIV Stage I frozen scorer (FAIL-CLOSED).  Usage: score_I.py [LOGDIR=logs]
Refuses (exit 2, writes nothing) unless: the frozen files match logs/FREEZE_I_HASHES.txt; every design run has finished (stop_done.json or
final.json in its checkpoint directory); every (unit, step) measurement in the design is present exactly once in unit_metrics_I.jsonl;
logs/i_sham_validity.json exists.  Verdicts:
  R-S (off-route sham) and R-N (location-matched noise), each over the 6 base x data-order pairs at step 1211:
    CONFIRMED (route-specific beyond this control): real flips >= 5/6 competent, control flips <= 1/6, and the paired mean of
      delta_real - delta_control >= 0.4 with the 95% bootstrap lower bound > 0;
    FALSIFIED: control flips >= 3/6, or mean delta_control >= 0.5 * mean delta_real;   otherwise INCONCLUSIVE.
    R-S is reported as SHAM INVALID (never CONFIRMED) if any sham fails the frozen validity check.
  R-P (early changes more persistent), over the 6 pairs: persistence index PI = delta(r=1900) / max(delta(r) : r in {0,50,200,500}),
    defined only if that max >= 0.1 and the run is competent at 4000 (KL <= 0.02);
    CONFIRMED: >= 4 pairs with both PI defined, PI_early > PI_late in all but at most one, mean difference >= 0.25 and bootstrap lower
      bound > 0;  FALSIFIED: >= 4 valid pairs and (mean difference <= 0.05 or PI_early > PI_late in at most half);  else INCONCLUSIVE.
delta = organisation shift relative to the step-matched baseline, oriented towards the intervention's predicted direction:
  E base: mix_mass(run) - mix_mass(base);  M base: mix_mass(base) - mix_mass(run)."""
import json, os, sys, hashlib
import numpy as np, optax
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG, "logs")
def refuse(msg): print("REFUSED:", msg); sys.exit(2)
for l in open(os.path.join(PKG, "logs", "FREEZE_I_HASHES.txt")):
    h, p = l.split(); 
    if hashlib.sha256(open(os.path.join(PKG, p), "rb").read()).hexdigest() != h: refuse(f"frozen file changed: {p}")
D = json.load(open(os.path.join(LD, "I_design.json")))
for r in D["runs"]:
    d = os.path.join(os.environ.get("SCORE_CKPT", os.path.join(PKG, "checkpoints", "ft")), r["tag"])
    if not os.path.exists(os.path.join(d, "stop_done.json" if r["stop"] else "final.json")): refuse(f"run not finished: {r['tag']}")
mf = os.path.join(LD, "unit_metrics_I.jsonl")
if not os.path.exists(mf): refuse("no unit_metrics_I.jsonl")
U = {}
for l in open(mf):
    x = json.loads(l); k = (x["unit"], x["step"])
    if k in U: refuse(f"duplicate measurement {k}")
    U[k] = x
miss = [(m["unit"], m["step"]) for m in D["measurements"] if (m["unit"], m["step"]) not in U]
if miss: refuse(f"{len(miss)} measurements missing, e.g. {miss[:3]}")
if not os.path.exists(os.path.join(LD, "i_sham_validity.json")): refuse("no i_sham_validity.json")
V = json.load(open(os.path.join(LD, "i_sham_validity.json")))
def spec(tag):
    p = tag.split("_"); return f"{p[0]}:fx:{p[2][1:]}:{p[3][1:]}:{p[4][1:]}" + (":" + "_".join(p[5:]) if len(p) > 5 else "")
def cls1211(x): return "EARLY" if (x["mix_mass"] < 0.5 and x["lik_RO1"] >= 0.4) else ("MIXTURE" if (x["mix_mass"] >= 0.8 and x["lik_RO1"] <= 0.15) else "HYBRID")
def cls4000(x): return "EARLY" if (x["lik_RO1"] >= 0.6 and x["mix_mass"] < 0.5) else ("MIXTURE" if (x["mix_mass"] >= 0.8 and x["lik_RO1"] <= 0.15) else "HYBRID")
def g(tag, k): return U[(spec(tag), k)]
def delta(b, run, base): return (run["mix_mass"] - base["mix_mass"]) if b == "E" else (base["mix_mass"] - run["mix_mass"])
def dlik(b, run, base): return (base["lik_RO1"] - run["lik_RO1"]) if b == "E" else (run["lik_RO1"] - base["lik_RO1"])
rng = np.random.default_rng(0)
def boot(v):
    v = np.asarray(v, float); I = rng.integers(0, len(v), (10000, len(v))); m = v[I].mean(1); return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
B = D["bases"]; out = dict(IA={}, IB={}, verdicts={})
def stem(b, Dd): lw = B[b]["LW"]; return f"uP160_fx_L{lw}_W{lw}_D{Dd}"
def basetag(b, Dd): return stem(b, Dd) + ("_base" if Dd in (21, 23) else "_stop1211_B_base")
def realtag(b, Dd): return stem(b, Dd) + (f"_{B[b]['realtag']}" if Dd in (21, 23) else f"_{B[b]['realtag']}_stop1211_B_{B[b]['real']}100")
# ---------------- I-A
rows = []
for b in ("E", "M"):
    for Dd in B[b]["D"]:
        st = stem(b, Dd); bx = g(basetag(b, Dd), 1211); row = dict(base=b, D=Dd, base_class=cls1211(bx), base_KL=bx["KL"])
        for cond, t in (("real", realtag(b, Dd)), ("sham", f"{st}_msk{B[b]['sham']}T100_stop1211_XIV"), ("lnoise", f"{st}_lnz{B[b]['noise_edges']}s{D['sigma'][b]}T100_stop1211_XIV")):
            x = g(t, 1211); c = cls1211(x); comp = x["KL"] <= 0.05
            row[cond] = dict(cls=c, competent=comp, flip=bool(comp and c != row["base_class"]), delta=delta(b, x, bx), dlik=dlik(b, x, bx), mix=x["mix_mass"], lik=x["lik_RO1"], KL=x["KL"], acc=x["acc"],
                             route_Q=x.get("route_Q"), route_R=x.get("route_R"), l2_attn=x.get("l2_attn"), l2_mlp=x.get("l2_mlp"))
        rows.append(row)
out["IA"]["rows"] = rows
nreal = sum(r["real"]["flip"] for r in rows)
for name, cond in (("R-S", "sham"), ("R-N", "lnoise")):
    nc = sum(r[cond]["flip"] for r in rows); diffs = [r["real"]["delta"] - r[cond]["delta"] for r in rows]; m, lo, hi = boot(diffs)
    mr, mc = np.mean([r["real"]["delta"] for r in rows]), np.mean([r[cond]["delta"] for r in rows])
    if nc >= 3 or mc >= 0.5 * mr: v = "FALSIFIED"
    elif nreal >= 5 and nc <= 1 and m >= 0.4 and lo > 0: v = "CONFIRMED"
    else: v = "INCONCLUSIVE"
    if cond == "sham" and not V.get("all_valid"): v = "SHAM INVALID"
    out["verdicts"][name] = v; out["IA"][name] = dict(real_flips=nreal, control_flips=nc, mean_delta_real=float(mr), mean_delta_control=float(mc), paired_diff_mean=m, ci95=[lo, hi])
pers = []
for b in ("E", "M"):
    st = stem(b, 801); bx = g(f"{st}_br1211_B_base", 4000); r = dict(base=b, base_class=cls4000(bx))
    for cond, t in (("real", f"{st}_br1211_B_{B[b]['real']}100"), ("sham", f"{st}_br1211_XIVsham"), ("lnoise", f"{st}_br1211_XIVlnoise")):
        x = g(t, 4000); r[cond] = dict(cls=cls4000(x), delta=delta(b, x, bx), KL=x["KL"], mix=x["mix_mass"], lik=x["lik_RO1"])
    pers.append(r)
out["IA"]["persistence_D801_4000"] = pers
# ---------------- I-B
sched = optax.warmup_cosine_decay_schedule(0.0, 1e-3, 200, 4000); lr = np.array([float(sched(i)) for i in range(4000)])
offs = D["recovery_offsets"]; prs = []
for b in ("E", "M"):
    for Dd in B[b]["D"]:
        st = stem(b, Dd); bt, rt = basetag(b, Dd), realtag(b, Dd); lt = f"{st}_{B[b]['real']}T0_br546_w546-646_XIV"
        e_tail = rt if Dd in (21, 23) else (f"{st}_br1211_B_{B[b]['real']}100" if Dd == 801 else f"{st}_br1211_XIVearly")
        b_tail = bt if Dd in (21, 23) else (f"{st}_br1211_B_base" if Dd == 801 else f"{st}_br1211_XIVbase")
        def ebase(k): return g(bt, k) if k <= 1211 else g(b_tail, k)
        def erun(k): return g(rt, k) if k <= 1211 else g(e_tail, k)
        def lbase(k):
            if k <= 1146: return g(f"{st}_stop1146_br546_XIVbref", k)
            if Dd in (21, 23): return g(f"{st}_stop1757_br1500_XIVbref" if k == 1757 else f"{st}_stop2546_br2000_XIVbref", k)
            return g(f"{st}_stop2546_br1211_XIVbref" if Dd == 801 else b_tail, k)
        E_ = {r: delta(b, erun(100 + r), ebase(100 + r)) for r in offs}; L_ = {r: delta(b, g(lt, 646 + r), lbase(646 + r)) for r in offs}
        Ef, Lf = g(e_tail, 4000), g(lt, 4000); bf = g(b_tail, 4000)
        def PI(dd, comp):
            mx = max(dd[r] for r in (0, 50, 200, 500)); return (dd[1900] / mx) if (mx >= 0.1 and comp) else None
        ce, cl = Ef["KL"] <= 0.02, Lf["KL"] <= 0.02
        prs.append(dict(base=b, D=Dd, early=dict(delta=E_, delta_final=delta(b, Ef, bf), cls_final=cls4000(Ef), KL_final=Ef["KL"], competent=ce, PI=PI(E_, ce),
                                                  dlik={r: dlik(b, erun(100 + r), ebase(100 + r)) for r in offs}),
                        late=dict(delta=L_, delta_final=delta(b, Lf, bf), cls_final=cls4000(Lf), KL_final=Lf["KL"], competent=cl, PI=PI(L_, cl),
                                  dlik={r: dlik(b, g(lt, 646 + r), lbase(646 + r)) for r in offs}),
                        base_cls_final=cls4000(bf),
                        lr_mass_block=dict(early=float(lr[0:100].sum()), late=float(lr[546:646].sum())),
                        lr_mass_post_release={str(r): dict(early=float(lr[100:100 + r].sum()), late=float(lr[646:646 + r].sum())) for r in offs},
                        lr_mass_to_final=dict(early=float(lr[100:].sum()), late=float(lr[646:].sum()))))
out["IB"]["pairs"] = prs
valid = [p for p in prs if p["early"]["PI"] is not None and p["late"]["PI"] is not None]
if len(valid) >= 4:
    d = [p["early"]["PI"] - p["late"]["PI"] for p in valid]; m, lo, hi = boot(d); npos = sum(x > 0 for x in d)
    if m <= 0.05 or npos <= len(d) / 2: v = "FALSIFIED"
    elif npos >= len(d) - 1 and m >= 0.25 and lo > 0: v = "CONFIRMED"
    else: v = "INCONCLUSIVE"
    out["IB"]["R-P"] = dict(valid_pairs=len(valid), n_early_more=int(npos), mean_PI_diff=m, ci95=[lo, hi])
else: v = "INCONCLUSIVE"; out["IB"]["R-P"] = dict(valid_pairs=len(valid))
out["verdicts"]["R-P"] = v
json.dump(out, open(os.path.join(LD, "score_I.json"), "w"), indent=1, default=float)
print(json.dumps(out["verdicts"]))
