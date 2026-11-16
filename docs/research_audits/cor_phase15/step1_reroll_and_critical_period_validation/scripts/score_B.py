"""XIII-E Workstream B confirmation scorer (frozen in freeze B).  FAIL-CLOSED: refuses (exit 2, writes nothing) unless every run, checkpoint
and metric listed by logs/B_design.json is present and consistent.  Usage: score_B.py [LOGDIR] [CKPTDIR] -> LOGDIR/score_B.json
Classes: at 1211 EARLY iff mix<0.5 and lik>=0.4; MIXTURE iff mix>=0.8 and lik<=0.15; else HYBRID.  At 4000: early iff lik>=0.6 and mix<0.5;
mixture iff mix>=0.8 and lik<=0.15; else hybrid.  Competent: KL<=0.05 at 1211, KL<=0.02 at 4000.  A FLIP is a class different from the
D-matched no-mask baseline at the same step; non-competent runs are excluded from flip counts (reported).
R3 (early mask specificity): for each base b in {E, M}: f_real = flips of the 100-step real route mask over the 3 D seeds, f_dist = flips of
   the severity-matched noise control, sham_ok = the D=801 sham run is bitwise identical to the D=801 baseline at 1211.
   CONFIRMED iff for both bases f_real >= 2, f_dist <= 1, sham_ok and >= 2 competent seeds per arm; FALSIFIED iff for either base f_real <= 1
   or f_dist >= 2 (with >= 2 competent seeds); otherwise INCONCLUSIVE.
R4 (specifically early selection vs general disruption): f_late = flips at 4000 over the 8 severity-matched late runs (2 bases x 2 conditions x
   2 D) relative to the D-matched baseline at 4000; f_strong = flips at 1211 of the strong noise control on the mixture base (3 D);
   dose = flips(25) <= flips(100), pooled over bases.  CONFIRMED iff f_late <= 2 and f_strong <= 1 and dose; FALSIFIED iff f_late >= 4 or
   f_strong >= 2; otherwise INCONCLUSIVE."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs")
CK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "checkpoints", "ft")
import numpy as np
def refuse(m): print("REFUSED:", m, file=sys.stderr); sys.exit(2)
try: D = json.load(open(os.path.join(LD, "B_design.json")))
except Exception as ex: refuse(f"design unreadable {ex}")
def jl(f):
    p = os.path.join(LD, f)
    if not os.path.exists(p): refuse(f"missing {f}")
    o = {}
    for l in open(p):
        try: x = json.loads(l)
        except Exception: refuse(f"malformed line in {f}")
        o[(x["unit"], x["step"])] = x
    return o
for r in D["runs"]:
    d = os.path.join(CK, r["tag"])
    need = ["stop_done.json", "step1211.npz"] if r["kind"] == "short" else ["final.json", "step4000.npz"]
    for f in need:
        if not os.path.exists(os.path.join(d, f)): refuse(f"missing {r['tag']}/{f}")
cont_spec = {}
for t in D["continuation"]:
    r = next(x for x in D["runs"] if x["tag"] == t); ct = f'{r["parent"]}_fx_L{r["L"]}_W{r["W"]}_D{r["D"]}_br1211{r["suffix"]}'
    if not os.path.exists(os.path.join(CK, ct, "final.json")): refuse(f"continuation {ct} not finished")
    cont_spec[t] = f'{r["parent"]}:fx:{r["L"]}:{r["W"]}:{r["D"]}:br1211{r["suffix"]}'
U1, U4 = jl("unit_metrics_B.jsonl"), jl("unit_metrics_Bfinal.jsonl")
for r in D["runs"]:
    key = (r["spec"], 1211) if r["kind"] == "short" else (r["spec"], 4000); src = U1 if r["kind"] == "short" else U4
    if key not in src: refuse(f"missing metrics {key}")
    f = src[key].get("factors")
    if f is None or [f[0], f[1], f[2], f[3]] != [r["parent"], r["L"], r["W"], r["D"]]: refuse(f"configuration mismatch {r['spec']}")
for b, sp in D["anchor_baselines"].items():
    if (sp, 1211) not in U1 or (sp, 4000) not in U4: refuse(f"missing anchor baseline metrics {sp}")
for t, sp in cont_spec.items():
    if (sp, 4000) not in U4: refuse(f"missing continuation metrics {sp}")
def c1(m): return "EARLY" if (m["mix_mass"] < 0.5 and m["lik_RO1"] >= 0.4) else ("MIXTURE" if (m["mix_mass"] >= 0.8 and m["lik_RO1"] <= 0.15) else "HYBRID")
def c4(m): return "early" if (m["lik_RO1"] >= 0.6 and m["mix_mass"] < 0.5) else ("mixture" if (m["mix_mass"] >= 0.8 and m["lik_RO1"] <= 0.15) else "hybrid")
anchorD = {b: D["D_seeds"][b][0] for b in "EM"}
def base_metrics(b, Dv, step):
    if Dv == anchorD[b]: return (U1 if step == 1211 else U4)[(D["anchor_baselines"][b], step)]
    r = next(x for x in D["runs"] if x["base"] == b and x["D"] == Dv and x["cond"] == "baseline")
    return U1[(r["spec"], 1211)] if step == 1211 else U4[(cont_spec[r["tag"]], 4000)]
rows = []
for r in D["runs"]:
    if r["cond"] == "baseline": continue
    st = 1211 if r["kind"] == "short" else 4000; m = (U1 if st == 1211 else U4)[(r["spec"], st)]; bm = base_metrics(r["base"], r["D"], st)
    cl, bc = (c1(m), c1(bm)) if st == 1211 else (c4(m), c4(bm)); comp = m["KL"] <= (0.05 if st == 1211 else 0.02)
    rows.append(dict(base=r["base"], D=r["D"], cond=r["cond"], step=st, cls=cl, base_cls=bc, flip=cl != bc, competent=comp, lik=m["lik_RO1"], mix=m["mix_mass"], KL=m["KL"],
                     d_lik=m["lik_RO1"] - bm["lik_RO1"], d_mix=m["mix_mass"] - bm["mix_mass"]))
def flips(b, cond):
    rs = [x for x in rows if x["base"] == b and x["cond"] == cond]; c = [x for x in rs if x["competent"]]; return sum(x["flip"] for x in c), len(c), len(rs)
sham_ok = {}
for b in "EM":
    s = next(x for x in D["runs"] if x["base"] == b and x["cond"] == "sham100"); bl = next(x for x in D["runs"] if x["base"] == b and x["D"] == s["D"] and x["cond"] == "baseline")
    a, c = np.load(os.path.join(CK, s["tag"], "step1211.npz")), np.load(os.path.join(CK, bl["tag"], "step1211.npz"))
    sham_ok[b] = max(float(np.abs(a[k] - c[k]).max()) for k in a.files) == 0.0
R3a = {b: dict(real100=flips(b, "real100"), dist100=flips(b, "dist100"), sham_bitwise=sham_ok[b]) for b in "EM"}
if all(R3a[b]["real100"][1] >= 2 and R3a[b]["dist100"][1] >= 2 for b in "EM"):
    if all(R3a[b]["real100"][0] >= 2 and R3a[b]["dist100"][0] <= 1 and R3a[b]["sham_bitwise"] for b in "EM"): v3 = "CONFIRMED"
    elif any(R3a[b]["real100"][0] <= 1 or R3a[b]["dist100"][0] >= 2 for b in "EM"): v3 = "FALSIFIED"
    else: v3 = "INCONCLUSIVE"
else: v3 = "INCONCLUSIVE"
late = [x for x in rows if x["step"] == 4000]; lc = [x for x in late if x["competent"]]; f_late = sum(x["flip"] for x in lc)
fs = flips("M", "strongdist100"); dose = sum(flips(b, "real25")[0] for b in "EM") <= sum(flips(b, "real100")[0] for b in "EM")
v4 = "CONFIRMED" if (f_late <= 2 and fs[0] <= 1 and dose and len(lc) >= 6) else ("FALSIFIED" if (f_late >= 4 or fs[0] >= 2) else "INCONCLUSIVE")
window = {b: {w: flips(b, f"real{w}") for w in (25, 50, 100)} for b in "EM"}
stab = []
for t, sp in cont_spec.items():
    r = next(x for x in D["runs"] if x["tag"] == t); m1 = U1[(r["spec"], 1211)]; m4 = U4[(sp, 4000)]; stab.append(dict(tag=t, cls1211=c1(m1), cls4000=c4(m4), KL4000=m4["KL"]))
res = dict(rows=rows, R3=dict(arms=R3a, verdict=v3), R4=dict(f_late=f_late, n_late_competent=len(lc), f_strong=fs, dose=dose, verdict=v4), windows=window, stability=stab,
           summary=dict(R3=v3, R4=v4))
json.dump(res, open(os.path.join(LD, "score_B.json"), "w"), indent=1, default=float); print(res["summary"])
