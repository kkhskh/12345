"""XIV Stage I design (frozen in FREEZE_I): every training run, its command, and every (unit, step) measurement.  Writes logs/I_design.json
and the queues logs/queue_I_train.txt (training) and logs/queue_I_measure.txt (gated measurement).  Run once before the freeze."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
SIG = dict(E=2.5, M=1.25)                                   # location-matched noise sigma (TV-matched, logs/i_calib.json; FREEZE_I.md)
BASES = dict(E=dict(LW=21, real="ro0", realtag="ro0T100", sham="oq0", noise_edges="ro0", D=(21, 801, 802)),
             M=dict(LW=23, real="l2q", realtag="l2qT100", sham="rd2", noise_edges="own2", D=(23, 801, 802)))
H = "FX_SAVEOPT=1 FX_SAVEOPT_AT=1211"
def tagstem(b, D): lw = BASES[b]["LW"]; return f"uP160_fx_L{lw}_W{lw}_D{D}"
def basetag(b, D): return tagstem(b, D) + ("_base" if D in (21, 23) else "_stop1211_B_base")
def realtag(b, D): return tagstem(b, D) + (f"_{BASES[b]['realtag']}" if D in (21, 23) else f"_{BASES[b]['realtag']}_stop1211_B_{BASES[b]['real']}100")
runs, meas = [], []
def add(group, b, D, cond, env, args, tag, stop=None): runs.append(dict(group=group, base=b, D=D, cond=cond, tag=tag, cmd=f"{env} python3 scripts/ft_factorial_c.py uP160 {args}", stop=stop))
for b, c in BASES.items():
    lw = c["LW"]
    for D in c["D"]:
        a = f"{lw} {lw} {D}"; st = tagstem(b, D)
        add("IA", b, D, "sham", f"FX_STOP=1211 {H} FX_INTERV=msk:100 FX_EDGES={c['sham']} FX_TAGSUF=_XIV", a, f"{st}_msk{c['sham']}T100_stop1211_XIV", 1211)
        add("IA", b, D, "lnoise", f"FX_STOP=1211 {H} FX_INTERV=lnz:100 FX_EDGES={c['noise_edges']} FX_LSIG={SIG[b]} FX_TAGSUF=_XIV", a, f"{st}_lnz{c['noise_edges']}s{SIG[b]}T100_stop1211_XIV", 1211)
        # I-B late block, same operation and duration as the early block, branched from the baseline at 546
        add("IB", b, D, "late", f"FX_INTERV={c['real']}:0 FX_IWIN=546:646 FX_BRANCH={basetag(b, D)}:546 FX_EXTRA_SNAPS=646,696,846,1146,1757,2546 FX_TAGSUF=_XIV", a,
            f"{st}_{c['real']}T0_br546_w546-646_XIV")
        # step-matched baseline references at the late-recovery steps (bitwise replays of the baseline trajectory)
        add("IB", b, D, "bref_late", f"FX_STOP=1146 FX_BRANCH={basetag(b, D)}:546 FX_EXTRA_SNAPS=646,696,846,1146 FX_TAGSUF=_XIVbref", a, f"{st}_stop1146_br546_XIVbref", 1146)
        if D in (21, 23):
            add("IB", b, D, "bref_1757", f"FX_STOP=1757 FX_BRANCH={basetag(b, D)}:1500 FX_EXTRA_SNAPS=1757 FX_TAGSUF=_XIVbref", a, f"{st}_stop1757_br1500_XIVbref", 1757)
            add("IB", b, D, "bref_2546", f"FX_STOP=2546 FX_BRANCH={basetag(b, D)}:2000 FX_EXTRA_SNAPS=2546 FX_TAGSUF=_XIVbref", a, f"{st}_stop2546_br2000_XIVbref", 2546)
        elif D == 801:
            add("IB", b, D, "bref_tail", f"FX_STOP=2546 FX_BRANCH={basetag(b, D)}:1211 FX_EXTRA_SNAPS=1757,2546 FX_TAGSUF=_XIVbref", a, f"{st}_stop2546_br1211_XIVbref", 2546)
        else:   # D802: full baseline continuation (needed for the final step as well) and early-block continuation
            add("IB", b, D, "base_cont", f"FX_BRANCH={basetag(b, D)}:1211 FX_EXTRA_SNAPS=1757,2546 FX_TAGSUF=_XIVbase", a, f"{st}_br1211_XIVbase")
            add("IB", b, D, "early_cont", f"FX_BRANCH={realtag(b, D)}:1211 FX_TAGSUF=_XIVearly", a, f"{st}_br1211_XIVearly")
        if D == 801:   # I-A persistence: continue sham and noise to 4000
            add("IA", b, D, "sham_cont", f"FX_BRANCH={st}_msk{c['sham']}T100_stop1211_XIV:1211 FX_TAGSUF=_XIVsham", a, f"{st}_br1211_XIVsham")
            add("IA", b, D, "lnoise_cont", f"FX_BRANCH={st}_lnz{c['noise_edges']}s{SIG[b]}T100_stop1211_XIV:1211 FX_TAGSUF=_XIVlnoise", a, f"{st}_br1211_XIVlnoise")
def spec(tag):
    p = tag.split("_"); return f"{p[0]}:fx:{p[2][1:]}:{p[3][1:]}:{p[4][1:]}" + (":" + "_".join(p[5:]) if len(p) > 5 else "")
# measurement plan: (unit, step) lists; I-A at 1211 (+4000 for D801 continuations); I-B at the recovery steps
for b, c in BASES.items():
    for D in c["D"]:
        st = tagstem(b, D); bt, rt = basetag(b, D), realtag(b, D)
        for t in (bt, rt, f"{st}_msk{c['sham']}T100_stop1211_XIV", f"{st}_lnz{c['noise_edges']}s{SIG[b]}T100_stop1211_XIV"): meas.append(dict(unit=spec(t), step=1211))
        if D == 801:
            for t in (f"{st}_br1211_XIVsham", f"{st}_br1211_XIVlnoise", f"{st}_br1211_B_{c['real']}100", f"{st}_br1211_B_base"): meas.append(dict(unit=spec(t), step=4000))
        # early block: release at 100 -> recovery offsets 0,50,200,500,1111,1900 -> steps 100,150,300,600,1211,2000; final 4000
        early_src = {100: rt, 150: rt, 300: rt, 600: rt, 1211: rt}
        early_tail = rt if D in (21, 23) else (f"{st}_br1211_B_{c['real']}100" if D == 801 else f"{st}_br1211_XIVearly")
        base_tail = bt if D in (21, 23) else (f"{st}_br1211_B_base" if D == 801 else f"{st}_br1211_XIVbase")
        for k in (100, 150, 300, 600, 1211): meas += [dict(unit=spec(rt), step=k), dict(unit=spec(bt), step=k)]
        for k in (2000, 4000): meas += [dict(unit=spec(early_tail), step=k), dict(unit=spec(base_tail), step=k)]
        # late block: release at 646 -> steps 646,696,846,1146,1757,2546; final 4000
        lt = f"{st}_{c['real']}T0_br546_w546-646_XIV"
        for k in (646, 696, 846, 1146, 1757, 2546, 4000): meas.append(dict(unit=spec(lt), step=k))
        for k in (646, 696, 846, 1146): meas.append(dict(unit=spec(f"{st}_stop1146_br546_XIVbref"), step=k))
        if D in (21, 23): meas += [dict(unit=spec(f"{st}_stop1757_br1500_XIVbref"), step=1757), dict(unit=spec(f"{st}_stop2546_br2000_XIVbref"), step=2546)]
        elif D == 801: meas += [dict(unit=spec(f"{st}_stop2546_br1211_XIVbref"), step=1757), dict(unit=spec(f"{st}_stop2546_br1211_XIVbref"), step=2546)]
        else: meas += [dict(unit=spec(base_tail), step=1757), dict(unit=spec(base_tail), step=2546)]
seen = set(); meas = [m for m in meas if not ((m["unit"], m["step"]) in seen or seen.add((m["unit"], m["step"])))]
D = dict(sigma=SIG, bases=BASES, runs=runs, measurements=meas, measure_env="UM_RHO=0 UM_STREAM=95300000 UM_TAG=_I",
         recovery_offsets=[0, 50, 200, 500, 1111, 1900], early_release=100, late_release=646)
json.dump(D, open(os.path.join(L, "I_design.json"), "w"), indent=1)
# training queue: I-A short runs and late runs first (longest), then references and continuations (continuations gated on their sources)
order = sorted(runs, key=lambda r: {"late": 0, "sham": 1, "lnoise": 1, "early_cont": 2, "base_cont": 2, "bref_late": 3, "bref_tail": 3, "bref_1757": 4, "bref_2546": 4, "sham_cont": 5, "lnoise_cont": 5}[r["cond"]])
with open(os.path.join(L, "queue_I_train.txt"), "w") as f:
    for r in order:
        if r["cond"] in ("sham_cont", "lnoise_cont"):
            src = r["cmd"].split("FX_BRANCH=")[1].split(":")[0]; f.write(f"bash scripts/when_file.sh checkpoints/ft/{src}/stop_done.json -- {r['cmd']}\n")
        else: f.write(r["cmd"] + "\n")
print(len(runs), "runs;", len(meas), "measurements")
