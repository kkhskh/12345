"""Apply the frozen criteria (PROTOCOL.md §6) and the addendum criteria to confirmation-split results.
Writes logs/verdicts.json and VERDICTS.md."""
import glob, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..")
def ld(p):
    try: return json.load(open(p))
    except Exception: return None
def ok(x): return x is not None and not (isinstance(x, float) and math.isnan(x))
def ge(a, b): return ok(a) and ok(b) and a >= b
def le(a, b): return ok(a) and ok(b) and a <= b


def frozen(r):
    s = r["s_star"]; tg = r["selection_target"]; t = r["tests"]; b = r["behaviour"]; pr, pi = r["probes"], r["probes_random_init"]
    c = {}
    c["P0"] = ge(b["acc"], b["own_question_bayes_acc"] + 0.5 * (b["bayes_acc"] - b["own_question_bayes_acc"]))
    r2, r2i = pr[f"{s}|readout|{tg}"], pi[f"{s}|readout|{tg}"]
    c["P1"] = ge(r2, 0.6) and ge(r2 - r2i, 0.3)
    d = t["D1_probe"]
    c["P2"] = ge(d["cond"]["iia"], 0.7) and ge(d["cond"]["iia"], t["random"]["cond"]["iia"] + 0.3) and ge(d["cond"]["kl_recovered"], 0.5)
    c["P3"] = ge(d["perm"]["iia"], 0.6)
    c["P4"] = le(d["target"]["change_rate"], 0.10)
    df = t["necessity"]["drop_frac"]
    c["P5"] = ge(df.get("D1_probe"), 0.5) and le(df.get("random_mean"), 0.2)
    c["P6"] = ("D1_fit_q012" in t) and ge(t["D1_fit_q012"]["cond_q345"]["iia"], 0.7 * d["cond"]["iia"])
    ct = r["content_D2"]; rel = max(ct.get("pi_minus", -1), ct.get("counts", -1))
    c["P7"] = ge(rel, ct["hist"] + 0.2)
    att = r["attention"]
    L = len([k for k in att if k.endswith("_pattern_patch_all")])
    route = any(ge(att[f"layer{l}_pattern_patch_all"]["iia"], 0.7) and ge(att[f"layer{l}_pattern_patch_all"]["kl_recovered"], 0.5) for l in range(L))
    gm = max([max(v) for k, v in att.items() if k.endswith("per_head")] or [float("nan")])
    c["TypeB"] = route and ge(gm, 0.5)
    c["TypeA"] = all(c[f"P{i}"] for i in range(8))
    c["P6_gated"] = c["P6"] and c["P2"]          # review fix: reuse is only meaningful when the base effect exists
    sw = t["site_location_sweep_cond"]
    last = r["sites"][-1]
    c["Distributed_literal"] = (not c["TypeA"]) and (not c["P2"]) and any(ge(v["iia"], 0.9) for k, v in sw.items() if "|full" in k and "episode" not in k)
    c["Distributed_nonfinal"] = (not c["TypeA"]) and (not c["P2"]) and any(ge(v["iia"], 0.9) for k, v in sw.items() if "|full" in k and "episode" not in k and not k.startswith(last))
    nums = dict(acc=b["acc"], bayes=b["bayes_acc"], own=b["own_question_bayes_acc"], s_star=s, probe_r2=r2, probe_r2_init=r2i,
                D1_cond_iia=d["cond"]["iia"], D1_cond_kl=d["cond"]["kl_recovered"], D1_perm_iia=d["perm"]["iia"],
                D1_target_change=d["target"]["change_rate"], D1_dropfrac=df.get("D1_probe"),
                D2_cond_iia=t["D2_contrast"]["cond"]["iia"], D2_target_change=t["D2_contrast"]["target"]["change_rate"],
                random_cond_iia=t["random"]["cond"]["iia"], content=ct, max_group_mass_r2=gm,
                best_pattern_patch=max((att[f"layer{l}_pattern_patch_all"]["iia"] for l in range(L)), default=float("nan")),
                best_nonfinal_full=max([v["iia"] for k, v in sw.items() if "|full" in k and "episode" not in k and not k.startswith(last)] or [float("nan")]))
    return c, nums


def tagc(r, fz_typeB):
    t = r["tests"]; pr = r["tag_probes"]; s = r["s_tag"]; c = {}
    c["K1"] = ge(pr[f"{s}|loo|trained"], 0.6) and ge(pr[f"{s}|loo|trained"] - pr[f"{s}|loo|init"], 0.3)
    ct = r["content_D2_tag"]
    c["K7"] = ge(ct["tag_target"], ct["other_question_histograms"] + 0.2) and ge(ct["tag_target"], ct["own_value_onehot"] + 0.2)
    df = t["necessity"]["drop_frac"]
    for suf, nm, reuse, rs in [("", "tag", "tag_fit_q012", "routing_shift_fraction_under_tag_patch"),
                               ("'", "tag_D2", "tag_D2_fit_q012", "routing_shift_fraction_under_tag_D2_patch")]:
        x = t[nm]
        c["K2" + suf] = ge(x["cond"]["iia"], 0.7) and ge(x["cond"]["iia"], t["random"]["cond"]["iia"] + 0.3) and ge(x["cond"]["kl_recovered"], 0.5)
        c["K3" + suf] = ge(x["perm"]["iia"], 0.6)
        c["K4" + suf] = le(x["target"]["change_rate"], 0.10)
        c["K5" + suf] = ge(df.get(nm), 0.5) and le(df.get("random_mean"), 0.2)
        c["K6" + suf] = ge(t[reuse]["cond_q345"]["iia"], 0.7 * x["cond"]["iia"])
        c["R1" + suf] = ge(r.get(rs), 0.5)
    c["K6'_gated"] = c["K6'"] and c["K2'"]
    c["TagRoute"] = all(c[k] for k in ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "R1"]) and fz_typeB
    c["TagRoute_D2"] = all(c[k] for k in ["K1", "K2'", "K3'", "K4'", "K5'", "K6'", "K7", "R1'"]) and fz_typeB
    c["TagOnly_D2"] = all(c[k] for k in ["K1", "K2'", "K3'", "K4'", "K5'", "K6'", "K7"]) and not c["TagRoute_D2"]
    nums = dict(s_tag=s, probe=pr[f"{s}|loo|trained"], tag_cond=t["tag"]["cond"]["iia"], tagD2_cond=t["tag_D2"]["cond"]["iia"],
                tagD2_kl=t["tag_D2"]["cond"]["kl_recovered"], tagD2_perm=t["tag_D2"]["perm"]["iia"], tagD2_target_change=t["tag_D2"]["target"]["change_rate"],
                tagD2_drop=df.get("tag_D2"), tagD2_reuse=t["tag_D2_fit_q012"]["cond_q345"]["iia"], routeD2=r.get("routing_shift_fraction_under_tag_D2_patch"),
                full_src_cond=t["full_src_q"]["cond"]["iia"], content=ct, overlap=r["overlap_tag_D2"])
    return c, nums


rows = {}
for f in sorted(glob.glob(os.path.join(R, "logs/analysis/main/*_conf.json"))) + sorted(glob.glob(os.path.join(R, "logs/analysis/curriculum/*_conf.json"))):
    if f.endswith("_tag_conf.json"): continue
    b = os.path.basename(f)[:-len("_conf.json")]
    task, size, _, seed = b.split("_")[:4]
    if "fromT1" in b: size = "M-curriculum[POST HOC]"
    fr = ld(f); tg = ld(f.replace("_conf.json", "_tag_conf.json"))
    if fr is None: continue
    c, n = frozen(fr)
    rec = dict(task=task, size=size, seed=seed, frozen=c, frozen_nums=n)
    if tg is not None:
        c2, n2 = tagc(tg, c["TypeB"]); rec["tag"] = c2; rec["tag_nums"] = n2
    rows[b] = rec


def verdict(recs, key, sub="frozen"):
    v = [r[sub][key] for r in recs if sub in r]
    if not v: return None
    return dict(passed=sum(v), n=len(v), holds=sum(v) >= math.ceil(2 * len(v) / 3))


groups = {}
for r in rows.values(): groups.setdefault((r["task"], r["size"]), []).append(r)
V = {}
for (t, s), recs in sorted(groups.items()):
    V[f"{t}/{s}"] = {k: verdict(recs, k) for k in ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "TypeA", "TypeB", "Distributed_literal", "Distributed_nonfinal", "P6_gated"]}
    V[f"{t}/{s}"].update({k: verdict(recs, k, "tag") for k in ["K1", "K2", "K2'", "K3'", "K4'", "K5'", "K6'", "K6'_gated", "K7", "R1", "R1'", "TagRoute", "TagRoute_D2", "TagOnly_D2"]})


def holds(g, k): return bool(V.get(g, {}).get(k) and V[g][k]["holds"])
a1, a4 = holds("T1/M", "TypeA"), holds("T4/M", "TypeA")
if a1 and a4: primary = "YES"
elif a1 or a4 or holds("T1/M", "TypeB") or holds("T4/M", "TypeB") or holds("T1/M", "Distributed_literal") or holds("T4/M", "Distributed_literal"): primary = "PARTIAL"
else: primary = "NO (scoped)"
sec = "YES" if holds("T1/M", "TagRoute_D2") and holds("T4/M", "TagRoute_D2") else ("PARTIAL" if holds("T1/M", "TagRoute_D2") or holds("T4/M", "TagRoute_D2") else "NO (scoped)")
out = dict(rows=rows, verdicts=V, primary_answer_frozen=primary, secondary_answer_tag_route_D2=sec)
json.dump(out, open(os.path.join(R, "logs/verdicts.json"), "w"), indent=1, default=float)

L = ["# Verdicts (confirmation split; generated by scripts/aggregate.py)", "",
     f"**Frozen primary answer (PROTOCOL.md §6): {primary}**  ", f"**Secondary (addendum 2, tag-and-route D2 variant): {sec}**", "",
     "## Per (task, size): seeds passing / seeds", "", "| group | " + " | ".join(["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "TypeA", "TypeB", "Distr(lit)", "Distr(nonfinal)", "K1", "K2", "K2'", "K3'", "K4'", "K5'", "K6'", "K7", "R1'", "TagRoute_D2"]) + " |",
     "|" + "---|" * 23]
for g, v in V.items():
    cell = lambda x: "—" if x is None else f"{x['passed']}/{x['n']}{'✓' if x['holds'] else ''}"
    L.append(f"| {g} | " + " | ".join(cell(v[k]) for k in ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "TypeA", "TypeB", "Distributed_literal", "Distributed_nonfinal", "K1", "K2", "K2'", "K3'", "K4'", "K5'", "K6'", "K7", "R1'", "TagRoute_D2"]) + " |")
L += ["", "## Per model numbers", "", "| model | acc | Bayes | own | s* | probe R² | D1 cond IIA | D2 cond IIA | D2 target chg | best pattern IIA | max head R² | s_tag | tag probe | tag cond | tagD2 cond | tagD2 kl | tagD2 perm | tagD2 target chg | tagD2 drop | tagD2 reuse | route shift D2 |", "|" + "---|" * 21]
f3 = lambda x: "—" if not ok(x) else (f"{x:.3f}" if isinstance(x, float) else str(x))
for b, r in rows.items():
    n = r["frozen_nums"]; m = r.get("tag_nums", {})
    L.append(f"| {b} | " + " | ".join(f3(x) for x in [n["acc"], n["bayes"], n["own"], n["s_star"], n["probe_r2"], n["D1_cond_iia"], n["D2_cond_iia"], n["D2_target_change"], n["best_pattern_patch"], n["max_group_mass_r2"],
                                                      m.get("s_tag"), m.get("probe"), m.get("tag_cond"), m.get("tagD2_cond"), m.get("tagD2_kl"), m.get("tagD2_perm"), m.get("tagD2_target_change"), m.get("tagD2_drop"), m.get("tagD2_reuse"), m.get("routeD2")]) + " |")
open(os.path.join(R, "VERDICTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:4]))
