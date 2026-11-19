"""Development band check (pre-review A2): the D801 development checkpoints re-measured with measure19 (stream 86 200 000,
logs/dev/m19_dev2.jsonl) scored with the frozen hypothesis set of logs/conf_spec19.json (min_pairs 1, one pair per base).
A missing development run is reported as MISSING (not as non-competent).  Development evidence only.  Writes logs/dev/dev_bandcheck19.json."""
import json, sys, os
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import score19
S = json.load(open(os.path.join(H, "..", "logs", "conf_spec19.json"))); recs = {}
for l in open(os.path.join(H, "..", "logs", "dev", "m19_dev2.jsonl")): r = json.loads(l); recs[r["run"]] = r
def find(base, arm):
    L = 21 if base == "E" else 23; pre = f"uP160_fx_L{L}_W{L}_D801"; t = "own0" if base == "E" else "own2"; fz = "L0_QK" if base == "E" else "L2_QK"
    tr = dict(vsg=f"{pre}_vsg{t}T100_stop1211_dev", vsgo=f"{pre}_vsgo{t}T100_stop1211_dev", vrs=f"{pre}_vrs{t}T100_stop1211_dev", frz=f"{pre}_stop1211_br100_ctl_frz{fz}T100",
              frzp=f"{pre}_frzp{fz}T100_stop1211_dev", pnz=f"{pre}_pnzs1T100_stop1211_dev", lbl=f"{pre}_stop1211_br100_ctl_lblp0.5")
    return recs.get(tr.get(arm, f"{pre}_stop1211_br100_{base}801_{arm}"))
arms = sorted({a["name"] for a in S["transplant_arms"]} | set(S["training_arms"])); bases = {a["name"]: a.get("bases", "EM") for a in S["transplant_arms"]}
missing = [(b, a) for b in "EM" for a in arms if b in bases.get(a, "EM") and find(b, a) is None]
D = dict(min_pairs=1, pairs=[dict(name="E801", base="E"), dict(name="M801", base="M")], families=dict(T=dict(anchors=dict(B="fullB", I="fullI"))),
         arms={a: dict(family="T", bases="".join(b for b in bases.get(a, "EM") if (b, a) not in missing)) for a in arms}, hypotheses=[])
for h in S["hypotheses"]:
    used = [a for a in arms if f"'{a}'" in h.get("expr", "") or a == h.get("arm")]
    if any((h["base"], a) in missing for a in used): continue
    D["hypotheses"].append(h)
M = {(p["name"], a): find(p["base"], a) for p in D["pairs"] for a in arms if p["base"] in D["arms"][a]["bases"]}
O = score19.score(D, M, dict(score19.CFG)); out = dict(note=__doc__, missing_dev_runs=missing, hypotheses={})
for h in S["hypotheses"]:
    v = O["hypotheses"].get(h["id"]); out["hypotheses"][h["id"]] = dict(band="MISSING (no dev run)", values={}) if v is None else dict(band=v["band"], values=v["values"], dropped=v["dropped"])
    print(h["id"].ljust(36), out["hypotheses"][h["id"]]["band"].ljust(14), {p: (round(x["v"], 2) if "v" in x else x["category"]) for p, x in out["hypotheses"][h["id"]]["values"].items()})
out["arms"] = {f"{p}|{a}": dict(cat=x["category"], phi=x["phi"], f=x["f"]) for p, P in O["pairs"].items() for a, x in P["families"]["T"]["arms"].items()}
json.dump(out, open(os.path.join(H, "..", "logs", "dev", "dev_bandcheck19.json"), "w"), indent=1)
