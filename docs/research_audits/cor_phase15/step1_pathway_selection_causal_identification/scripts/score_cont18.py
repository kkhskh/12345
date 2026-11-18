"""XVIII stability readout at step 2546 (design v2 §10.4, pre-review B addendum fix 4): for every continued pair, phi_nec of each continued arm
against the continued paired baseline and block, banded with the structural + causal classes only (no behavioural gate).  Secondary; it
never changes the primary verdict.  Reads logs/score_XVIII.json, logs/XVIII_design.json, logs/measure_c18_cont.jsonl; writes logs/score_cont18.json."""
import json, os, math
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
S = json.load(open(os.path.join(P, "logs", "score_XVIII.json"))); D = json.load(open(os.path.join(P, "logs", "XVIII_design.json")))
M = {json.loads(l)["run"]: json.loads(l) for l in open(os.path.join(P, "logs", "measure_c18_cont.jsonl"))} if os.path.exists(os.path.join(P, "logs", "measure_c18_cont.jsonl")) else {}
ell = lambda x: math.log(max(x, 0.0) + 0.01); OPP = {"EARLY": "MIXTURE", "MIXTURE": "EARLY"}
sc = lambda r: r["cls2_struct"] if r["cls2_struct"] == r["cls2_causal"] else "HYBRID"
ct = lambda tag: tag.replace("_stop1211", "_stop2546") + "_cont"; out = {}
for key, sp in D["runs"].items():
    if sp["tag"] not in S["stability_continuations"] or sp["arm"] in ("baseline", "block"): continue
    pk = f"{sp['base']}|{sp['D']}"; g = lambda arm: M.get(ct(D["runs"][f"{pk}|{arm}"]["tag"]), {}).get("final", {}).get("2546")
    A, B, K = g(sp["arm"]), g("baseline"), g("block")
    if not (A and B and K): out[key] = "NOT_MEASURED"; continue
    ph = 0.5 * sum((ell(A[f"nec_{r}"]) - ell(B[f"nec_{r}"])) / (ell(K[f"nec_{r}"]) - ell(B[f"nec_{r}"])) for r in ("E", "M"))
    bd = "F" if (ph >= 0.8 and sc(A) == OPP.get(sc(B))) else ("N" if (ph <= 0.2 and sc(A) == sc(B)) else "P")
    out[key] = dict(phi_nec_2546=ph, band_2546=bd, primary_1211=S["pairs"][pk]["arms"][sp["arm"]]["primary"], transient=bool(S["pairs"][pk]["arms"][sp["arm"]]["primary"] == "P" and bd in ("F", "N")))
json.dump(out, open(os.path.join(P, "logs", "score_cont18.json"), "w"), indent=1); print(json.dumps(out)[:2000])
