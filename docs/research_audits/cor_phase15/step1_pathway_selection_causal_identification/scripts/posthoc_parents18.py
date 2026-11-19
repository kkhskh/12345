"""XVIII POST-HOC parent analysis (not frozen): the frozen parent scorer's bands with baseline/block classes from structural + causal agreement
(behavioural gate removed, as in posthoc_nobehav18.py).  Writes logs/posthoc_parents18.json."""
import json, os, math
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
PJ = json.load(open(os.path.join(P, "logs", "XVIII_parents.json"))); M = {json.loads(l)["run"]: json.loads(l) for l in open(os.path.join(P, "logs", "measure_p18.jsonl"))}
ell = lambda x: math.log(max(x, 0.0) + 0.01); OPP = {"EARLY": "MIXTURE", "MIXTURE": "EARLY"}; sc = lambda r: r["cls2_struct"] if r["cls2_struct"] == r["cls2_causal"] else "HYBRID"
comp = lambda a, b: bool(a["KL"] <= 0.045 and a["acc"] >= b["acc"] - 0.03); out = {}
for p, v in PJ["parents"].items():
    if not v["bistable"]: out[p] = "not bistable"; continue
    res = {}
    for b in ("E", "M"):
        g = lambda a: M[v["runs"][f"{b}|{a}"]]["final"]["1211"]; B, K = g("baseline"), g("block")
        ev = bool(comp(B, B) and sc(B) in OPP and comp(K, B) and sc(K) == OPP[sc(B)]); r = dict(evaluable=ev, block="F" if ev else "NONEVAL")
        for a in ("vrg", "gsv", "sha_hi_rolled", "sha_lo_real"):
            A = g(a); ph = 0.5 * sum((ell(A[f"nec_{x}"]) - ell(B[f"nec_{x}"])) / (ell(K[f"nec_{x}"]) - ell(B[f"nec_{x}"])) for x in ("E", "M"))
            bd = "MISSING" if not comp(A, B) else ("F" if (ph >= 0.8 and sc(A) == OPP.get(sc(B))) else ("N" if (ph <= 0.2 and sc(A) == sc(B)) else "P"))
            r[a] = dict(phi_nec=ph, band=bd if ev else "NONEVAL")
        res[b] = r
    out[p] = res
json.dump(dict(label="POST-HOC (not frozen)", parents=out), open(os.path.join(P, "logs", "posthoc_parents18.json"), "w"), indent=1, default=float); print(json.dumps(out, default=lambda x: round(x, 2)))
