"""XVIII POST-HOC analysis (not frozen; written after the frozen scorer's output was seen).  The pre-registered sensitivity analysis
('every verdict also computed without the behavioural diagnostic', design §10.8) was implemented in score_XVIII.py for the BANDS only;
evaluability still used the behaviour-gated baseline class, so 4 of 6 pairs were non-evaluable in both analyses.  This script applies the
pre-registered text in full: baseline and block classes from structural + causal agreement, bands without the behavioural gate; everything
else (phi_nec, thresholds, competence, the symmetric 5/2 rule, per-base 3/2 rule, predictions) is the frozen scorer's.
Reads logs/score_XVIII.json and logs/measure_c18.jsonl; writes logs/posthoc_nobehav18.json."""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
S = json.load(open(os.path.join(P, "logs", "score_XVIII.json"))); D = json.load(open(os.path.join(P, "logs", "XVIII_design.json")))
M = {json.loads(l)["run"]: json.loads(l) for l in open(os.path.join(P, "logs", "measure_c18.jsonl"))}
sc = lambda r: r["cls2_struct"] if r["cls2_struct"] == r["cls2_causal"] else "HYBRID"; OPP = {"EARLY": "MIXTURE", "MIXTURE": "EARLY"}
comp = lambda a, b: bool(a["KL"] <= 0.045 and a["acc"] >= b["acc"] - 0.03)
PRED = {"H1d": dict(vrg="F", gsg="N", gsv="N", sha_hi_rolled="F", sha_lo_real="N"), "Hm_window": dict(vrg="med", gsg="med", gsv="med", sha_hi_rolled="N", sha_lo_real="F"),
        "Hm_release": dict(vrg="med", gsg="med", gsv="med", sha_hi_rolled="med", sha_lo_real="med"), "H2g_R": dict(vrg="N", gsg="F", gsv="F", sha_hi_rolled="N", sha_lo_real="N"),
        "H2g_RV": dict(vrg="N", gsg="N", gsv="F", sha_hi_rolled="N", sha_lo_real="N"), "H4": dict(vrg="F", gsg="N", gsv="N", sha_hi_rolled="N", sha_lo_real="N"),
        "H1d_plus_Hm": dict(vrg="F", gsg="P", gsv="P", sha_hi_rolled="P", sha_lo_real="P"), "H1d_plus_H2g": dict(vrg="F", gsg="P", gsv="PF", sha_hi_rolled="F", sha_lo_real="N")}
ev = {}
for pk, p in S["pairs"].items():
    B = M[D["runs"][f"{pk}|baseline"]["tag"]]["final"]["1211"]; K = M[D["runs"][f"{pk}|block"]["tag"]]["final"]["1211"]
    ev[pk] = bool(comp(B, B) and sc(B) in OPP and comp(K, B) and sc(K) == OPP[sc(B)])
def judge(pairs, nc, nm):
    out = {}
    for h, pr in PRED.items():
        arms = {}
        for arm, pb in pr.items():
            mt = mm = 0; det = []
            for pk in pairs:
                r = S["pairs"][pk]["arms"][arm]; o = r["primary_nb"]
                if o not in ("F", "P", "N"): continue
                al = r["mediator_band"] if pb == "med" else pb; ok = o in al; mt += ok; mm += (not ok); det.append((pk, o, al))
            arms[arm] = dict(pred=pb, matches=mt, mismatches=mm, status="contradicted" if mm >= nm else ("consistent" if mt >= nc else "inconclusive"), detail=det)
        st = [a["status"] for a in arms.values()]; out[h] = dict(verdict="CONTRADICTED" if "contradicted" in st else ("CONSISTENT" if all(x == "consistent" for x in st) else "INCONCLUSIVE"), arms=arms)
    return out
evp = [k for k, v in ev.items() if v]
res = dict(label="POST-HOC (not frozen)", evaluable=ev, pooled=judge(evp, 5, 2), per_base={b: judge([k for k in evp if k.startswith(b + "|")], 3, 2) for b in D["bases"]},
           bands={pk: {a: S["pairs"][pk]["arms"][a]["primary_nb"] for a in S["pairs"][pk]["arms"]} for pk in S["pairs"]},
           phi={pk: {a: round(S["pairs"][pk]["arms"][a]["phi_nec"], 3) for a in S["pairs"][pk]["arms"]} for pk in S["pairs"]})
json.dump(res, open(os.path.join(P, "logs", "posthoc_nobehav18.json"), "w"), indent=1)
print("evaluable", ev); print("pooled", {h: v["verdict"] for h, v in res["pooled"].items()})
for b, v in res["per_base"].items(): print(b, {h: x["verdict"] for h, x in v.items()})
for pk, v in res["bands"].items(): print(pk, v)
