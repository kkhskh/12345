"""XVIII report checker: key numbers and labels in the reports against the score files and logs."""
import json, os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rd = lambda f: open(os.path.join(P, f)).read()
S = json.load(open(os.path.join(P, "logs", "score_XVIII.json"))); PH = json.load(open(os.path.join(P, "logs", "posthoc_nobehav18.json")))
MAIN, FRZ, PATH, PAR = rd("MAIN_RESEARCH_SYNTHESIS.md"), rd("FROZEN_PREDICTIONS_AND_RESULTS.md"), rd("PATHWAY_INFORMATION_VS_ATTENTION_COMPETITION.md"), rd("INDEPENDENT_PARENT_GENERALIZATION.md")
ok = []
def chk(n, c): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n)
fv = {h: v["verdict"] for h, v in S["hypotheses"].items()}
chk("frozen: 6 CONTRADICTED, 2 INCONCLUSIVE (Hm_release, H1d_plus_H2g)", sorted(h for h, v in fv.items() if v == "INCONCLUSIVE") == ["H1d_plus_H2g", "Hm_release"] and list(fv.values()).count("CONTRADICTED") == 6 and "Six of eight" in MAIN)
chk("frozen: 2 evaluable pairs stated", sum(p["evaluable"] for p in S["pairs"].values()) == 2 and "2 of 6 pairs" in MAIN)
chk("frozen arm bands all INCONCLUSIVE", all(v["pooled"] == "INCONCLUSIVE" for v in S["arms"].values()))
chk("post-hoc: 5 evaluable pairs", sum(PH["evaluable"].values()) == 5 and "5 evaluable" in MAIN)
b = PH["bands"]; E = [f"E|{d}" for d in (811, 812, 813)]; Mv = ["M|811", "M|812"]
chk("post-hoc: vrg and sha_hi_rolled F in all 5 evaluable", all(b[k][a] == "F" for k in E + Mv for a in ("vrg", "sha_hi_rolled")))
chk("post-hoc: sha_lo_real N in all 6", all(b[k]["sha_lo_real"] == "N" for k in b))
chk("post-hoc: gsv F in all 5 evaluable", all(b[k]["gsv"] == "F" for k in E + Mv))
chk("post-hoc: gsg E F,P,F and M P", [b[k]["gsg"] for k in E] == ["F", "P", "F"] and all(b[k]["gsg"] == "P" for k in Mv))
chk("H3 E frozen not excluded", S["h3_E"]["excluded"] is False and "frozen H3 rule is not met" in MAIN)
chk("post-hoc labelled in MAIN", "[sens]" in MAIN and "my error" in MAIN.lower())
chk("withdrawn unifying description", "withdrawn as a finding" in MAIN.lower() and "Withdrawn as a finding" in PATH)
chk("C1 withdrawal present", "restricting a route's attention share does not change the organisation" in MAIN and "withdrawn" in MAIN)
chk("parents: 1 of 5 bistable, MIXED vrg/gsv", "1 of 5" in MAIN and "MIXED" in PAR)
chk("not human peer review", "not human peer review" in MAIN); chk("not standalone", "not standalone" in MAIN); chk("O/D not validated", "not validated" in MAIN)
print("ALL_PASS", all(ok), f"({sum(ok)}/{len(ok)})")
