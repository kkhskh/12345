"""Summarise logs/scorer_validation19.json -> logs/scorer_validation_summary19.json (stability of the causal class across 3 streams;
agreement of the functional class with the causal class among causally pure, competent models; minimum margins to the thresholds)."""
import json, os, collections
H = os.path.dirname(os.path.abspath(__file__)); J = json.load(open(os.path.join(H, "..", "logs", "scorer_validation19.json"))); S = list(J); N = list(J[S[0]])
out = dict(streams=S, n_models=len(N)); c = collections.Counter(); bad = []
for n in N:
    cs = {J[s][n]["causal"] for s in S}; c["causal_stable" if len(cs) == 1 else "causal_varies"] += 1
    comp = all(J[s][n]["competent"] for s in S); c["competent_all_streams" if comp else "not_competent_some_stream"] += 1
    for s in S:
        r = J[s][n]
        if r["causal"] in ("EARLY", "MIXTURE") and r["competent"]:
            k = "agree" if r["functional"] == r["causal"] else ("ambiguous" if r["functional"] == "AMBIGUOUS" else "contradict"); c["func_vs_causal_" + k] += 1
            if k == "contradict": bad.append((s, n))
out["counts"] = dict(c); out["contradictions"] = bad
# margins of pure competent anchors to the class thresholds (how far from flipping)
m = []
for n in N:
    for s in S:
        r = J[s][n]
        if r["causal"] == "EARLY": m.append(min(r["nec_E"] - 0.30, 0.10 - r["nec_M"]))
        if r["causal"] == "MIXTURE": m.append(min(r["nec_M"] - 0.30, 0.15 - r["nec_E"]))
out["min_margin_pure"] = min(m); out["n_pure_model_streams"] = len(m); out["n_margin_below_0.05"] = sum(x < 0.05 for x in m)
json.dump(out, open(os.path.join(H, "..", "logs", "scorer_validation_summary19.json"), "w"), indent=1); print(json.dumps(out, indent=1))
