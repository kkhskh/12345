"""Descriptive tables for the documents (not a confirmatory test).  -> logs/summary_tables.json"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs", "conf"); SN = (250, 500, 1000, 2000, 4000)
REG = ("R1_with_colluder_candidate", "R2_with_third_group", "R3_with_no_group"); out = {}
for a, S in (("T", range(70, 78)), ("P", range(80, 88))):
    for st in SN:
        rows = [json.load(open(os.path.join(L, f"model_x{a}{s}_{st}.json"))) for s in S]; t = {}
        for f in ("F45", "F34", "F23", "F0"):
            F = [r["families"][f] for r in rows]
            t[f] = dict(B=float(np.mean([x["B_logratio"] for x in F])), B_sd=float(np.std([x["B_logratio"] for x in F], ddof=1)),
                        B_prob=float(np.mean([x["B_prob"] for x in F])), dmM=float(np.mean([x["mean_dmM"] for x in F])), dmB=float(np.mean([x["mean_dmB"] for x in F])),
                        slope=float(np.mean([x["slope_dmM_on_dmB"] for x in F])), rank_model=float(np.mean([x["rank_change_model"] for x in F])), rank_bayes=float(F[0]["rank_change_bayes"]),
                        regimes={rg[:2]: dict(B=float(np.mean([x["by_regime"][rg]["B_logratio"] for x in F if rg in x["by_regime"]])),
                                              dmM=float(np.mean([x["by_regime"][rg]["B_logratio"] + x["by_regime"][rg]["mean_dmB"] for x in F if rg in x["by_regime"]])),
                                              dmB=float(np.mean([x["by_regime"][rg]["mean_dmB"] for x in F if rg in x["by_regime"]])),
                                              n_models_dmM_neg=int(sum(x["by_regime"][rg]["B_logratio"] + x["by_regime"][rg]["mean_dmB"] < 0 for x in F if rg in x["by_regime"])))
                                 for rg in REG})
        q = {k: [r["quality"][k] for r in rows] for k in rows[0]["quality"]}
        t["quality"] = {k: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in q.items()}
        out[f"{a}@{st}"] = t
json.dump(out, open(os.path.join(HERE, "..", "logs", "summary_tables.json"), "w"), indent=1)
for k, t in out.items():
    print(k, "F45 R1 dmM %.3f (B %.3f, neg %d/8) | R2 %.3f | R3 %.3f || F34 R1 dmM %.3f B %.3f | F23 R1 B %.3f | slope45 %.2f slope34 %.2f | KL %.3f acc %.3f ECE %.3f T* %.2f scale %.2f" % (
        t["F45"]["regimes"]["R1"]["dmM"], t["F45"]["regimes"]["R1"]["B"], t["F45"]["regimes"]["R1"]["n_models_dmM_neg"], t["F45"]["regimes"]["R2"]["B"], t["F45"]["regimes"]["R3"]["B"],
        t["F34"]["regimes"]["R1"]["dmM"], t["F34"]["regimes"]["R1"]["B"], t["F23"]["regimes"]["R1"]["B"], t["F45"]["slope"], t["F34"]["slope"], t["quality"]["KL_bayes"]["mean"], t["quality"]["acc"]["mean"], t["quality"]["ECE"]["mean"], t["quality"]["T_star"]["mean"], t["quality"]["margin_scale"]["mean"]))
print("Bayes R1 dmB F45 %.3f F34 %.3f F23 %.3f; rank reversal Bayes F45 %.3f" % (out["T@4000"]["F45"]["regimes"]["R1"]["dmB"], out["T@4000"]["F34"]["regimes"]["R1"]["dmB"], out["T@4000"]["F23"]["regimes"]["R1"]["dmB"], out["T@4000"]["F45"]["rank_bayes"]))
print("rank change model F45 T %.3f P %.3f" % (out["T@4000"]["F45"]["rank_model"], out["P@4000"]["F45"]["rank_model"]))
