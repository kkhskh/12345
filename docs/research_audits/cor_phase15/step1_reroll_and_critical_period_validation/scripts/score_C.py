"""XIII-E Workstream C frozen predictions for the HELD-OUT carrier unit C-C2 (uP165 s21 early01), development = C-C1 (uP160 s23).
FAIL-CLOSED: refuses (exit 2, writes nothing) unless logs/c_audit.json holds a complete record for uP165:s:21 (C2 decoding+geometry with the
named sets, C3 overlaps, C4 k = 1..6).  -> logs/score_C.json
R5 (dimensionality): SUFF(k=1) >= 0.85 * SUFF(k=3); seed stability(k=1) >= 0.9; seed stability(k=3) <= 0.8.
R6 (decoder geometry under matched site and pairs): (a) held-out decoding R2 of m from the DEC_m projection >= 0.9 x that from ENC_m;
   (b) SUFF(DEC_m) <= 0.1, SUFF(ENC_m) >= 0.5, mean SUFF over the 3 variance-matched random subspaces <= 0.2;
   (c) signal variance along ENC_m >= 5 x that along DEC_m; (d) Euclidean overlap DAS|ENC_m >= 0.2 and DAS|DEC_m <= 0.15.
Each prediction: CONFIRMED iff every clause holds, else FALSIFIED (no non-diagnostic clause: the unit is competent by XIII-D P3)."""
import json, os, sys
LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
def refuse(m): print("REFUSED:", m, file=sys.stderr); sys.exit(2)
try: R = json.load(open(os.path.join(LD, "c_audit.json")))["uP165:s:21"]
except Exception as ex: refuse(f"no complete C-C2 record: {ex}")
try:
    c4 = {int(k): v for k, v in R["C4"].items()}; g = R["C2"]["geometry"]; dc = R["C2"]["decoding"]; c3 = R["C3"]
    assert set(c4) == set(range(1, 7)); [g[n]["SUFF_slope"] for n in ("DEC_m", "ENC_m", "VMRAND0", "VMRAND1", "VMRAND2")]; dc["proj_R2_m_DEC_m"]; dc["proj_R2_m_ENC_m"]; c3["DAS|ENC_m"]; c3["DAS|DEC_m"]
except Exception as ex: refuse(f"incomplete record: {ex}")
r5 = dict(ratio=c4[1]["SUFF_slope"] / c4[3]["SUFF_slope"], stab1=c4[1]["seed_stability_mean_cos"], stab3=c4[3]["seed_stability_mean_cos"])
r5["verdict"] = "CONFIRMED" if (r5["ratio"] >= 0.85 and r5["stab1"] >= 0.9 and r5["stab3"] <= 0.8) else "FALSIFIED"
vm = sum(g[f"VMRAND{j}"]["SUFF_slope"] for j in range(3)) / 3
r6 = dict(dec_over_enc_R2=dc["proj_R2_m_DEC_m"] / dc["proj_R2_m_ENC_m"], suff_dec=g["DEC_m"]["SUFF_slope"], suff_enc=g["ENC_m"]["SUFF_slope"], suff_vmrand=vm,
          signal_ratio=g["ENC_m"]["signal_var"] / g["DEC_m"]["signal_var"], ov_das_enc=c3["DAS|ENC_m"]["overlap"], ov_das_dec=c3["DAS|DEC_m"]["overlap"])
r6["verdict"] = "CONFIRMED" if (r6["dec_over_enc_R2"] >= 0.9 and r6["suff_dec"] <= 0.1 and r6["suff_enc"] >= 0.5 and vm <= 0.2 and r6["signal_ratio"] >= 5 and r6["ov_das_enc"] >= 0.2 and r6["ov_das_dec"] <= 0.15) else "FALSIFIED"
json.dump(dict(R5=r5, R6=r6, summary=dict(R5=r5["verdict"], R6=r6["verdict"])), open(os.path.join(LD, "score_C.json"), "w"), indent=1); print(dict(R5=r5["verdict"], R6=r6["verdict"]))
