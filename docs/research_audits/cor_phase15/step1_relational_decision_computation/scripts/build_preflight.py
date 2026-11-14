"""Merge development outputs into logs/<analysis>_pre.json (preflight units for score_xiii.py --dev).  Later files override earlier ones."""
import json, os
L = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
SRC = {"l2_scan": ["l2_scan_devall", "l2_scan_dev"], "l2_mix_causal": ["l2_mix_causal_dev", "l2_mix_causal_dev2", "l2_mix_causal_dev3", "l2_mix_causal_dev4"],
       "l2_mediation": ["l2_mediation_dev"], "score_contrast": ["score_contrast_dev", "score_contrast_devv", "score_contrast_devdecoy", "score_contrast_devdecoyv"],
       "l1abl_mix": ["l1abl_mix_dev", "l1abl_mix_devv"], "decoy_disc": ["decoy_disc_dev2"]}
for k, fs in SRC.items():
    m = {}
    for f in fs:
        f = os.path.join(L, f + ".json")
        if os.path.exists(f): m.update({a: b for a, b in json.load(open(f)).items() if ":" in a})
    json.dump(m, open(os.path.join(L, k + "_pre.json"), "w")); print(k, sorted(m)[:8])
