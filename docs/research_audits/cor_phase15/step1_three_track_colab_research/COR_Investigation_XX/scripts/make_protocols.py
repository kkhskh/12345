"""Write the machine-readable protocol files from cor_research/protocol.py (the single source).  python scripts/make_protocols.py"""
import os, sys, json
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG)
from cor_research import protocol as PR
P = PR.full(); w = lambda rel, obj: json.dump(obj, open(os.path.join(PKG, rel), "w"), indent=1, sort_keys=True)
w("protocols/frozen_hypotheses/track_a_hypotheses.json", dict(version=P["version"], hypotheses=P["HYPOTHESES_A"], exploratory=P["EXPLORATORY_A"], band=P["BAND"], delta=P["DELTA"],
  min_pairs_per_parent=P["MIN_PAIRS_PER_PARENT"], min_parents=P["MIN_PARENTS"], competence=P["COMP"], causal_class=P["CLASS"],
  eps=P["EPS"], eps_sensitivity=P["EPS_SENS"], parent_rule=P["PARENT_RULE"], cross_rule=P["CROSS_RULE"], validity=P["VALIDITY"]))
w("protocols/causal_design/track_a_design.json", {k: P[k] for k in ("version", "HIST_PARENTS", "PREVIOUSLY_OBSERVED", "FRESH_PARENTS", "PILOT_PARENTS", "DEV_PARENT", "PARENT_VALIDITY", "PRETRAIN", "SCREEN", "CONF_D", "PILOT_D", "STREAMS", "STEPS", "WINDOW", "ALLG", "ARMS", "HIGH_EXTRA_ARMS", "ARMSET")})
w("protocols/functional_design/track_b_design.json", dict(version=P["version"], **P["TRACK_B"]))
w("protocols/od_design/track_c_protocol.json", dict(version=P["version"], **P["TRACK_C"]))
w("configurations/presets.json", dict(version=P["version"], presets=P["PRESETS"]))
print("protocol files written")
