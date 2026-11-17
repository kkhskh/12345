"""XVI Workstream B search [num]: two-reliability conditions at arity 2 whose gate value differs while the competing agreement cue is matched.
Grid over (b1, b0, nu); exact references via xvi_math.analyse (no flip slopes, for speed).  Output logs/xvi_value_grid.json."""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, xvi_math as XM
B1 = [0.8, 0.85, 0.9, 0.95]; NU = [0.05, 0.15, 0.25, 1 / 3, 0.45]; B0 = [round(x, 3) for x in np.arange(0.4, 0.991, 0.05)]
rows = []
for b1, nu, b0 in itertools.product(B1, NU, B0):
    r = XM.analyse(f"2:{b1}:{b0}:{nu}", fl=False)
    rows.append(dict(b1=b1, b0=b0, nu=nu, gain_gate=r["gain_gate"], gain_int=r["gain_intermediate"], overlap=r["cue_overlap_acc"], MI_agree=r["MI_g_agree"],
                     bayes_nll=r["bayes"]["NLL"], bayes_acc=r["bayes"]["acc"], o1_nll=r["order1"]["NLL"], o1_acc=r["order1"]["acc"], oa_nll=r["order2"]["NLL"], oa_acc=r["order2"]["acc"],
                     acc_gain_gate=r["acc_gain_gate"], min_order=r["min_decision_order"]))
json.dump(rows, open(os.path.join(HERE, "..", "logs", "xvi_value_grid.json"), "w"), indent=0); print(len(rows))
