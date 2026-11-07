import sys, warnings; from pathlib import Path; import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import ckpt, mediator as MD, run_protocol as RP
from sklearn.linear_model import LogisticRegression
for sub, s in (("historical", 2), ("new", 7), ("new", 9)):
    P = ckpt.load(ROOT / "checkpoints" / sub / f"T1_TF2_s{s}.npz"); site = MD.Site(P, "REL-XNOR", "resid_mid1", "R")
    A, C = RP.pairs("REL-XNOR", 800, 5001); bank = RP.sample("REL-XNOR", 1500, 6003, "train"); test = RP.sample("REL-XNOR", 1500, 42, "train")
    U, k, _ = MD.s_rel(site, A, C); Ua = MD.s_alt(site, bank, U, k)
    _, hb = site.get(bank); _, ht = site.get(test)
    for nm, V in (("S_rel", U), ("S_alt", Ua)):
        clf = LogisticRegression(max_iter=3000).fit(hb @ V, bank["s"]); print(s, nm, "test decode acc", clf.score(ht @ V, test["s"]))
    Pc = np.eye(16) - U @ U.T
    clf = LogisticRegression(max_iter=3000).fit(hb @ Pc, bank["s"]); print(s, "full complement decode", clf.score(ht @ Pc, test["s"]))
