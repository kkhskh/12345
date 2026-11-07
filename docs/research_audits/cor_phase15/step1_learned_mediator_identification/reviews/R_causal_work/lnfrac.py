import sys, warnings; from pathlib import Path; import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import ckpt, mediator as MD, run_protocol as RP
for sub, s in (("historical", 2), ("historical", 4), ("new", 7), ("new", 8), ("new", 9)):
    P = ckpt.load(ROOT / "checkpoints" / sub / f"T1_TF2_s{s}.npz")
    A, C = RP.pairs("REL-XNOR", 800, 5001); bank = RP.sample("REL-XNOR", 1500, 6003, "train")
    U, k, _ = MD.s_rel(MD.Site(P, "REL-XNOR", "resid_mid1", "R"), A, C)
    _, hf = MD.Site(P, "REL-XNOR", "resid_post1", "R").get(bank)
    _, hm = MD.Site(P, "REL-XNOR", "resid_mid1", "R").get(bank)
    hc = hf - hf.mean(1, keepdims=True); Uc = U - U.mean(0, keepdims=True)
    frac = np.median(np.linalg.norm(hc @ np.linalg.qr(Uc)[0], axis=1) / np.linalg.norm(hc, axis=1))
    mlp_out = hf - hm
    print(s, "median fraction of LN-centred final state norm in span(U):", round(float(frac), 3),
          "| median ||MLP1 out U-coord|| / ||resid_mid1 U-coord||:", round(float(np.median(np.abs(mlp_out @ U).sum(1) / np.abs(hm @ U).sum(1))), 3))
