"""Reproduces logs/aggnl_reversals.json: row-reversal counts and within-z spread for AGG-NL margins (31x31 grid)."""
import sys, itertools, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import ckpt, mediator as MD
res = {}
for arch in ["TF1", "TF2", "TF2AO"]:
    for s in range(5):
        P = ckpt.load(ROOT / f"checkpoints/new/T6_{arch}_s{s}.npz")
        g = np.linspace(-3, 3, 31); Aa, Bb = np.meshgrid(g, g, indexing="ij")
        site = MD.Site(P, "AGG-NL", "resid_mid0" if arch != "TF2" else "resid_mid1")
        lg, _ = site.get(dict(a=Aa.ravel(), b=Bb.ravel(), p=np.zeros(Aa.size)))
        m = MD.margin(lg).reshape(31, 31); tol = 0.05 * m.std(); n = 0
        for i, j in itertools.combinations(range(31), 2):
            dd = m[i] - m[j]; n += (dd > tol).any() and (dd < -tol).any()
        z = (Aa + Bb).ravel(); mm = m.ravel(); idx = np.round(z / 0.2).astype(int)
        dev = np.mean([mm[idx == v].std() for v in np.unique(idx)]) / mm.std()
        res[f"{arch}/s{s}"] = dict(reversing_row_pairs=int(n), total=465, within_z_sd_over_sd=float(dev))
(ROOT / "logs" / "aggnl_reversals.json").write_text(json.dumps(res, indent=1)); print(res)
