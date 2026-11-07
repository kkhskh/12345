"""C5 (exploratory): a relation-free pre-norm Transformer with the same decisions as C' on its operating region.

Block 1: attention off. Per-position ReLU MLP computes phi(t) = clamp(k t, -1, 1) of the position's OWN
evidence coordinate (e1 at A, e2 at B) and writes it onto fU. No unit reads two sources.
Block 2: a uniform head at R averages fU over P, A, B, R; P carries -tau on fU (prior offset).
Margin = <fU, h_R> = (phi_A + phi_B - tau)/4 (up to the LN scale of block-2's value read).
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath
from cbmath import Layer, Model
from cprime import closed_form, certify

def relation_free(k=40.0, tau=0.3, d=10):
    m0, D, emb = cbmath.rereading()
    e1, e2, fU, mkP = D["e1"], D["e2"], D["fU"], D["mkP"]
    zQ = np.zeros((1, d, 2))
    Win = np.column_stack([k * e1, k * e1, k * e2, k * e2])
    bin_ = np.array([1.0, -1.0, 1.0, -1.0])
    Wout = np.stack([fU, -fU, fU, -fU])   # relu(kt+1) - relu(kt-1) = clamp(kt,-1,1) + 1; bout removes the +2 offset
    L1 = Layer(WQ=zQ, WK=zQ.copy(), WV=zQ.copy(), WO=np.zeros((1, 2, d)), bO=np.zeros(d), Win=Win, bin=bin_,
               Wout=Wout, bout=-2.0 * fU, act="relu", attn_on=False)
    WV = np.zeros((1, d, 2)); WV[0, :, 0] = fU - tau * mkP
    WO = np.zeros((1, 2, d)); WO[0, 0] = fU
    L2 = Layer(WQ=zQ.copy(), WK=zQ.copy(), WV=WV, WO=WO, bO=np.zeros(d), Win=np.zeros((d, 1)), bin=np.zeros(1),
               Wout=np.zeros((1, d)), bout=np.zeros(d), mlp_on=False)
    return Model(layers=[L1, L2], WU=np.stack([fU / 2, -fU / 2], 1), bU=np.zeros(2)), emb

def main():
    mR, emb = relation_free()
    g = np.concatenate([-np.linspace(3, 0.1, 30), np.linspace(0.1, 3, 30)])
    same = tot = 0
    cert_same = cert_tot = 0
    for a in g:
        for b in g:
            mc = closed_form(a, b)["m"]; mr = cbmath.margin(cbmath.forward(mR, emb(a, b), 3))
            tot += 1; same += np.sign(mc) == np.sign(mr)
    # on certified C' operating rectangles (tau-floors from CPRIME doc)
    rng = np.random.default_rng(61)
    for _ in range(20000):
        a, b = rng.uniform(-3, 3, 2)
        if (a > 0 and b > 0 and min(a, b) < 0.09) or (a * b < 0 and min(abs(a), abs(b)) < 0.02):
            continue
        mc = closed_form(a, b)["m"]; mr = cbmath.margin(cbmath.forward(mR, emb(a, b), 3))
        cert_tot += 1; cert_same += np.sign(mc) == np.sign(mr)
    # does any unit read two sources?  check MLP read directions
    L1 = mR.layers[0]
    reads_both = any(abs(L1.Win[:, j] @ np.array(cbmath.rereading()[1]["e1"])) > 1e-12 and abs(L1.Win[:, j] @ np.array(cbmath.rereading()[1]["e2"])) > 1e-12 for j in range(L1.Win.shape[1]))
    out = dict(grid_same=float(same / tot), random_points_outside_dead_bands_same=float(cert_same / cert_tot), n=cert_tot,
               any_unit_reads_both_sources=bool(reads_both))
    print(json.dumps(out))
    (HERE.parent / "logs" / "relation_free_equivalent.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
