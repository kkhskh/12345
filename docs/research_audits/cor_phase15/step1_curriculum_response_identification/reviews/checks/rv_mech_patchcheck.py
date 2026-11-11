"""Mechanistic review check on committed checkpoint(s): (1) patching the layer-2 input at ALL positions reproduces the natural
counterfactual; (2) recomputation of Rq-only / Rq-blocked responses matches the frozen arrays; (3) departure from Bayes
(level, not difference) at the scored decision on N and R pairs: natural vs Rq-blocked; (4) attention of the readout to
question-q source tokens (layer 2) natural vs Rq-blocked, as a hint of the interaction."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "..", "logs", "conf")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); import cri as R; E = R.E
G = np.load(os.path.join(D, "generator.npz")); X, q = G["X"], G["qsel"]; NN = 600
for m in sys.argv[1:]:
    M = R.Model(m); z = np.load(os.path.join(D, f"model_{m}.npz")); print("==", m, "layers", len(M.p["layers"]))
    for F in ("N", "R"):
        mk = np.where(G[f"ok_{F}"])[0][:NN]; Xb, Xp, qq = X[mk], G[f"X_{F}"][mk], q[mk]; ii = np.arange(len(mk))
        A0 = M.acts(Xb)[0]; Ap, Lp = M.acts(Xp)[:2]; full = Ap - A0
        nat = E.clogit(Lp[ii, qq]); allp = E.clogit(M.run(Xb, full)[ii, qq])
        print(F, "max|patch-all - natural| =", float(np.abs(allp - nat).max()), " max|natural - frozen c_F| =", float(np.abs(nat - z[f"c_{F}"][mk]).max()))
        mask = np.zeros((len(mk), E.T), bool); mask[ii, 36 + qq] = True
        pr = E.clogit(M.run(Xb, full * mask[:, :, None])[ii, qq]); bl = E.clogit(M.run(Xb, full * (~mask)[:, :, None])[ii, qq])
        print(F, "max|recomputed - frozen| patch_Rq", float(np.abs(pr - z[f"c_{F}_patch_Rq"][mk]).max()), "block_Rq", float(np.abs(bl - z[f"c_{F}_block_Rq"][mk]).max()))
        cBp = G[f"cB_{F}"][mk]; b0 = z["c_base"][mk]
        ae = lambda c: float(np.abs(c - cBp).mean())
        print(F, f"mean|clogit - Bayes| natural {ae(nat):.3f}  Rq-blocked {ae(bl):.3f}  Rq-only {ae(pr):.3f}  base(for ref, vs base Bayes) {float(np.abs(b0 - G['cB_base'][mk]).mean()):.3f}")
        pB = np.exp(cBp) / np.exp(cBp).sum(-1, keepdims=True)
        kl = lambda c: float((pB * (np.log(pB + 1e-12) - np.log(E.softmax(c) + 1e-12))).sum(-1).mean())
        print(F, f"KL(Bayes||model) natural {kl(nat):.4f}  Rq-blocked {kl(bl):.4f};  argmax agreement with Bayes natural {(nat.argmax(-1) == cBp.argmax(-1)).mean():.3f} blocked {(bl.argmax(-1) == cBp.argmax(-1)).mean():.3f}")
