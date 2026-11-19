"""XIX functional (route-independent) diagnostic.  For a list of models, compute predictive distributions on a probe set drawn from a
given stream, and the symmetric-KL functional distance between every pair of models, plus each model's KL to the exact Bayes posterior.
Probe set: 1500 natural episodes (one question each) plus 1500 relational-edit episodes (an agreement-changing edit of one source pair, the
XIII-C 'PR' construction), so that differences in how evidence is combined are exposed, not only accuracy.
Usage: func19.py STREAM OUT.npz SPEC [SPEC ...]   SPEC = dir@step@parent   (writes probs, names, Bayes posteriors)"""
import os, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import numpy as np, jax, jax.numpy as jnp
import xviiim as XM
stream, out = int(sys.argv[1]), sys.argv[2]; specs = sys.argv[3:]
S = XM.MT.sets(stream, n=1500); PR = S["PR"]
Xn, qn, PBn = S["Xn"][:1500], S["qn"][:1500], S["PBn"][:1500]
Xp = np.concatenate([Xn, PR["Xd"]]); qp = np.concatenate([qn, PR["q"]]); PBp = np.concatenate([PBn, PR["PBd"]])
P = []
for sp in specs:
    d, st, par = sp.split("@"); M = XM.load_dir(d, par, int(st)); f = jax.jit(lambda X_: XM.F.forward(M, X_))
    L = np.asarray(XM.F.batched(f, Xp))[np.arange(len(qp)), qp]; P.append(XM.RP.sm(L))
np.savez(out, probs=np.array(P), names=np.array(specs), bayes=PBp, n_natural=len(qn))
