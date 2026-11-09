"""Review response (R2-causal F11): the PP2 'picks the top-agreement group' index computed with the same probe pipeline on
RAW INPUT STATISTICS instead of model states (no model).  If raw graded counts already give 1.00, the index measures which
statistic is linearly decodable, not a configuration constraint.  Writes logs/pp2_probe_control.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_relational_computation_atlas", "src"))
import numpy as np, gen_family as G
from sklearn.linear_model import LogisticRegression
f4 = G.Fam(); cf = f4.configs
d = G.sample(f4, 4000, np.random.default_rng(6_400_000)); h = np.array([cf[c][1] for c in d["cfg"]])
p0 = G.sample(G.Fam(pc=0.0), 3000, np.random.default_rng(6_200_002))["X"]
def stats(X, kind):
    A = np.stack([(X[:, :, a] == X[:, :, b]) for a, b in G.GROUPS], -1).astype(float)
    if kind == "counts": return A.sum(1)
    if kind == "per_question_indicators": return A.reshape(len(X), -1)
    return A.all(1).astype(float)                                   # all-agree indicator (T4-sufficient)
cnt = np.stack([(p0[:, :, a] == p0[:, :, b]).sum(1) for a, b in G.GROUPS], -1); srt = np.sort(cnt, -1); uniq = srt[:, 2] > srt[:, 1]
out = {}
for kind in ("counts", "per_question_indicators", "all_agree_indicator"):
    Ztr = stats(d["X"], kind); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6; pr = LogisticRegression(max_iter=2000).fit((Ztr - mu) / sd, h)
    P = pr.predict_proba((stats(p0, kind) - mu) / sd)
    out[kind] = dict(train_acc=float(pr.score((Ztr - mu) / sd, h)), picks_top_agreement_group=float((P.argmax(1) == cnt.argmax(1))[uniq].mean()), commit=float(P.max(1).mean()))
    print(kind, out[kind])
json.dump(out, open(os.path.join(HERE, "..", "logs", "pp2_probe_control.json"), "w"), indent=1)
