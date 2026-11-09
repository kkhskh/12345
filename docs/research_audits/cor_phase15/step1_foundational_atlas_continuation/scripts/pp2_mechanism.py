"""PP2 internal reason (descriptive; on p_c = 0 episodes, so NOT part of the frozen predictor): does the model's internal
colluder code commit to some group when no colluder exists?  commit = mean max probe probability on p_c = 0 episodes
(1/3 = no commitment).  Also: commitment split by whether the top-agreement group is unique.  Writes logs/pp2_mechanism.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, pp2_sig as S, gen_family as G
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
sys.path.insert(0, HERE); from pp2_dev_models import M
dev = json.load(open(os.path.join(HERE, "..", "logs", "pp2_dev.json"))); out = {}
f4 = S.f4; cf = f4.configs
d = G.sample(f4, 4000, np.random.default_rng(6_400_000)); h = np.array([cf[c][1] for c in d["cfg"]])
p0 = G.sample(G.Fam(pc=0.0), 3000, np.random.default_rng(6_200_002))["X"]
cnt = np.stack([(p0[:, :, a] == p0[:, :, b]).sum(1) for a, b in G.GROUPS], -1); top = np.sort(cnt, -1); uniq = top[:, 2] > top[:, 1]
for nm, (k, p) in M.items():
    ap = S.load(k, os.path.join(S.LR, p)); _, r = ap(d["X"]); mu, sd = r.mean(0), r.std(0) + 1e-6
    pr = LogisticRegression(max_iter=2000).fit((r - mu) / sd, h); _, r0 = ap(p0); P = pr.predict_proba((r0 - mu) / sd)
    out[nm] = dict(commit=float(P.max(1).mean()), commit_unique_top=float(P.max(1)[uniq].mean()), commit_tied_top=float(P.max(1)[~uniq].mean()),
                   picks_top_agreement_group=float((P.argmax(1) == cnt.argmax(1))[uniq].mean()), local_index=dev[nm]["local_index"])
    print(nm, {a: round(b, 3) for a, b in out[nm].items()}, flush=True)
n = list(M); out["spearman_commit_vs_local"] = float(spearmanr([out[x]["commit"] for x in n], [out[x]["local_index"] for x in n])[0])
print(out["spearman_commit_vs_local"]); json.dump(out, open(os.path.join(HERE, "..", "logs", "pp2_mechanism.json"), "w"), indent=1)
