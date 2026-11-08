"""Baseline curriculum: mlp_grouphist trained 6000 steps on T1, then 3000 steps on T4 (same optimiser/schedule).
Usage: curriculum_base.py SEED OUTDIR"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, jax
import nets as N, ckpt as C, bayes as B
seed, out = int(sys.argv[1]), sys.argv[2]; os.makedirs(out, exist_ok=True)
p1, h1, d1 = N.train("T1", "M", seed, 6000, arch="mlp_grouphist")
e1 = N.evaluate(p1, "T1", seed=3_300_003, arch="mlp_grouphist")
# T4 has 6 R values: the last layer's R logits differ in size -> re-initialise only the output layer's R slice
fresh = N.init_base("mlp_grouphist", "M", B.Env("T4"), seed)
p0 = jax.tree_util.tree_map(np.asarray, p1); last = p0["net"][-1]; fl = jax.tree_util.tree_map(np.asarray, fresh["net"][-1])
nshared = last["W"].shape[1] - 3
last["W"] = np.concatenate([last["W"][:, :nshared], fl["W"][:, nshared:]], 1); last["b"] = np.concatenate([last["b"][:nshared], fl["b"][nshared:]])
p2, h2, d2 = N.train("T4", "M", seed, 3000, arch="mlp_grouphist", p0=p0)
e2 = N.evaluate(p2, "T4", seed=3_300_004, arch="mlp_grouphist")
json.dump(dict(seed=seed, stage1_T1=e1, stage2_T4=e2, seconds=d1 + d2), open(os.path.join(out, f"mlp_grouphist_curriculum_s{seed}.json"), "w"), indent=1)
print(seed, e1["acc"], e2["acc"], e2["bayes_acc"], e2["own_question_bayes_acc"])
