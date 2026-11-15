"""XIII-B frozen confirmation queues (seed-21 fine-tunes; new streams 79_000_000+; new evaluation set logs/eval_13b.npz).
Two lanes (K1, K2); every job waits for its fine-tune's final.json.  Outputs are tagged _conf13b and named as score_13b.py expects."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
P = ["P160", "P161", "P163", "P164", "P165"]; SFX = "_conf13b"
spec = lambda p: f"u{p}:full_c1:21:_eval_conf13"; fin = lambda p: f"checkpoints/native_train/u{p}_full_c1_s4000_seed21_eval_conf13/final.json"
wait = lambda fs: "while " + " || ".join(f"[ ! -f {f} ]" for f in fs) + "; do sleep 30; done; "
Q = {0: [], 1: []}
for i, p in enumerate(P):
    sp = spec(p); w = wait([fin(p)]); k = i % 2
    Q[k] += [w + f"CF_STREAM=79000000 CF_TAG={SFX}_{p} python3 scripts/class_features.py {sp}",
             w + f"FF_STREAM=79100000 FF_TAG={SFX}_{p} python3 scripts/fit_family.py {sp}",
             w + f"L2C_STREAM=79200000 L2C_TAG={SFX}_{p} python3 scripts/l2_components.py {sp}",
             w + f"DB_TRAIN=79300000 DB_EVAL=79400000 DB_TAG={SFX} python3 scripts/das_bal.py {sp}",
             w + f"AD_BETA=0.6 AD_CONDS=full,early01 AD_STREAM=79500000 AD_OUT={SFX} python3 scripts/adapt2.py {sp}"]
Q[0].append(wait([fin(p) for p in P]) + f"python3 scripts/algorithm_discrimination.py logs/eval_13b.npz " + " ".join(spec(p) for p in P) + f" && mv logs/algdisc_eval_13b.json logs/algdisc{SFX}.json")
for k in Q: open(os.path.join(L, f"queue_conf13b_K{k + 1}.txt"), "w").write("\n".join(Q[k]) + "\n")
print({k: len(v) for k, v in Q.items()})
