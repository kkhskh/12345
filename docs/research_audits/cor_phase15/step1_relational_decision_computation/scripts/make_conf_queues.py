"""Investigation XIII: frozen fresh-cohort queues.  Training lanes T1/T2 (full_c1 seed 20 for EVERY parent, after its gated pretraining);
confirmation lanes C1/C2 (per-parent analyses, each waiting for the files it needs).  All analysis streams are new (74_000_000+)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
P = ["P160", "P161", "P162", "P163", "P164", "P165"]; S = "_eval_conf13"; EV = "logs/eval_conf13.npz"
FT = lambda p: f"checkpoints/native_train/u{p}_full_c1_s4000_seed20{S}/final.json"
waitf = lambda fs: "while " + " || ".join(f"[ ! -f {f} ]" for f in fs) + "; do sleep 30; done; "
T = {0: [], 1: []}
for i, p in enumerate(P): T[i % 2].append(waitf([f"checkpoints/z{p}/gate.json"]) + f"python3 scripts/native_train.py u{p} full_c1 4000 20 label {EV}")
for k in T: open(os.path.join(L, f"queue_train_T{k + 1}.txt"), "w").write("\n".join(T[k]) + "\n")
C = {0: [], 1: []}
for i, p in enumerate(P):
    sp = f"u{p}:full_c1:20:{S}"; tag = f"u{p}_full_c1_s4000_seed20{S}"; f0 = [FT(p)]; R = [f"logs/decoy_conf/{tag}_{r}_R.npy" for r in ("E", "A1", "PERM")]
    jobs = [waitf(f0) + f"DD_EVAL={EV} XI_OUT=decoy_conf python3 scripts/das_decoy.py {sp} {r}" for r in ("E", "A1", "PERM")]
    jobs += [waitf(f0) + f"L2_STREAM=74000000 L2_TAG=_conf_{p} python3 scripts/l2_scan.py {sp}",
             waitf(f0) + f"MC_STREAM=74100000 MC_TAG=_conf_{p} python3 scripts/l2_mix_causal.py {sp}",
             waitf(f0) + f"FA_EVAL=stream:74400000 MD_TAG=_conf_{p} python3 scripts/l2_mediation.py {sp}",
             waitf(f0 + R) + f"DECOY_DIR=logs/decoy_conf XII_RDIR=logs/none SCC_STREAM=74500000 SCC_TAG=_conf_{p} python3 scripts/score_contrast.py - {sp}",
             waitf(f0 + R) + f"DECOY_DIR=logs/decoy_conf XII_RDIR=logs/none LA_STREAM=74300000 LA_TAG=_conf_{p} python3 scripts/l1abl_mix.py - {sp}",
             waitf(f0 + R) + f"DD_STREAM=74200000 DD_TAG=_conf_{p} python3 scripts/decoy_disc.py logs/decoy_conf {sp}",
             waitf(f0) + f"AD_OUT=adapt_sel_conf AD_STREAM=74600000 python3 scripts/adapt_sel.py - {sp}"]
    C[i % 2] += jobs
C[0].append("bash scripts/conf_merge.sh")
for k in C: open(os.path.join(L, f"queue_conf_C{k + 1}.txt"), "w").write("\n".join(C[k]) + "\n")
print({k: len(v) for k, v in T.items()}, {k: len(v) for k, v in C.items()})
