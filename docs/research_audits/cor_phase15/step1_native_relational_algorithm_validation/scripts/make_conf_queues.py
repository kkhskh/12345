"""Writes the frozen fresh-cohort queues (Investigation XII).  Training lanes T1/T2 and confirmation lanes C1/C2.
Every confirmation job waits for the files it needs (final.json of the fine-tunes, R files), so lanes can be started in any order.
Same-parent replicates (seed 11): first two COMPLETE parents in seed order, decided by scripts/pick_replicates.py from pre.json only."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
# (Amendment 1 adds P155/P156 via scripts/amend1_queues.py)
P = ["P150", "P151", "P152", "P153", "P154"]; S = "_eval_conf12"; EV = "logs/eval_conf12.npz"; RS = "_ev70500000"
FT = lambda p, spec, seed: f"checkpoints/native_train/v{p}_{spec}_s4000_seed{seed}{S}/final.json"
waitpre = lambda p: f"while [ ! -f checkpoints/z{p}/pre.json ]; do sleep 30; done; "
waitf = lambda fs: "while " + " || ".join(f"[ ! -f {f} ]" for f in fs) + "; do sleep 30; done; "
T1 = [waitpre("P150") + f"python3 scripts/native_train.py vP150 full_c1 4000 10 label {EV}",
      waitpre("P152") + f"python3 scripts/native_train.py vP152 full_c1 4000 10 label {EV}",
      waitpre("P154") + f"python3 scripts/native_train.py vP154 full_c1 4000 10 label {EV}",
      "bash scripts/rep_train.sh 0"]
T2 = [waitpre("P151") + f"python3 scripts/native_train.py vP151 full_c1 4000 10 label {EV}",
      waitpre("P150") + f"python3 scripts/native_train.py vP150 c1 4000 10 label {EV}",
      waitpre("P151") + f"python3 scripts/native_train.py vP151 c1 4000 10 label {EV}",
      waitpre("P153") + f"python3 scripts/native_train.py vP153 full_c1 4000 10 label {EV}",
      waitpre("P152") + f"python3 scripts/native_train.py vP152 c1 4000 10 label {EV}",
      waitpre("P153") + f"python3 scripts/native_train.py vP153 c1 4000 10 label {EV}",
      waitpre("P154") + f"python3 scripts/native_train.py vP154 c1 4000 10 label {EV}",
      "bash scripts/rep_train.sh 1"]
open(os.path.join(L, "queue_train_T1.txt"), "w").write("\n".join(T1) + "\n"); open(os.path.join(L, "queue_train_T2.txt"), "w").write("\n".join(T2) + "\n")
# confirmation, per parent (all parents; scorer separates complete / incomplete)
ENV = f"XII_RDIR=logs/das FA_EVAL=stream:70000000"
C = []
spec = lambda p, sp="full_c1", s=10: f"v{p}:{sp}:{s}:{S}"
for p in P:
    fs = [FT(p, "full_c1", 10), FT(p, "c1", 10)]
    C.append(waitf([FT(p, "full_c1", 10)]) + f"DAS_EVAL_STREAM=70500000 python3 scripts/das.py v{p} full_c1 10 {S} E 2 q7 6")
    C.append(waitf([FT(p, "c1", 10)]) + f"DAS_EVAL_STREAM=70500000 python3 scripts/das.py v{p} c1 10 {S} E 2 q7 6")
    R = [f"logs/das/v{p}_full_c1_s4000_seed10{S}_E_L2_q7_k6{RS}_R.npy", f"logs/das/v{p}_c1_s4000_seed10{S}_E_L2_q7_k6{RS}_R.npy"]
    C.append(waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/factorial.py {RS} {spec(p)} {spec(p, 'c1')}")
    C.append(waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/alt_ablation.py {RS} {spec(p)}")
    C.append(waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/circuit_l1.py {RS} {spec(p)}")
    C.append(waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/restore.py {RS} {spec(p)}")
    C.append(waitf(fs + R) + f"XII_RDIR=logs/das DI_STREAM=70800000 DI_TAG=_conf_{p} python3 scripts/disc_ident.py {RS} {spec(p)}")
    C.append(waitf(fs + R) + f"XII_RDIR=logs/das VS_TRAIN=70100000 VS_EVAL=70600000 VS_TAG=_conf_{p} python3 scripts/vs_linear.py {RS} {spec(p)}")
    C.append(waitf(fs) + f"TRAJ_EVAL={EV} TRAJ_MODELS={spec(p)} TRAJ_TAG=_xii_{p} python3 scripts/traj_alg.py")
    C.append(waitf(fs + [f"logs/traj_alg_xii_{p}.json"]) + f"FA_EVAL=stream:70000000 FA_TAG=_conf_{p} python3 scripts/dyn_mech.py logs/traj_alg_xii_{p}.json {spec(p)}")
    C.append(waitf(fs + R) + f"XII_RDIR=logs/das AD_OUT=adapt_c2_conf AD_STREAM=71100000 python3 scripts/adapt_c2.py {RS} {spec(p)}")
# replicates + transfer (wait for replicate list)
half = [C[i] for i in range(len(C)) if i % 2 == 0], [C[i] for i in range(len(C)) if i % 2 == 1]
half[0].extend(["bash scripts/conf_replicates.sh", "bash scripts/conf_transfer.sh", "bash scripts/conf_merge.sh"])
open(os.path.join(L, "queue_conf_C1.txt"), "w").write("\n".join(half[0]) + "\n"); open(os.path.join(L, "queue_conf_C2.txt"), "w").write("\n".join(half[1]) + "\n")
print(len(T1), len(T2), len(C))
