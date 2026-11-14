"""Amendment 1: append P155/P156 jobs (identical protocol) to the running lanes' queues.  Pretraining -> queue_pre_2; fine-tunes -> T1/T2;
per-parent confirmation analyses -> C2 (the lane without the final transfer/merge jobs, avoiding a wait deadlock)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs"); sys.argv = ["x"]
src = open(os.path.join(HERE, "make_conf_queues.py")).read()
g = {"__file__": os.path.join(HERE, "make_conf_queues.py")}; exec(compile(src.split("T1 = [")[0], "mcq", "exec"), g)
waitpre, waitf, FT, S, EV, RS = g["waitpre"], g["waitf"], g["FT"], g["S"], g["EV"], g["RS"]
NEW = ["P155", "P156"]
open(os.path.join(L, "queue_pre_2b.txt"), "w").write("".join(f"python3 scripts/cohort12_pretrain.py P {p[1:]}\n" for p in NEW))
with open(os.path.join(L, "queue_train_T1.txt"), "a") as f: f.write(waitpre("P155") + f"python3 scripts/native_train.py vP155 full_c1 4000 10 label {EV}\n" + waitpre("P155") + f"python3 scripts/native_train.py vP155 c1 4000 10 label {EV}\n")
with open(os.path.join(L, "queue_train_T2.txt"), "a") as f: f.write(waitpre("P156") + f"python3 scripts/native_train.py vP156 full_c1 4000 10 label {EV}\n" + waitpre("P156") + f"python3 scripts/native_train.py vP156 c1 4000 10 label {EV}\n")
ENV = "XII_RDIR=logs/das FA_EVAL=stream:70000000"; spec = lambda p, sp="full_c1", s=10: f"v{p}:{sp}:{s}:{S}"; C = []
for p in NEW:
    fs = [FT(p, "full_c1", 10), FT(p, "c1", 10)]; R = [f"logs/das/v{p}_full_c1_s4000_seed10{S}_E_L2_q7_k6{RS}_R.npy", f"logs/das/v{p}_c1_s4000_seed10{S}_E_L2_q7_k6{RS}_R.npy"]
    C += [waitf([FT(p, "full_c1", 10)]) + f"DAS_EVAL_STREAM=70500000 python3 scripts/das.py v{p} full_c1 10 {S} E 2 q7 6",
          waitf([FT(p, "c1", 10)]) + f"DAS_EVAL_STREAM=70500000 python3 scripts/das.py v{p} c1 10 {S} E 2 q7 6",
          waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/factorial.py {RS} {spec(p)} {spec(p, 'c1')}",
          waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/alt_ablation.py {RS} {spec(p)}",
          waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/circuit_l1.py {RS} {spec(p)}",
          waitf(fs + R) + f"{ENV} FA_TAG=_conf_{p} python3 scripts/restore.py {RS} {spec(p)}",
          waitf(fs + R) + f"XII_RDIR=logs/das DI_STREAM=70800000 DI_TAG=_conf_{p} python3 scripts/disc_ident.py {RS} {spec(p)}",
          waitf(fs + R) + f"XII_RDIR=logs/das VS_TRAIN=70100000 VS_EVAL=70600000 VS_TAG=_conf_{p} python3 scripts/vs_linear.py {RS} {spec(p)}",
          waitf(fs) + f"TRAJ_EVAL={EV} TRAJ_MODELS={spec(p)} TRAJ_TAG=_xii_{p} python3 scripts/traj_alg.py",
          waitf(fs + [f"logs/traj_alg_xii_{p}.json"]) + f"FA_EVAL=stream:70000000 FA_TAG=_conf_{p} python3 scripts/dyn_mech.py logs/traj_alg_xii_{p}.json {spec(p)}",
          waitf(fs + R) + f"XII_RDIR=logs/das AD_OUT=adapt_c2_conf AD_STREAM=71100000 python3 scripts/adapt_c2.py {RS} {spec(p)}"]
with open(os.path.join(L, "queue_conf_C2.txt"), "a") as f: f.write("\n".join(C) + "\n")
print("appended", len(C), "confirmation jobs")
