export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
python3 - <<'PY' > logs/expl/gradsig_jobs.txt
import json; d=json.load(open("logs/score_XV.json")); D=json.load(open("logs/XV_design.json"))
for r in D["runs"]:
    m=d["models"][f"{r['cond']}_s{r['seed']}"]; print(r["spec"], r["dir"], f"0,{m['t_S']}", r["seed"], f"logs/expl/gradsig_{r['cond']}_s{r['seed']}.json")
PY
while read c dir st s out; do python3 scripts/xv_gradsig.py $c $dir $st $s $out > /dev/null 2>&1; done < logs/expl/gradsig_jobs.txt
python3 scripts/x_mech.py 2:0.9:0.05 confirm_outputs/A_s201 825 logs/expl/mech_A_s201.json > /dev/null 2>&1
python3 scripts/x_mech.py 2:0.9:0.3333333333333333 confirm_outputs/B_s201 500 logs/expl/mech_B_s201.json > /dev/null 2>&1
python3 scripts/x_mech.py 3:0.9:0.3333333333333333 confirm_outputs/C_s204 2500 logs/expl/mech_C_s204.json > /dev/null 2>&1
python3 scripts/x_mech.py 3:0.9:0.05 confirm_outputs/D_s203 2250 logs/expl/mech_D_s203.json > /dev/null 2>&1
python3 scripts/x_mech.py 3:0.9:0.3333333333333333 confirm_outputs/C_s201 4000 logs/expl/mech_C_s201_censored.json > /dev/null 2>&1
echo done > logs/expl/DONE
