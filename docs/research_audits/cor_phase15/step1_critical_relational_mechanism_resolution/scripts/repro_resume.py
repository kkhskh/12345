"""XIII-C: bitwise test of ft_factorial resume.  Run A: 160 steps uninterrupted.  Run B: same, killed right after the step-100 checkpoint,
then restarted (resumes from resume.pkl).  Compare step160 parameters.  -> logs/repro_resume.json"""
import json, os, sys, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); import numpy as np
env = dict(os.environ, XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
ck = os.path.join(HERE, "..", "checkpoints", "ft"); a_dir = os.path.join(ck, "uP161_fx_L22_W22_D22_s160"); b_dir = os.path.join(ck, "uP161_fx_L22_W22_D23_s160")
for d in (a_dir, b_dir): shutil.rmtree(d, ignore_errors=True)
ft = os.path.join(HERE, "ft_factorial.py")
subprocess.run([sys.executable, ft, "uP161", "22", "22", "22", "160"], env=env, check=True, stdout=subprocess.DEVNULL)
# run B uses a different tag only to keep directories apart; to compare, B must be the SAME run -> use same args in a fresh dir
shutil.move(a_dir, a_dir + "_A")
r1 = subprocess.run([sys.executable, ft, "uP161", "22", "22", "22", "160"], env=dict(env, FX_DIE_AFTER="100"), stdout=subprocess.DEVNULL).returncode
r2 = subprocess.run([sys.executable, ft, "uP161", "22", "22", "22", "160"], env=env, stdout=subprocess.DEVNULL).returncode
A = np.load(os.path.join(a_dir + "_A", "step160.npz")); B = np.load(os.path.join(a_dir, "step160.npz"))
d = max(float(np.abs(A[k] - B[k]).max()) for k in A.files)
res = dict(killed_rc=r1, resumed_rc=r2, max_abs_diff=d, bitwise=d == 0.0); print(res)
shutil.rmtree(a_dir); shutil.rmtree(a_dir + "_A"); json.dump(res, open(os.path.join(HERE, "..", "logs", "repro_resume.json"), "w"), indent=1)
