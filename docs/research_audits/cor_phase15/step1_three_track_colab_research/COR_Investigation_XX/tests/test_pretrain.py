"""Pretraining wrapper == train_tasks.train (bitwise on CPU), including a crash + resume.  Smoke test (40 steps); not evidence."""
import os, sys, subprocess, tempfile, json
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def test_pretrain_equivalence_and_resume(tmp=None):
    tmp = tmp or tempfile.mkdtemp(); env = dict(os.environ, COR_RUN_ROOT=tmp, COR_PARENT_ROOT=os.path.join(tmp, "parents"), COR_DEVICE="cpu", COR_PRETRAIN_CKPT="10", PYTHONPATH=PKG)
    r = subprocess.run([sys.executable, "-m", "cor_research.training.pretrain", "7", "40"], env=dict(env, COR_DIE_AFTER="20"), cwd=PKG, capture_output=True, text=True); assert r.returncode == 3, r.stderr[-2000:]
    r = subprocess.run([sys.executable, "-m", "cor_research.training.pretrain", "7", "40"], env=env, cwd=PKG, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
    code = f"""
import sys; sys.path.insert(0, {PKG!r})
from cor_research import env as E; E.configure('cpu'); E.legacy_paths()
import numpy as np, jax, train_tasks as TT, ckpt as C
p, h, _ = TT.train('pcount', 7, 40); q = C.load({os.path.join(tmp, 'parents', 'zP7', 'pre.npz')!r})
a = jax.tree_util.tree_leaves(p); b = jax.tree_util.tree_leaves(q); assert len(a) == len(b)
print(max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(a, b)))"""
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=PKG, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
    d = float(r.stdout.strip().split()[-1]); assert d == 0.0, d
    assert json.load(open(os.path.join(tmp, "parents", "zP7", "resume_log.json")))[0]["resumed_at"] == 20
    return dict(max_abs_diff=d, resumed_at=20)
if __name__ == "__main__": print(test_pretrain_equivalence_and_resume())
