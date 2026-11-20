"""The planner's tag rules equal the harness's own tag rule for every window mode and base (1-step runs; software test only)."""
import os, sys, subprocess, tempfile
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG)
os.environ.setdefault("COR_DEVICE", "cpu")
from cor_research import env as E
E.configure("cpu")
from cor_research.jobs import plan as PL
from cor_research import protocol as PR
def test_window_tags():
    tmp = tempfile.mkdtemp(); C = PR.effective("standard"); bad = []
    for base, L in (("E", 21), ("M", 23)):
        for w in ("I", "vsgo", "rsg", "lbl"):
            env = {k: v.format(T=PL.T_of(base), lblseed=PL.lbl_seed(base, 2001)) for k, v in C["WINDOW"][w].items()}
            env = dict(os.environ, PYTHONPATH=PKG, COR_RUN_ROOT=tmp, FX_NOFALLBACK="1", FX_STOP="1", FX_TAGSUF="_s100x20", **env)
            r = subprocess.run([sys.executable, E.HARNESS, "uP160", str(L), str(L), "2001"], cwd=tmp, env=env, capture_output=True, text=True)
            want = PL.win_tag("uP160", L, 2001, w, base, C).replace("_stop100_", "_stop1_")
            if not os.path.isdir(os.path.join(tmp, "ft", want)): bad.append((base, w, want, os.listdir(os.path.join(tmp, "ft")), r.stderr[-500:]))
    assert not bad, bad
    return "ok"
if __name__ == "__main__": print(test_window_tags())
