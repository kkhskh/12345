"""Track C reproduces the reference classification and key exact values (CPU, a few minutes)."""
import os, sys, json, subprocess, tempfile
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG)
def test_track_c():
    out = os.path.join(tempfile.mkdtemp(), "c.json")
    r = subprocess.run([sys.executable, "-m", "cor_research.mathematics.run_track_c", out], cwd=PKG, env=dict(os.environ, PYTHONPATH=PKG, COR_DEVICE="cpu"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    R = json.load(open(out)); ref = json.load(open(os.path.join(PKG, "tests", "fixtures", "track_c_math_reference.json")))
    assert R["classification"] == ref["classification"], (R["classification"], ref["classification"])
    for k in ref["repaired_gate"]:
        for f in ("R1_max_err_signed_identity", "R5_max_err_deletion_effect", "O_positive_mass", "R4_O0_deletion_invariance_violations"):
            assert abs(R["repaired_gate"][k][f] - ref["repaired_gate"][k][f]) < 1e-12, (k, f)
    assert R["t4_reference"]["closed_vs_brute_max_abs"] < 1e-12
    return R["classification"]
if __name__ == "__main__": print(test_track_c())
