"""Package-only tests. They use nothing outside this package and never import the research repo.

    python -m pytest tests/test_paper_reproducibility.py -q

No test is skipped and no failure is caught and ignored. Comparison against the research
repository's own implementation lives in scripts/integration_check_against_repo.py instead.
"""
from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
SCRIPTS = PKG / "scripts"
sys.path.insert(0, str(SCRIPTS))

import p1lib  # noqa: E402
import verify_paper_claims as vpc  # noqa: E402


@pytest.fixture(scope="module")
def recomputed():
    return p1lib.compute_all()


# ------------------------------------------------------------------ primitives
def test_softmax_and_divergences():
    rng = np.random.default_rng(0)
    p = p1lib.softmax(rng.normal(size=61) * 3)
    q = p1lib.softmax(rng.normal(size=61) * 3)
    assert p.sum() == pytest.approx(1.0)
    assert p1lib.softmax(np.array([1000.0, 1000.0])) == pytest.approx([0.5, 0.5])
    assert p1lib.js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert p1lib.js_divergence(p, q) == pytest.approx(p1lib.js_divergence(q, p))
    assert 0 < p1lib.js_divergence(p, q) <= np.log(2)
    assert p1lib.js_divergence([1, 0], [0, 1]) == pytest.approx(np.log(2), rel=1e-9)
    assert p1lib.total_variation(p, q) == pytest.approx(0.5 * np.abs(p - q).sum())


def test_token_mapping_is_bijective_onto_moves():
    toks = [p1lib.square_to_token(s) for s in range(64) if s not in p1lib.CENTER_SQUARES]
    assert sorted(toks) == list(range(1, 61))
    assert p1lib.square_to_token(0) == 1 and p1lib.square_to_token(63) == 60
    with pytest.raises(p1lib.AnalysisError):
        p1lib.square_to_token(27)


def test_y_components_on_a_hand_example():
    probs = np.zeros(61)
    probs[p1lib.square_to_token(0)] = 0.25
    probs[p1lib.square_to_token(1)] = 0.10
    probs[p1lib.square_to_token(2)] = 0.65
    out = p1lib.y_components(probs, became_legal_squares=[0], became_illegal_squares=[1, 2])
    assert (out["G"], out["Q"], out["Y"]) == pytest.approx((0.25, 0.75, -0.5))
    with pytest.raises(p1lib.AnalysisError):
        p1lib.y_components(probs, became_legal_squares=[], became_illegal_squares=[1])


def test_cluster_bootstrap_matches_its_definition():
    vals = np.array([0.1, 0.3, -0.2, 0.4, 0.0, 0.2])
    clusters = ["a", "a", "b", "c", "c", "d"]
    out = p1lib.cluster_bootstrap(vals, clusters, n_boot=500, seed=7)
    cm = np.array([0.2, -0.2, 0.2, 0.2])
    rng = np.random.default_rng(7)
    means = [cm[rng.choice(np.arange(4), size=4, replace=True)].mean() for _ in range(500)]
    assert out["ci95_low"] == pytest.approx(np.quantile(means, 0.025))
    assert out["ci95_high"] == pytest.approx(np.quantile(means, 0.975))
    assert out["point"] == pytest.approx(vals.mean())
    with pytest.raises(p1lib.AnalysisError):
        p1lib.cluster_bootstrap([np.nan], ["a"])


# ------------------------------------------------------------------ isolation
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_package_module_imports_the_repository():
    offenders = [str(p.relative_to(PKG)) for p in PKG.rglob("*.py")
                 if p.name != "integration_check_against_repo.py" and "src" in _imports(p)]
    assert offenders == []


def test_analysis_runs_without_repository_on_path():
    code = ("import sys; sys.path.insert(0, %r); import p1lib; p1lib.verify_raw_hashes(); "
            "assert 'src' not in sys.modules; print('ok')") % str(SCRIPTS)
    proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=tempfile.gettempdir(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# ------------------------------------------------------------------ recomputation
def test_raw_hashes_match_manifest():
    out = p1lib.verify_raw_hashes()
    assert out["ok"] == 1 and out["files_checked"] > 100


def test_recomputation_equals_shipped_analysis(recomputed):
    saved = json.loads((PKG / "analysis" / "results.json").read_text(encoding="utf-8"))
    assert vpc.deep_compare(p1lib.strip_rows(recomputed), saved) == []


def test_recomputed_intervals_match_original_runs(recomputed):
    env = vpc.ledger_env(p1lib.strip_rows(recomputed))
    assert env["MAXBOOTDIFF"]() < 1e-7


def test_headline_numbers(recomputed):
    fj = recomputed["fixed_js"]["slices"]["pooled"]
    nst = recomputed["nst"]["slices"]["pooled"]
    assert fj["delta_specific"]["point"] == pytest.approx(0.0265506, abs=1e-6)
    assert fj["delta_specific"]["ci95_low"] > 0 and fj["delta_random"]["ci95_low"] > 0
    assert nst["wrong"]["point"] == pytest.approx(0.0200570, abs=1e-6)
    assert recomputed["nst"]["summary"]["mean_Y"]["learned"] < 0
    assert recomputed["stage_share"]["share"] == pytest.approx(0.5604, abs=1e-4)


def test_probe_invariance_is_relative_to_editor_a(recomputed):
    s = recomputed["fixed_js"]["summary"]
    assert s["probe_max_abs_vs_editor_A"] < 1e-13
    assert s["probe_max_abs_vs_natural_recipient_min"] > 0.5


# ------------------------------------------------------------------ tables, figures, manuscript
def test_tables_regenerate_byte_identically():
    vpc.failures.clear()
    vpc.passes.clear()
    vpc.tables_regenerate()
    assert vpc.failures == []


def test_figures_regenerate_byte_identically():
    rf = importlib.import_module("reproduce_figures")
    prov = json.loads((PKG / "figures" / "figure_provenance.json").read_text(encoding="utf-8"))
    import matplotlib
    assert prov["environment"]["matplotlib"] == matplotlib.__version__, "figures were drawn with another matplotlib version"
    with tempfile.TemporaryDirectory() as td:
        old = rf.FIG
        rf.FIG = Path(td)
        try:
            out = {}
            for f in (rf.figure1, rf.figure2, rf.figure3, rf.figure4, rf.figure5):
                f(out)
        finally:
            rf.FIG = old
        for p in sorted(Path(td).glob("*.p*")):
            assert p.read_bytes() == (PKG / "figures" / p.name).read_bytes(), p.name


def test_manuscript_structure():
    md = (PKG / "manuscript" / "paper_master.md").read_text(encoding="utf-8")
    heads = [l for l in md.splitlines() if l.startswith("#")]
    assert not [h for h in heads if h.lstrip("#").strip()[:1].isdigit()]
    assert len(heads) == len(set(heads))
    assert "unedited recipient" not in md
    assert "Editor A reference activation" in md
    for letter in vpc.LETTERS:
        assert f"`{letter}`" in md


def test_tex_is_generated_from_current_markdown():
    tex = (PKG / "manuscript" / "paper_master.tex").read_text(encoding="utf-8")
    assert vpc.sha256(PKG / "manuscript" / "paper_master.md") in tex.splitlines()[0]
    assert "\\section{4.3" not in tex and "\\subsection{4.3" not in tex


def test_verify_paper_claims_passes():
    proc = subprocess.run([sys.executable, str(SCRIPTS / "verify_paper_claims.py")], cwd=tempfile.gettempdir(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert " 0 failed" in proc.stdout


@pytest.mark.parametrize("old,new,expect", [
    ("\\(+0.0266\\)", "\\(+0.0276\\)", "0.0276"),
    ("in 13 of 32 triplets (40.6%)", "in 19 of 32 triplets (40.6%)", "uncovered"),
    ("`SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED`", "`SYMMETRIC_TARGET_SWITCHING_SUPPORTED`", "letter verbatim"),
])
def test_verifier_rejects_altered_manuscript(old, new, expect):
    md = (PKG / "manuscript" / "paper_master.md").read_text(encoding="utf-8")
    assert old in md
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "paper_master.md"
        p.write_text(md.replace(old, new), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(SCRIPTS / "verify_paper_claims.py"), "--manuscript", str(p)],
                              cwd=td, capture_output=True, text=True)
    assert proc.returncode == 1
    assert expect in proc.stdout
