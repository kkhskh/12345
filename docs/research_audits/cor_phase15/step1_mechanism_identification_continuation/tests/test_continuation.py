"""Tests for the continuation package. Run: python3 -m pytest -q tests/ (from the package root)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "step1_causal_bridge_deep_investigation" / "src"))

import cbmath  # noqa: E402
from cprime import P, certify, closed_form, forward_margin, interval_m, I  # noqa: E402


def test_closed_form_matches_forward():
    rng = np.random.default_rng(7)
    for a, b in rng.uniform(-3, 3, (60, 2)):
        assert abs(closed_form(a, b)["m"] - forward_margin(a, b)) < 1e-12


@pytest.mark.parametrize("ab,ref", [((0.01, 0.01), -0.6308419542), ((0.05, 0.05), -0.3576101163),
                                    ((0.08, 0.08), -0.0605622099), ((1.0, -0.001), 0.1946247153),
                                    ((0.0, 0.0), -0.6577909845), ((1.0, 0.0), 0.2651922423)])
def test_audit_counterexamples(ab, ref):
    assert abs(closed_form(*ab)["m"] - ref) < 1e-9


def test_interval_encloses_point_values():
    rng = np.random.default_rng(8)
    for a0, b0 in rng.uniform(-3, 3, (40, 2)):
        box = I(a0, a0 + 0.01), I(b0, b0 + 0.01)
        F = interval_m(*box)
        for a, b in [(a0, b0), (a0 + 0.01, b0 + 0.01), (a0 + 0.005, b0 + 0.003)]:
            c = closed_form(a, b)
            Fp = sum(math.exp(c["s"][k]) * c["v"][k] for k in "PABR")
            assert F.lo - 1e-12 <= Fp <= F.hi + 1e-12


def test_certified_rectangles():
    pos, neg, und = certify((0.09, 3.0, 0.09, 3.0), min_w=5e-3)
    assert not neg and not und
    pos, neg, und = certify((0.02, 3.0, -3.0, -0.02), min_w=5e-3)
    assert not pos and not und


def test_cprime_star_sound_fresh_points():
    from cprime_regions import suff
    rng = np.random.default_rng(9)
    for a, b in rng.uniform(-3, 3, (2000, 2)):
        s = suff(a, b)
        if s:
            assert s == np.sign(closed_form(a, b)["m"])


def test_A1star_sign_law_shared_ln():
    from mlp_norm import ln, q_of, gelu, relu
    Bs = cbmath.mean_zero_basis(6)
    e1, e2 = Bs[0], Bs[1]
    rng = np.random.default_rng(10)
    for _ in range(2000):
        x, y = rng.normal(0, 2, 2)
        w = Bs[2:].T @ rng.normal(0, 3, 3)
        z = ln(x * e1 + y * e2 + w)
        for act in (gelu, relu):
            assert np.sign(q_of(z, e1, e2, act=act)) == np.sign(x * y)


def test_crosstalk_derived_law():
    from mlp_norm import ln, q_of
    Bs = cbmath.mean_zero_basis(6)
    e1, e2, e5 = Bs[0], Bs[1], Bs[4]
    rng = np.random.default_rng(11)
    for _ in range(500):
        t12, t21 = rng.uniform(-0.6, 0.6, 2)
        f1, f2 = e1 + t12 * e2, e2 + t21 * e1
        x, y = rng.normal(0, 2, 2)
        z = ln(x * e1 + y * e2 + 3 * e5)
        u, v = z @ e1, z @ e2
        q = q_of(z, e1, e2, dirs=[f1 + f2, -(f1 + f2), f1 - f2, -(f1 - f2)])
        assert np.sign(q) == np.sign((u + t12 * v) * (v + t21 * u))


def test_sum_only_identical_on_matched_aggregate():
    mS, _, embS, coordsS = cbmath.sum_only_family()
    def S(a, b):
        u, v = coordsS(cbmath.forward(mS, embS(a, b), 2)); return u + v
    S0 = S(0.5, 0.5)
    lo, hi = -6.0, 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (S(2.5, mid) - S0 > 0) == (S(2.5, lo) - S0 > 0):
            lo = mid
        else:
            hi = mid
    b2 = (lo + hi) / 2
    m1 = cbmath.margin(cbmath.forward(mS, embS(0.5, 0.5), 2)); m2 = cbmath.margin(cbmath.forward(mS, embS(2.5, b2), 2))
    assert abs(m1 - m2) < 1e-9


def test_total_margin_theorem_qk3():
    from program_b import qk3, run
    from semantics import sample_k
    m, emb = qk3(); mo, embo = qk3(qk_off=True)
    bk = sample_k("T5b", 200, np.random.default_rng(12))
    for zA, zB in zip(bk.zA, bk.zB):
        k = int(zA.argmax()); j = int(np.argsort(zA)[-2])
        t, to = run(m, emb, zA, zB), run(mo, embo, zA, zB)
        dm = (t.logits[k] - t.logits[j]) - (to.logits[k] - to.logits[j])
        vm = float(t.VO[1][0][1] @ (m.WU[:, k] - m.WU[:, j]))
        if abs(dm) > 1e-9:
            assert np.sign(dm) == np.sign(t.pattern[1][0][2, 1] - to.pattern[1][0][2, 1]) * np.sign(vm)


def test_relation_free_rule_construction():
    for K in (2, 3, 5):
        for kA in range(K):
            for kB in range(K):
                for p in range(K):
                    s = np.zeros(K); s[kA] += 1; s[kB] += 1; s[p] += 1.5
                    want = kA if kA == kB else p
                    assert int(np.argmax(s)) == want


def test_xnor_not_additive_lp():
    from additive_lp import feasible
    assert not feasible("T1", 0, 12)
    assert feasible("T3", 1, 12) and feasible("T3", -1, 12) and feasible("T4", 0, 12)


def test_relation_free_twin_matches_cprime():
    from relation_free_equivalent import relation_free
    mR, emb = relation_free()
    rng = np.random.default_rng(13)
    for a, b in rng.uniform(-3, 3, (400, 2)):
        if (a > 0 and b > 0 and min(a, b) < 0.09) or (a * b < 0 and min(abs(a), abs(b)) < 0.02):
            continue
        assert np.sign(closed_form(a, b)["m"]) == np.sign(cbmath.margin(cbmath.forward(mR, emb(a, b), 3)))


def test_bilinear_impossibility_witness():
    # complement of the agreement cone is not convex: p, q outside C0, midpoint inside
    p, q = np.array([0, 1, -2.0]), np.array([0, -2, 1.0])
    assert p.argmax() != 0 and q.argmax() != 0 and ((p + q) / 2).argmax() == 0
