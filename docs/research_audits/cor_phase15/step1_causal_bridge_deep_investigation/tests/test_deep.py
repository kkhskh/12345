"""Checks of the theorems in THEOREMS_AND_COUNTEREXAMPLES.md.
A passing test checks an implementation or an explicit construction. It is not a proof of a universal claim."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import positive_control_ref as ref  # noqa: E402
from cbmath import (  # noqa: E402
    agreement_family, even_part, forward, gauge_head_redistribution, gauge_qk, gauge_relu_rescale,
    gauge_residual_rotation, gauge_split_neuron, gauge_vo, is_relation_blind, layernorm, margin,
    mean_zero_basis, positive_control, qk_gate, random_orthogonal_fixing_one, reconciliation,
    replace_site, sum_only_family,
)

BIN = [(a, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)]

# ---------------------------------------------------- correction A: same cut, no bypass


def test_SSstar_bypass_counterexample_and_no_bypass_invariance():
    c1, c2 = 2.0, -1.0
    S, Y = c1 + c2, (c1 + c2) + c1
    S2, Y2 = (c1 + 3) + (c2 - 3), ((c1 + 3) + (c2 - 3)) + (c1 + 3)
    assert S == S2 and Y != Y2                       # a bypass breaks the old statement
    # no bypass: heads are consumed only through the residual add -> zero-sum edits are inert
    m, D, embed = qk_gate(n_heads2=2)
    X, r = embed(0.7, -0.3)
    base = forward(m, X, r)
    rng = np.random.default_rng(0)
    for _ in range(10):
        Dl = rng.normal(0, 2, base.head_out[1].shape); Dl -= Dl.mean(axis=0, keepdims=True)
        tr = forward(m, X, r, lambda li, s, a: a + Dl if (li == 1 and s == "head_out") else a)
        assert np.allclose(tr.logits, base.logits, atol=1e-12)

# ---------------------------------------------------- positive control


@pytest.mark.parametrize("final_ln", [True, False])
def test_positive_control_signs_and_decomposition(final_ln):
    model, (e1, e2, e3), embed = positive_control(final_ln=final_ln)
    for a, b in BIN:
        tr = forward(model, embed(a, b), 2)
        assert np.sign(margin(tr)) == (1 if a == b else -1)
        assert np.allclose(tr.mid[0][2], a * e1 + b * e2, atol=1e-12)        # read receives a e1 + b e2
        assert np.allclose(tr.pattern[0][0][2], [1 / 3] * 3)                  # uniform attention
        assert abs(tr.mid[0][2] @ e3) < 1e-12                                 # nothing on e3 before the MLP
    if final_ln:
        assert np.isclose(margin(forward(model, embed(1.0, 1.0), 2)), 3.2142216080, atol=1e-9)


def test_positive_control_matches_independent_pure_python_implementation():
    model, _, embed = positive_control(final_ln=True)
    for a, b in BIN:
        assert np.isclose(margin(forward(model, embed(a, b), 2)), ref.margin(a, b), atol=1e-12)


def test_positive_control_attention_clamps_and_mlp_ablation():
    model, _, embed = positive_control(final_ln=True)
    rng = np.random.default_rng(1)
    for _ in range(20):
        w = rng.dirichlet([1, 1, 1])
        hk = lambda li, s, A, w=w: (np.vstack([A[:2], w]) if s == "pattern0" else A)
        for a, b in BIN:
            assert np.sign(margin(forward(model, embed(a, b), 2, hk))) == (1 if a == b else -1)
    for a, b in BIN:
        assert abs(margin(forward(model, embed(a, b), 2, replace_site(0, "mlp_out", np.zeros((3, 4)))))) < 1e-12

# ---------------------------------------------------- Theorem A1 (agreement law) and its boundaries


def test_A1_identity_abs_sum_vs_abs_difference():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=10000), rng.normal(size=10000)
    assert np.all(np.sign(np.abs(u + v) - np.abs(u - v)) == np.sign(u * v))


@pytest.mark.parametrize("act", ["gelu", "gelu_erf", "relu", "silu", "softplus"])
def test_A1_sign_law_in_recomputed_block(act):
    m, _, emb, coords = agreement_family(act=act)
    for a in np.linspace(-1.5, 1.5, 9):
        for b in np.linspace(-1.5, 1.5, 9):
            if a == 0 or b == 0:
                continue
            tr = forward(m, emb(a, b), 2)
            x, y = coords(tr)
            assert np.sign(margin(tr)) == np.sign(x * y) == np.sign(a * b)
            if act == "relu":
                assert np.isclose(margin(tr), 2 * np.sign(x * y) * min(abs(x), abs(y)), atol=1e-12)


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
def test_A1_fails_for_odd_or_constant_even_part(act):
    m, _, emb, _ = agreement_family(act=act)
    assert max(abs(margin(forward(m, emb(a, b), 2))) for a, b in BIN) < 1e-12


def test_A1_each_removed_condition_has_a_witness():
    """Removing any one condition of Theorem A1 produces a cell with sign(m) != sign(ab).
    The witness is searched for on a fine grid (existence claim), not hard-coded."""
    e1, e2 = mean_zero_basis(6)[:2]
    variants = {
        "skewed read": dict(read_dirs=[2 * e1 + e2, -(2 * e1 + e2), e1 - e2, -(e1 - e2)]),
        "read offset": dict(read_offset=[0.3, 0.3, 0, 0]),
        "unequal out": dict(out_w=(1.3, 1.3, -1, -1)),
        "direct path": dict(readout_mix=0.5),
        "LN bias": dict(final_ln=True, kappa_bias=0.3),
        "odd activation": dict(act="tanh"),
    }
    fine = [v for v in np.linspace(-1.5, 1.5, 61) if abs(v) > 1e-9]
    for name, kw in variants.items():
        m, _, emb, _ = agreement_family(**kw)
        wit = next(((a, b) for a in fine for b in fine
                    if np.sign(margin(forward(m, emb(a, b), 2))) != np.sign(a * b)), None)
        assert wit is not None, name


def test_A_not_candidate_relative():
    m, _, emb, _ = agreement_family()
    for a in (0.3, 1.2):
        for b in (-0.7, 0.9):
            assert np.isclose(margin(forward(m, emb(a, b), 2)), margin(forward(m, emb(-a, -b), 2)), atol=1e-12)

# ---------------------------------------------------- Theorems A2-A4 (what the binary control cannot show)


def test_A2_binary_indistinguishability_from_sum_only_reader():
    md, (e1, e2, e3), embed = positive_control(final_ln=False)
    ms, _, _ = positive_control(final_ln=False)
    L = ms.layers[0]; s2 = np.sqrt(0.5 + ms.eps)
    L.Win = s2 * np.column_stack([e1 + e2, -(e1 + e2)]); L.bin = np.zeros(2)
    L.Wout = np.stack([2 * e3, 2 * e3])
    h2 = float(even_part("gelu", np.array(2.0)))
    ms.bU = np.array([-h2, h2])
    for a, b in BIN:
        assert np.isclose(margin(forward(md, embed(a, b), 2)), margin(forward(ms, embed(a, b), 2)), atol=1e-12)


def test_A3_interaction_and_odd_odd_component_do_not_separate():
    h = lambda t: even_part("gelu", t)
    q = lambda x, y: h(x + y) - h(x - y)
    s = lambda x, y: h(x + y)
    Eoo = lambda f, x, y: (f(x, y) - f(-x, y) - f(x, -y) + f(-x, -y)) / 4
    I = lambda f, x, y: f(x, y) - f(x, 0) - f(0, y) + f(0, 0)
    rng = np.random.default_rng(2)
    for x, y in rng.uniform(-2, 2, (200, 2)):
        assert np.isclose(Eoo(s, x, y), q(x, y) / 2)                    # exact identity
        assert np.isclose(Eoo(q, x, y), q(x, y))
        assert np.sign(I(q, x, y)) == np.sign(x * y)
    # interaction of the sum-only reader also has sign(xy) in the small-signal regime
    for x, y in rng.uniform(-0.3, 0.3, (200, 2)):
        assert np.sign(I(s, x, y)) == np.sign(x * y)


def test_A4_matched_post_LN_sum_separates_designed_from_sum_only():
    from scipy.optimize import brentq
    vals = {}
    for name, fam in (("designed", agreement_family), ("sum", sum_only_family)):
        m, basis, emb, _ = fam(act="gelu")
        e1, e2, e5 = basis[0], basis[1], basis[4]
        tr0 = forward(m, emb(1.0, 1.0), 2)
        def run(x, y):
            mid = tr0.mid[0].copy(); mid[2] = x * e1 + y * e2 + 3.0 * e5
            t = forward(m, emb(1.0, 1.0), 2, replace_site(0, "mid", mid))
            return margin(t), float(layernorm(t.mid[0][2], m.layers[0].g2, m.layers[0].b2, m.eps) @ (e1 + e2))
        tgt = run(1, 1)[1]
        vals[name] = [run(t * dx, t * dy)[0] for dx, dy in ((1, 1), (2, 0), (3, -1))
                      for t in [brentq(lambda t: run(t * dx, t * dy)[1] - tgt, 1e-3, 10)]]
    assert np.ptp(vals["sum"]) < 1e-9
    assert vals["designed"][0] > 0 > vals["designed"][2] and abs(vals["designed"][1]) < 1e-12


def test_A4_raw_sum_matching_is_not_enough_because_LN2_reads_the_norm():
    m, basis, emb, _ = sum_only_family(act="gelu")
    e1, e2, e5 = basis[0], basis[1], basis[4]
    tr0 = forward(m, emb(1.0, 1.0), 2)
    out = []
    for x, y in ((1, 1), (2, 0), (3, -1)):
        mid = tr0.mid[0].copy(); mid[2] = x * e1 + y * e2 + 3.0 * e5
        out.append(margin(forward(m, emb(1.0, 1.0), 2, replace_site(0, "mid", mid))))
    assert np.ptp(out) > 0.1           # the preregistered PA9 failure, kept as a test

# ---------------------------------------------------- Theorem B1 (QK gate)


def test_B1_score_is_bilinear_and_gate_is_monotone():
    model, D, embed = qk_gate()
    rows = []
    for a in np.linspace(-1.5, 1.5, 7):
        for b in np.linspace(-1.5, 1.5, 7):
            X, r = embed(a, b); tr = forward(model, X, r)
            zq = layernorm(tr.mid[0][r], model.layers[1].g1, model.layers[1].b1, model.eps) @ D["fQ"]
            zk = layernorm(tr.mid[0][1], model.layers[1].g1, model.layers[1].b1, model.eps) @ D["fB"]
            s = tr.scores[1][0][r]
            assert np.isclose(s[1], 4.0 * zq * zk / np.sqrt(2)) and abs(s[0]) < 1e-12 and abs(s[2]) < 1e-12
            rows.append((zq * zk, tr.pattern[1][0][r][1]))
    rows.sort()
    assert all(rows[i][1] <= rows[i + 1][1] + 1e-15 for i in range(len(rows) - 1))


def test_B1_exact_margin_formula_without_downstream_readers():
    model, D, embed = qk_gate()
    fU = D["fU"]
    for a in (-1.0, 0.4, 1.3):
        for b in (-0.8, 0.5, 1.1):
            X, r = embed(a, b); tr = forward(model, X, r)
            h1 = 2 * tr.head_out[0][0][r] @ fU
            zB = layernorm(tr.mid[0][1], model.layers[1].g1, model.layers[1].b1, model.eps) @ D["fB"]
            wB = tr.pattern[1][0][r][1]
            assert np.isclose(margin(tr), h1 + 2 * 1.5 * zB * wB, atol=1e-12)


def test_B_dissection_and_negative_controls():
    model, D, embed = qk_gate()
    mo, _, _ = qk_gate(qk_off=True)
    LV = [-1.0, -0.5, 0.5, 1.0]
    Tg = np.array([[margin(forward(model, *embed(a, b))) for b in LV] for a in LV])
    To = np.array([[margin(forward(mo, *embed(a, b))) for b in LV] for a in LV])
    assert not is_relation_blind(Tg) and is_relation_blind(To)
    for a in LV:                                                    # masking
        assert np.isclose(margin(forward(model, *embed(a, 1.0, layout="ARB"))), margin(forward(model, *embed(a, -1.0, layout="ARB"))))


def test_B_downstream_rereading_defeats_the_head_level_claim():
    mm, D, embed = qk_gate(reread_mlp=6.0)
    X0, r = embed(-1.0, 0.5); X1, _ = embed(0.5, 0.5)
    t0, t1 = forward(mm, X0, r), forward(mm, X1, r)
    dh = 2 * (t1.head_out[1][0][r] - t0.head_out[1][0][r]) @ D["fU"]
    assert dh > 0 > margin(t1) - margin(t0)

# ---------------------------------------------------- Theorem C1 (two-block reconciliation)


def test_C1_gate_function_properties():
    beta, gamma, theta = 1.0, 1.5, 0.6
    g = lambda S, q: np.maximum(beta * S + gamma * q - theta, 0) - np.maximum(-beta * S + gamma * q - theta, 0)
    rng = np.random.default_rng(3)
    for S, q in rng.uniform(-2, 2, (2000, 2)):
        assert np.isclose(g(-S, q), -g(S, q))
        assert np.sign(g(S, q)) in (0, np.sign(S))
        assert abs(g(S, q + 0.1)) >= abs(g(S, q)) - 1e-12


def test_C_decision_follows_net_evidence_and_bottleneck_screens():
    model, D, embed = reconciliation()
    e1, e2 = D["e1"], D["e2"]
    for a in np.linspace(-1.5, 1.5, 9):
        for b in np.linspace(-1.5, 1.5, 9):
            tr = forward(model, embed(a, b), 2)
            S = layernorm(tr.resid_in[1][2], model.layers[1].g2, model.layers[1].b2, model.eps) @ (e1 + e2)
            if abs(S) > 1e-9:
                assert np.sign(margin(tr)) == np.sign(S)
    mb, _, eb = reconciliation(bottleneck=True)
    # exact matched-sum pair in input space (source LN inverted)
    x = lambda a: a / np.sqrt((a * a + 4.0) / 8 + 1e-5)
    ainv = lambda v: np.sign(v) * np.sqrt(v * v * (4.0 / 8 + 1e-5) / (1 - v * v / 8))
    ag, dg = (ainv(0.6), ainv(0.6)), (ainv(1.5), ainv(-0.3))
    assert np.isclose(x(ag[0]) + x(ag[1]), x(dg[0]) + x(dg[1]))
    assert np.isclose(margin(forward(mb, eb(*ag), 2)), margin(forward(mb, eb(*dg), 2)), atol=1e-12)
    m_ag, m_dg = margin(forward(model, embed(*ag), 2)), margin(forward(model, embed(*dg), 2))
    assert m_ag > m_dg > 0
    prior = 0.5 * (m_ag + m_dg)
    mp, _, ep = reconciliation(prior=prior)
    assert margin(forward(mp, ep(*ag), 2)) > 0 > margin(forward(mp, ep(*dg), 2))

# ---------------------------------------------------- identifiability


def test_head_redistribution_changes_per_head_quantities_not_the_function():
    model, D, embed = qk_gate(n_heads2=2)
    E = np.random.default_rng(3).normal(0, 1.0, model.layers[1].WO[0].shape)
    mg = gauge_head_redistribution(model, 1, 0, 1, E)
    X, r = embed(1.0, 1.0)
    t, tg = forward(model, X, r), forward(mg, X, r)
    assert np.allclose(t.logits, tg.logits, atol=1e-12)
    assert np.allclose(t.head_out[1].sum(0), tg.head_out[1].sum(0), atol=1e-12)
    assert not np.allclose(t.head_out[1][0] @ D["fU"], tg.head_out[1][0] @ D["fU"])


def test_other_gauges():
    m1, D1, emb1 = qk_gate()
    rng = np.random.default_rng(4)
    X, r = emb1(0.7, -0.4)
    for mm in (gauge_qk(m1, 1, 0, rng.normal(size=(2, 2)) + 2 * np.eye(2)), gauge_vo(m1, 1, 0, rng.normal(size=(2, 2)) + 2 * np.eye(2))):
        assert np.allclose(forward(m1, X, r).logits, forward(mm, X, r).logits, atol=1e-12)
    mc, _, ec = reconciliation()
    mr = gauge_relu_rescale(mc, 1, np.array([0.3, 4.0, 2.0, 0.5]))
    assert np.allclose(forward(mc, ec(0.4, -0.9), 2).logits, forward(mr, ec(0.4, -0.9), 2).logits, atol=1e-12)
    mp, (e1, e2, e3), ep = positive_control()
    R = random_orthogonal_fixing_one(4, 11)
    mR, mapX = gauge_residual_rotation(mp, R)
    ms = gauge_split_neuron(mp, 0, 0)
    for a, b in BIN:
        t, tR = forward(mp, ep(a, b), 2), forward(mR, mapX(ep(a, b)), 2)
        assert np.allclose(t.logits, tR.logits, atol=1e-12) and np.allclose(t.mlp_pre[0], tR.mlp_pre[0], atol=1e-12)
        assert np.allclose(t.logits, forward(ms, ep(a, b), 2).logits, atol=1e-12)

# ---------------------------------------------------- C' rereading through attention (post-review)


def test_Cprime_agreement_routes_to_sources_conflict_defers_to_prior():
    from cbmath import rereading
    m, D, emb = rereading()
    for a in (-1.5, -0.5, 0.5, 1.5):
        for b in (-1.5, -0.5, 0.5, 1.5):
            mm = margin(forward(m, emb(a, b), 3))
            assert (np.sign(mm) == np.sign(a)) if a * b > 0 else mm < 0


def test_Cprime_one_layer_cannot_route_on_the_relation():
    """In one attention layer, log(alpha_A/alpha_P) at the read row depends only on the read,
    A and P tokens (pre-norm keys and queries read only their own token), not on b."""
    import copy
    from cbmath import rereading
    m, D, emb = rereading()
    rng = np.random.default_rng(0)
    for _ in range(10):
        one = copy.deepcopy(m); one.layers = [copy.deepcopy(m.layers[1])]
        one.layers[0].WQ = rng.normal(size=one.layers[0].WQ.shape)
        one.layers[0].WK = rng.normal(size=one.layers[0].WK.shape)
        r = [np.log(forward(one, emb(0.8, b), 3).pattern[0][0][3][1] / forward(one, emb(0.8, b), 3).pattern[0][0][3][0]) for b in (-1.5, 0.2, 1.5)]
        assert np.ptp(r) < 1e-10


def test_Cprime_mediator_interchange_and_ablation():
    from cbmath import rereading
    m, D, emb = rereading()
    ta, td = forward(m, emb(1.0, 1.0), 3), forward(m, emb(1.0, -1.0), 3)
    p = td.mlp_out[0].copy(); p[3] = ta.mlp_out[0][3]
    assert margin(td) < 0 < margin(forward(m, emb(1.0, -1.0), 3, replace_site(0, "mlp_out", p)))
    mab, _, e2 = rereading(block1_mlp=False)
    assert np.allclose(forward(mab, e2(1.0, 1.0), 3).pattern[1][0][3], forward(mab, e2(1.0, -1.0), 3).pattern[1][0][3])


def test_C_exact_match_at_block2_reader_gap_vanishes_when_ablated():
    from scipy.optimize import brentq
    gaps = {}
    for kw in ({}, {"block1_mlp": False}):
        mc, Dc, ec = reconciliation(**kw)
        s = Dc["e1"] + Dc["e2"]
        def S(a, b):
            tr = forward(mc, ec(a, b), 2)
            return float(layernorm(tr.resid_in[1][2], mc.layers[1].g2, mc.layers[1].b2, mc.eps) @ s), margin(tr)
        tgt, mag = S(0.5, 0.5)
        bd = brentq(lambda b: S(1.6, b)[0] - tgt, -1.6, 0.0)
        gaps[bool(kw)] = mag - S(1.6, bd)[1]
    assert gaps[False] > 4.0 and abs(gaps[True]) < 1e-9
