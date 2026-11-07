"""Checks of the theorems in THEOREM_REGISTER.md.

Proofs are in proofs/PROOFS.md.  These tests check the implementation and
the explicit constructions; a passing test is not a proof of a universal claim.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proofs"))
from bridge_math import (  # noqa: E402
    Model, centered_functional, consensus_update, decision_null_basis, dilution_model,
    factorial, forward, has_crossover, head_direct_quantities, is_relation_blind, kappa,
    dedicated_heads_model, margin, mixed_step, prompt, random_model, rb_forces, rb_margin, sigma,
)

# ---------------------------------------------------------------- T-SS ------


def test_SS_sum_preserving_head_edits_leave_every_logit_unchanged():
    """Theorem SS: same-cut writers act on everything downstream only through
    their per-position sum.  Two layers, three heads, edits at layer 0."""
    m = random_model(7, n_layers=2, H=3, V=5)
    X = np.random.default_rng(1).normal(0, 1, (5, 8))
    base = forward(m, X, r=4)
    rng = np.random.default_rng(2)
    for _ in range(20):
        D = rng.normal(0, 3.0, (3, 5, 8))
        D -= D.mean(axis=0, keepdims=True)  # zero sum over heads at each position

        def hook(li, site, arr, D=D):
            return arr + D if (li == 0 and site == "head_out") else arr

        tr = forward(m, X, r=4, hook=hook)
        assert np.allclose(tr.logits, base.logits, atol=1e-10)
        # the pairwise relation between the edited writers moved a lot
        cos0 = _cos(base.head_out[0][0][4], base.head_out[0][1][4])
        cos1 = _cos(tr.head_out[0][0][4], tr.head_out[0][1][4])
        assert np.isfinite(cos0) and np.isfinite(cos1)


def test_SS_same_relation_different_sum_changes_decision():
    """F25 at the component level: identical pairwise relation, different sum."""
    m = random_model(7, n_layers=1, H=2, V=5)
    X = np.random.default_rng(1).normal(0, 1, (5, 8))
    base = forward(m, X, r=4)
    c = base.head_out[0]

    def scale(k):
        def hook(li, site, arr):
            return arr * k if site == "head_out" else arr
        return hook

    t1 = forward(m, X, 4, scale(1.0)); t2 = forward(m, X, 4, scale(3.0))
    assert np.isclose(_cos(t1.head_out[0][0][4], t1.head_out[0][1][4]),
                      _cos(t2.head_out[0][0][4], t2.head_out[0][1][4]))
    assert not np.isclose(margin(t1.logits, 0, 1), margin(t2.logits, 0, 1))
    assert c is not None


def _cos(x, y):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

# ---------------------------------------------------------------- T-NC ------


def _nc_setup(seed=3, V=6):
    m = random_model(seed, n_layers=1, H=2, V=V)
    X = np.random.default_rng(seed + 10).normal(0, 1, (4, 8))
    return m, X


def test_NC_exact_decomposition_and_sign_law():
    """m(h+v) = a(h)/sigma(h+v) + kappa for every v in N_C; sign(R)=sign(a)sign(sigma-sigma')."""
    m, X = _nc_setup()
    C = [0, 1, 2]
    B = decision_null_basis(m, C)
    assert B.shape[1] == m.d - len(C)
    base = forward(m, X, r=3)
    h = base.final[3]
    rng = np.random.default_rng(4)
    for _ in range(50):
        v = B @ rng.normal(0, rng.uniform(0.1, 6.0), B.shape[1])

        def hook(li, site, arr, v=v):
            if site == "final":
                arr = arr.copy(); arr[3] = arr[3] + v
            return arr

        tr = forward(m, X, r=3, hook=hook)
        s0, s1 = sigma(m, h), sigma(m, h + v)
        hc = h - h.mean()
        assert np.isclose(s1 ** 2 - s0 ** 2, (2 * hc @ v + v @ v) / m.d, atol=1e-10)
        for c in C:
            for cp in C:
                if c == cp:
                    continue
                ut = centered_functional(m, c, cp)
                a = ut @ h
                pred = a / s1 + kappa(m, c, cp)
                assert np.isclose(margin(tr.logits, c, cp), pred, atol=1e-10)
                R = margin(tr.logits, c, cp) - margin(base.logits, c, cp)
                if abs(R) > 1e-9:
                    assert np.sign(R) == np.sign(a) * np.sign(s0 - s1)


def test_NC_flip_iff_content_and_bias_disagree_and_threshold_reachable():
    """A pair flips under some decision-null edit iff sign(a) != sign(kappa) and the flip
    norm sigma* = -a/kappa is reachable: sigma* > sigma_min = sqrt(|h~_perp|^2/d + eps)."""
    seen = {"flip": 0, "same_sign": 0, "differ_unreachable": 0}
    for seed in (5, 6, 8, 12):  # seed 12 contains a disagreeing but unreachable pair
        m, X = _nc_setup(seed=seed)
        C = [0, 1, 2]
        B = decision_null_basis(m, C)
        h = forward(m, X, r=3).final[3]
        hc = h - h.mean()
        hN = B @ (B.T @ hc)
        smin = np.sqrt((hc - hN) @ (hc - hN) / m.d + m.eps)
        n = hN / np.linalg.norm(hN)
        # this family sweeps sigma continuously over the whole reachable set [smin, inf)
        family = [-hN + s * n for s in np.linspace(0, np.linalg.norm(hN), 400)] + \
                 [t * n for t in np.geomspace(1e-3, 1e5, 800)]
        for c in C:
            for cp in C:
                if c >= cp:
                    continue
                a = centered_functional(m, c, cp) @ h
                k = kappa(m, c, cp)
                differ = np.sign(a) != np.sign(k)
                flippable = differ and (-a / k > smin)
                seen["flip" if flippable else ("differ_unreachable" if differ else "same_sign")] += 1
                signs = {np.sign(a / sigma(m, h + v) + k) for v in family}
                assert (len(signs) == 2) == flippable
    assert all(v > 0 for v in seen.values()), seen


def test_NC_flip_direction_is_toward_bias_ranking_when_norm_grows():
    m, X = _nc_setup(seed=5)
    C = [0, 1, 2]
    B = decision_null_basis(m, C)
    h = forward(m, X, r=3).final[3]
    v = B[:, 0] * 1e5

    def hook(li, site, arr):
        if site == "final":
            arr = arr.copy(); arr[3] = arr[3] + v
        return arr

    L = forward(m, X, 3, hook).logits
    for c in C:
        for cp in C:
            if c != cp:
                assert np.isclose(L[c] - L[cp], kappa(m, c, cp), atol=1e-3)
    assert h is not None


def test_NC_vacuous_for_full_vocabulary_when_V_exceeds_d():
    m = random_model(0, d=8, V=20)
    assert decision_null_basis(m, list(range(20))).shape[1] == 0

# ---------------------------------------------------------------- T-CI ------


def test_CI_single_replacement_identity():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(2, 9)
        w = rng.uniform(0.01, 5, n); v = rng.normal(0, 2, n); j = rng.integers(n)
        M, M2, ident = consensus_update(w, v, j, rng.uniform(0, 5), rng.normal(0, 2))
        assert np.isclose(M2 - M, ident, atol=1e-12)


def test_CI_general_identity_in_recomputed_two_layer_multihead_model():
    """Head output change at the read position, projected on u~, equals
    sum_j w'_j (v'_j - M) / Z' for an arbitrary residual edit (all w, v move)."""
    m = random_model(1, n_layers=2, H=3, V=4)
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (5, 8))
    ut = centered_functional(m, 0, 1)
    tr = forward(m, X, 4)
    for _ in range(10):
        X2 = X.copy(); X2[rng.integers(0, 4)] += rng.normal(0, 2, 8)
        tr2 = forward(m, X2, 4)
        for h in range(3):
            w, v = head_direct_quantities(m, tr, 1, h, 4, ut)
            w2, v2 = head_direct_quantities(m, tr2, 1, h, 4, ut)
            M = w @ v / w.sum()
            ident = w2 @ (v2 - M) / w2.sum()
            actual = (tr2.head_out[1][h][4] - tr.head_out[1][h][4]) @ ut
            assert np.isclose(actual, ident, atol=1e-10)


def test_CI_unanimity_and_convex_hull_bound():
    """Direct head margin lies in [min_j v_j, max_j v_j]; unanimous sources fix its sign."""
    m = random_model(4, n_layers=1, H=2, V=4)
    rng = np.random.default_rng(5)
    ut = centered_functional(m, 0, 1)
    for _ in range(30):
        X = rng.normal(0, 1, (6, 8))
        tr = forward(m, X, 5)
        for h in range(2):
            w, v = head_direct_quantities(m, tr, 0, h, 5, ut)
            de = tr.head_out[0][h][5] @ ut
            assert v.min() - 1e-12 <= de <= v.max() + 1e-12


def test_CI_exact_total_margin_decomposition():
    """Delta m = Delta a / sigma' + a (1/sigma' - 1/sigma), with Delta a split exactly into
    head direct terms (Theorem CI), MLP-at-read term and read-token skip term."""
    m = random_model(9, n_layers=1, H=2, V=4)
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (5, 8)); X2 = X.copy(); X2[2] += rng.normal(0, 1.5, 8)
    ut = centered_functional(m, 0, 1)
    t1, t2 = forward(m, X, 4), forward(m, X2, 4)
    h1, h2 = t1.final[4], t2.final[4]
    a1 = ut @ h1
    s1, s2 = sigma(m, h1), sigma(m, h2)
    dHeads = 0.0
    for h in range(2):
        w, v = head_direct_quantities(m, t1, 0, h, 4, ut)
        w2, v2 = head_direct_quantities(m, t2, 0, h, 4, ut)
        dHeads += w2 @ (v2 - w @ v / w.sum()) / w2.sum()
    dMLP = (t2.mlp_out[0][4] - t1.mlp_out[0][4]) @ ut
    dSkip = (X2[4] - X[4]) @ ut
    da = dHeads + dMLP + dSkip
    assert np.isclose(ut @ h2 - a1, da, atol=1e-10)
    pred = da / s2 + a1 * (1 / s2 - 1 / s1)
    assert np.isclose(margin(t2.logits, 0, 1) - margin(t1.logits, 0, 1), pred, atol=1e-10)

# ------------------------------------------------- construction D (dilution) ---


def test_D_dilution_crossover_in_full_prenorm_block():
    """Weak supporting evidence raises the margin when the other slot is neutral and
    lowers it when the other slot holds strong supporting evidence."""
    for mlp, gain in (("off", 0.0), ("random", 1.0)):
        model, E = dilution_model(mlp=mlp, mlp_gain=gain)
        T = factorial(model, E, ["N", "W"], ["N", "S"])
        assert T[1, 0] > T[0, 0]       # W helps when B neutral
        assert T[1, 1] < T[0, 1]       # W hurts when B strong (agreeing!) evidence
        assert has_crossover(T)
        assert not is_relation_blind(T)


def test_D_crossover_locus_is_the_consensus():
    """With attention only, sign of A's effect = sign(e_W - M_B) where M_B is the head's
    consensus without A, exactly as Theorem CI says."""
    model, E = dilution_model(mlp="off")
    ut = centered_functional(model, 0, 1)
    for b in ["N", "W", "S", "OPP"]:
        t0 = forward(model, prompt(E, "N", b), 3)
        t1 = forward(model, prompt(E, "W", b), 3)
        w, v = head_direct_quantities(model, t0, 0, 0, 3, ut)
        M = w @ v / w.sum()
        w2, v2 = head_direct_quantities(model, t1, 0, 0, 3, ut)
        de = (t1.head_out[0][0][3] - t0.head_out[0][0][3]) @ ut
        assert np.sign(de) == np.sign(w2[1] * (v2[1] - M) - w[1] * (v[1] - M))
        assert np.sign(margin(t1.logits, 0, 1) - margin(t0.logits, 0, 1)) == np.sign(de)


def test_D_downstream_mlp_reverses_the_direct_prediction():
    """Multi-path counterexample: the consensus law fixes the direct effect, not the total."""
    model, E = dilution_model(mlp="reverse", mlp_gain=0.5)
    ut = centered_functional(model, 0, 1)
    t0 = forward(model, prompt(E, "N", "S"), 3)
    t1 = forward(model, prompt(E, "W", "S"), 3)
    de = (t1.head_out[0][0][3] - t0.head_out[0][0][3]) @ ut
    tot = margin(t1.logits, 0, 1) - margin(t0.logits, 0, 1)
    assert de < 0 < tot

# ------------------------------------------------- construction LN (norm coupling) ---


def test_LN_final_norm_coupling_creates_crossover_without_attention():
    d = 8
    e = np.zeros(d); e[0], e[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    n = np.zeros(d); n[2], n[3] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    WU = np.zeros((d, 2)); WU[:, 0] = e; WU[:, 1] = -e
    m = Model(layers=[], gf=np.ones(d), bf=np.zeros(d), WU=WU, bU=np.zeros(2))
    h0 = np.zeros(d); h0[4], h0[5] = 0.3, -0.3
    A = {"N": np.zeros(d), "W": 0.5 * e + 3.0 * n}
    Bv = {"N": np.zeros(d), "S": 3.0 * e}
    T = np.array([[margin(forward(m, (h0 + A[a] + Bv[b])[None], 0).logits, 0, 1) for b in Bv] for a in A])
    assert T[1, 0] > T[0, 0] and T[1, 1] < T[0, 1]
    assert not is_relation_blind(T)

# ---------------------------------------------------------------- T-RB ------


def test_RB_additive_tables_and_monotone_transforms():
    rng = np.random.default_rng(0)
    for _ in range(30):
        f = rng.normal(0, 1, 3); g = rng.normal(0, 1, 4)
        T = f[:, None] + g[None, :]
        assert is_relation_blind(T)
        for psi in (np.tanh, np.exp, lambda x: x ** 3, lambda x: 1 / (1 + np.exp(-x))):
            assert is_relation_blind(psi(T))


def test_RB_verdict_invariant_under_monotone_transform_of_nonblind_tables():
    model, E = dilution_model(mlp="off")
    T = factorial(model, E, ["N", "W", "S"], ["N", "W", "S", "OPP"])
    for psi in (np.tanh, np.exp, lambda x: x ** 3, lambda x: 1 / (1 + np.exp(-x))):
        assert is_relation_blind(psi(T)) == is_relation_blind(T)


def test_RB_crossover_implies_not_blind_random():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(400):
        T = rng.normal(0, 1, (3, 3))
        if has_crossover(T):
            hits += 1
            assert not is_relation_blind(T)
    assert hits > 50


def test_RB_two_by_two_without_crossover_is_always_blind():
    """Loftus (1978): in a 2x2 design only crossover interactions are nonremovable."""
    rng = np.random.default_rng(2)
    n = 0
    for _ in range(2000):
        T = rng.normal(0, 1, (2, 2))
        if not has_crossover(T):
            n += 1
            assert is_relation_blind(T)
    assert n > 300


def test_RB_double_cancellation_violation_without_crossover():
    T = np.array([[3.0, 6.0, 6.5], [5.0, 7.0, 8.0], [7.0, 7.5, 8.5]])
    assert not has_crossover(T)
    assert not is_relation_blind(T)
    # the violated instance: (a0,b1)>=(a1,b0), (a1,b2)>=(a2,b1) but (a0,b2)<(a2,b0)
    assert T[0, 1] >= T[1, 0] and T[1, 2] >= T[2, 1] and T[0, 2] < T[2, 0]


def test_RB_heldout_ordering_prediction_is_prospective_and_falsifiable():
    """Training comparisons (row and column monotonicity plus two cross comparisons)
    force, under relation-blindness, the order of a held-out pair that was never
    compared.  The double-cancellation table above violates the forced order."""
    mono = []
    for i in range(3):
        for j in range(2):
            mono.append(((i, j + 1), (i, j)))
            mono.append(((j + 1, i), (j, i)))
    train = mono + [((0, 1), (1, 0)), ((1, 2), (2, 1))]
    assert rb_forces(3, 3, train, (0, 2), (2, 0))          # RB predicts m(a0,b2) > m(a2,b0)
    assert not rb_forces(3, 3, mono, (0, 2), (2, 0))       # monotonicity alone does not
    T = np.array([[3.0, 6.0, 6.5], [5.0, 7.0, 8.0], [7.0, 7.5, 8.5]])
    assert all(T[p] > T[q] for p, q in train)              # training observations hold
    assert T[0, 2] < T[2, 0]                                # held-out prediction fails -> RB falsified

# ---------------------------------------------------------------- T-MIX -----


def test_MIX_output_term_nonnegative_consistency_term_unsigned():
    rng = np.random.default_rng(0)
    neg_cons_pos_total = 0
    for _ in range(300):
        Q = rng.normal(0, 1, (4, 4)); A = Q @ Q.T + 0.5 * np.eye(4)
        g = rng.normal(0, 1, 4); J = rng.normal(0, 1, 4)
        tot, cons, outp = mixed_step(A, g, J, beta=2.0)
        assert np.isclose(tot, cons + outp)
        assert outp >= -1e-12
        if cons < 0 < tot:
            neg_cons_pos_total += 1
    assert neg_cons_pos_total > 0


def test_D_nearby_negative_model_dedicated_heads_is_relation_blind():
    """Same tokens and evidence values, but each source reaches the read residual through
    its own head (a sum, Theorem SS) instead of sharing one softmax: no crossover,
    relation-blind margins, additive numerator up to the finite-score leakage."""
    m, P = dedicated_heads_model()
    levels = ["N", "W", "S", "OPP"]
    ut = centered_functional(m, 0, 1)
    Ta = np.array([[ut @ forward(m, P(a, b), 3).final[3] for b in levels] for a in levels])
    Tm = np.array([[margin(forward(m, P(a, b), 3).logits, 0, 1) for b in levels] for a in levels])
    assert np.abs(Ta - Ta[:, [0]] - Ta[[0], :] + Ta[0, 0]).max() < 1e-4
    assert not has_crossover(Tm)
    assert is_relation_blind(Tm)


def test_CI_star_relation_enters_only_through_attention_reallocation():
    """Corrected CI (review C FATAL-2, review D SERIOUS-5): for a substitution at one
    position, Delta M = [w_j (v'_j - v_j) + (w'_j - w_j)(v'_j - M)] / Z' (common score scale).
    The consensus deviation matters only through the weight change.  In Construction D the
    crossover is carried by the reallocation term; with the weight held fixed it vanishes."""
    m, E = dilution_model(mlp="off")
    ut = centered_functional(m, 0, 1)
    for b, expect_sign in (("N", 1), ("S", -1), ("OPP", 1)):
        t0 = forward(m, prompt(E, "N", b), 3); t1 = forward(m, prompt(E, "W", b), 3)
        w = np.exp(t0.scores[0][0][3][:4]); w2 = np.exp(t1.scores[0][0][3][:4])
        _, v = head_direct_quantities(m, t0, 0, 0, 3, ut); _, v2 = head_direct_quantities(m, t1, 0, 0, 3, ut)
        M = w @ v / w.sum(); Z2 = w2.sum()
        own = w[1] * (v2[1] - v[1]) / Z2
        realloc = (w2[1] - w[1]) * (v2[1] - M) / Z2
        de = (t1.head_out[0][0][3] - t0.head_out[0][0][3]) @ ut
        assert np.isclose(own + realloc, de, atol=1e-10)   # other positions unchanged at layer 0
        assert own > 0                                      # own-change term is relation-free
        assert np.sign(de) == expect_sign
        # holding the weight fixed (w'_j = w_j) removes the dependence on the other source
        assert np.sign(w[1] * (v2[1] - v[1])) == 1


def test_NC_rmsnorm_analogue_never_flips():
    """Review D SERIOUS-1: with RMSNorm (no centring, no bias) and no unembedding bias,
    m = a/rho with kappa = 0, so null edits rescale every margin and flip none."""
    rng = np.random.default_rng(0)
    d, V = 8, 3
    g = 1 + rng.normal(0, 0.1, d); WU = rng.normal(0, 1, (d, V)); h = rng.normal(0, 1, d)
    rows = np.array([g * (WU[:, c] - WU[:, 0]) for c in range(1, V)])
    _, s, Vt = np.linalg.svd(rows); B = Vt[(s > 1e-10).sum():].T
    def logits(x):
        return (g * x / np.sqrt((x @ x) / d + 1e-5)) @ WU
    base = np.argsort(logits(h))
    for _ in range(2000):
        v = B @ rng.normal(0, rng.uniform(0.1, 50), B.shape[1])
        assert (np.argsort(logits(h + v)) == base).all()


def test_RB_ties_count_as_independence_violations():
    T = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert has_crossover(T) and not is_relation_blind(T)


def test_NC_two_flips_on_a_non_monotone_path():
    """Review B, S5: 'at most one flip along any path' is false; the single crossing is in
    sigma.  An out-and-back path in N_C flips a flippable pair twice."""
    m, X = _nc_setup(seed=5)
    B = decision_null_basis(m, [0, 1, 2]); h = forward(m, X, r=3).final[3]
    a = centered_functional(m, 0, 1) @ h; k = kappa(m, 0, 1)
    n = B[:, 0]
    path = [t * n for t in np.concatenate([np.geomspace(1e-3, 1e4, 300), np.geomspace(1e4, 1e-3, 300)])]
    signs = [np.sign(a / sigma(m, h + v) + k) for v in path]
    flips = sum(1 for x, y in zip(signs, signs[1:]) if x != y)
    assert flips == 2
