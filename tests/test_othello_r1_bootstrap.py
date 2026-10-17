from __future__ import annotations

import numpy as np

from src.othello_r1 import cluster_bootstrap, donor_dependence, paired_cluster_diff
from src.toy_othello import bootstrap_by_game


def test_bootstrap_clusters_repeated_recipient_rows():
    values = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
    games = ["g0", "g0", "g0", "g1", "g1"]
    mean, lo, hi = bootstrap_by_game(values, games, n_boot=400, seed=7)
    # Game-mean of g0 is 1, g1 is 10; overall game-mean is 5.5, not 4.6.
    assert abs(mean - 5.5) < 0.6
    assert lo < hi
    # Independent-row mean would be 4.6; clustered CI should not collapse onto that.
    row_mean = float(values.mean())
    assert abs(mean - row_mean) > 0.2


def test_cluster_bootstrap_records_seed_and_count():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    games = ["a", "a", "b", "b"]
    out = cluster_bootstrap(values, games, n_boot=200, seed=11)
    assert out["n_boot"] == 200
    assert out["seed"] == 11
    assert out["n_clusters"] == 2
    assert out["n"] == 4
    assert out["ci95_low"] <= out["ci95_high"]


def test_paired_diff_and_donor_dependence():
    a = np.array([0.2, 0.2, 0.4])
    b = np.array([0.1, 0.1, 0.1])
    rec = ["r0", "r0", "r1"]
    don = ["d0", "d0", "d0"]
    dep = donor_dependence(rec, don)
    assert dep["n_pairs"] == 3
    assert dep["n_distinct_recipient_games"] == 2
    assert dep["n_distinct_donor_games"] == 1
    assert dep["donor_reuse"] is True
    diff = paired_cluster_diff(a, b, rec, n_boot=100, seed=3)
    assert diff["point"] > 0
    assert diff["n_clusters"] == 2
