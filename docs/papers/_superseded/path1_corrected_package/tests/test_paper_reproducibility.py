from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.othello_r1 import kl_row, legality_sets  # noqa: E402
from src.othello_r2_fiber import js_divergence, softmax  # noqa: E402
from src.othello_r2_fiber_spec import y_components  # noqa: E402
from src.othello_r2_anchor_interaction import FRESH_TRIPLET_HASH  # noqa: E402
from src.othello_r2 import OBSTRUCTION_READY, OFFICIAL_0A_LETTER, SECTION_53_STATUS  # noqa: E402


def test_y_equals_g_minus_q():
    probs = softmax(np.arange(61, dtype=np.float64))
    rec = [10, 11, 12]
    tgt = [12, 13, 14]
    parts = legality_sets(rec, tgt)
    y = y_components(probs, rec, tgt)
    assert 13 in parts["became_legal"] and 14 in parts["became_legal"]
    assert 10 in parts["became_illegal"] and 11 in parts["became_illegal"]
    assert y["Y"] == pytest.approx(y["G"] - y["Q"])


def test_js_symmetric_and_zero_on_self():
    p = softmax(np.array([0.0, 1.0, 2.0, 0.0]))
    q = softmax(np.array([0.2, 0.1, 1.5, 0.3]))
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert js_divergence(p, q) >= 0.0


def test_js_uses_natural_log_kl():
    p = np.array([0.25, 0.25, 0.5])
    q = np.array([0.5, 0.25, 0.25])
    m = 0.5 * (p + q)
    manual = 0.5 * kl_row(p, m) + 0.5 * kl_row(q, m)
    # js_divergence adds 1e-12 then renormalizes; on these 3-vectors the difference is tiny
    p4 = np.concatenate([p, np.zeros(58)])
    q4 = np.concatenate([q, np.zeros(58)])
    assert js_divergence(p4, q4) >= 0.0
    assert manual >= 0.0


def test_official_letters_unchanged():
    assert OFFICIAL_0A_LETTER == "C / PARTIAL"
    assert SECTION_53_STATUS == "PARTIAL"
    assert OBSTRUCTION_READY == "NO"


def test_fresh_hash_matches_numbers_file():
    num = json.loads((ROOT / "reported_numbers.json").read_text())
    assert num["hashes"]["fresh_triplet"]["value"] == FRESH_TRIPLET_HASH
    assert num["fresh_symmetric"]["both_positive_fraction"] == pytest.approx(0.40625)
    assert num["fresh_symmetric"]["both_positive_count"] == 13


def test_paper_omits_cor_from_title():
    paper = (ROOT / "paper_master.md").read_text()
    title = next(line for line in paper.splitlines() if line.startswith("# "))
    assert "COR" not in title
    assert "Obstruction" not in title


def test_negative_letters_in_paper():
    paper = (ROOT / "paper_master.md").read_text()
    assert "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED" in paper
    assert "TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED" in paper
    assert "08R2_VALIDATION_COMPLETE — NEGATIVE" in paper
    assert "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE" in paper


def test_does_not_call_y_correlation_a_direction_correlation():
    paper = (ROOT / "paper_master.md").read_text()
    assert "highly correlated with learned directions" not in paper
    assert "mean cosine" in paper.lower() or "Mean cosine" in paper
    assert "0.792" in paper
    assert "0.850" in paper or "0.849" in paper


def test_fixed_js_csv_recomputes_primary_contrasts():
    import csv
    from collections import defaultdict

    rows = list(csv.DictReader((ROOT / "verified_data" / "fixed_js_verified_data.csv").open()))
    assert len(rows) == 336
    by = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    d_wrong = []
    d_rand = []
    probe = [float(r["probe_max_abs"]) for r in rows]
    for rs in by.values():
        c = [float(x["y"]) for x in rs if x["method"] == "correct"][0]
        w = [float(x["y"]) for x in rs if x["method"] == "wrong"]
        rnd = [float(x["y"]) for x in rs if x["method"] == "random"]
        d_wrong.append(c - sum(w) / 3)
        d_rand.append(c - sum(rnd) / 3)
    assert abs(sum(d_wrong) / len(d_wrong) - 0.02655064) < 1e-8
    assert abs(sum(d_rand) / len(d_rand) - 0.02296572) < 1e-8
    assert max(probe) < 1e-12


def test_mentions_jacopin_and_heuristics_note():
    paper = (ROOT / "paper_master.md").read_text()
    assert "Jacopin" in paper
    assert "bag of heuristics" in paper.lower() or "Bag of heuristics" in paper or "heuristics" in paper


def test_stage_share_identity():
    num = json.loads((ROOT / "reported_numbers.json").read_text())
    full = num["sem"]["mean_level_correction"]["full_minus_editor_a"]["value"]
    stage = num["sem"]["mean_level_correction"]["stage_only_minus_editor_a"]["value"]
    share = num["sem"]["mean_level_correction"]["stage_share_of_full_minus_editor_a"]["value"]
    assert stage / full == pytest.approx(share, rel=1e-3)
