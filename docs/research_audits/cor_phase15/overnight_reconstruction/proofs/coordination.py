"""Small exact examples that keep four coordination notions apart."""

from __future__ import annotations

import numpy as np


def deletion_necessary(margin_fn, baseline_parts, index, threshold=0.0):
    """A deletion is necessary when the margin is positive and the deleted
    evaluation is not. `baseline_parts` is a tuple of addend values.
    """
    full = margin_fn(*baseline_parts)
    deleted = list(baseline_parts)
    deleted[index] = 0.0
    reduced = margin_fn(*deleted)
    return full > threshold and reduced <= threshold


def four_term_interaction(fn, u, v):
    return fn(u, v) - fn(u, 0.0) - fn(0.0, v) + fn(0.0, 0.0)


def additive_margin(b, c1, c2):
    return b + c1 + c2
