"""08R1 measurement utilities. Does not change official 0A letters or historical CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.toy_othello import (
    CENTER_SQUARES,
    PASS_SQUARE,
    bootstrap_by_game,
    square_to_token,
)


PLAYABLE = [sq for sq in range(64) if sq not in CENTER_SQUARES]


def token_of(square: int) -> int | None:
    if square in (PASS_SQUARE, -100) or square in CENTER_SQUARES:
        return None
    return int(square_to_token(square))


def tokens_of_squares(squares: Iterable[int]) -> list[int]:
    out = []
    for sq in squares:
        tok = token_of(int(sq))
        if tok is not None:
            out.append(tok)
    return out


def total_legal_mass(probs: np.ndarray, squares: Iterable[int]) -> float:
    """sum(P(move) for move in legal_set)."""
    toks = tokens_of_squares(squares)
    if not toks:
        return float("nan")
    return float(np.asarray(probs, dtype=np.float64)[toks].sum())


def mean_prob_per_move(probs: np.ndarray, squares: Iterable[int]) -> float:
    """mean(P(move) for move in legal_set). Old 0A helper reported only this."""
    toks = tokens_of_squares(squares)
    if not toks:
        return float("nan")
    return float(np.asarray(probs, dtype=np.float64)[toks].mean())


def _delta_on(before: np.ndarray, after: np.ndarray, squares: Iterable[int], how: str) -> float:
    if how == "sum":
        a = total_legal_mass(after, squares)
        b = total_legal_mass(before, squares)
    elif how == "mean":
        a = mean_prob_per_move(after, squares)
        b = mean_prob_per_move(before, squares)
    else:
        raise ValueError(how)
    if a != a or b != b:
        return float("nan")
    return float(a - b)


def legality_sets(rec_legal: Iterable[int], target_legal: Iterable[int]) -> dict[str, list[int]]:
    rec = {int(s) for s in rec_legal if int(s) not in CENTER_SQUARES and int(s) != PASS_SQUARE}
    tgt = {int(s) for s in target_legal if int(s) not in CENTER_SQUARES and int(s) != PASS_SQUARE}
    playable = set(PLAYABLE)
    return {
        "became_legal": sorted(tgt - rec),
        "became_illegal": sorted(rec - tgt),
        "stayed_legal": sorted(rec & tgt),
        "stayed_illegal": sorted(playable - rec - tgt),
    }


def intervention_probability_metrics(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    rec_legal: Iterable[int],
    target_legal: Iterable[int],
) -> dict:
    """Corrected legal-set probability effects. Recipient vs target boards, same side-to-move assumed in the sets."""
    parts = legality_sets(rec_legal, target_legal)
    row: dict = {}
    for name, squares in parts.items():
        row[f"n_{name}"] = len(squares)
        row[f"{name}_d_mass"] = _delta_on(probs_before, probs_after, squares, "sum")
        row[f"{name}_d_mean"] = _delta_on(probs_before, probs_after, squares, "mean")
        row[f"{name}_mass_before"] = total_legal_mass(probs_before, squares)
        row[f"{name}_mass_after"] = total_legal_mass(probs_after, squares)
    row["donor_legal_mass_before"] = total_legal_mass(probs_before, target_legal)
    row["donor_legal_mass_after"] = total_legal_mass(probs_after, target_legal)
    row["recip_legal_mass_before"] = total_legal_mass(probs_before, rec_legal)
    row["recip_legal_mass_after"] = total_legal_mass(probs_after, rec_legal)
    row["donor_legal_mass_lift"] = row["donor_legal_mass_after"] - row["donor_legal_mass_before"]
    row["old_mean_donor_legal_before"] = mean_prob_per_move(probs_before, target_legal)
    row["old_mean_donor_legal_after"] = mean_prob_per_move(probs_after, target_legal)
    row["old_mean_donor_legal_lift"] = row["old_mean_donor_legal_after"] - row["old_mean_donor_legal_before"]
    return row


def kl_row(before: np.ndarray, after: np.ndarray) -> float:
    p = np.asarray(before, dtype=np.float64) + 1e-12
    q = np.asarray(after, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (np.log(p) - np.log(q))).sum())


def cluster_bootstrap(
    values: np.ndarray,
    cluster_ids: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    point = float(np.nanmean(arr))
    finite = [(v, c) for v, c in zip(arr, cluster_ids) if v == v]
    if not finite:
        return {
            "n": 0,
            "n_clusters": 0,
            "point": float("nan"),
            "boot_mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n_boot": int(n_boot),
            "seed": int(seed),
        }
    vals = np.asarray([v for v, _ in finite], dtype=np.float64)
    cids = [c for _, c in finite]
    boot_mean, lo, hi = bootstrap_by_game(vals, cids, n_boot, seed)
    return {
        "n": int(len(vals)),
        "n_clusters": int(len(set(cids))),
        "point": point,
        "boot_mean": float(boot_mean),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def paired_cluster_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    cluster_ids: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    diff = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    out = cluster_bootstrap(diff, cluster_ids, n_boot, seed)
    out["comparison"] = "a_minus_b"
    return out


def donor_dependence(recip_games: list[str], donor_games: list[str]) -> dict:
    n = len(recip_games)
    n_recip = len(set(recip_games))
    n_donor = len(set(donor_games))
    pair_games = [f"{r}|{d}" for r, d in zip(recip_games, donor_games)]
    reused_donors = n_donor < n
    return {
        "n_pairs": n,
        "n_distinct_recipient_games": n_recip,
        "n_distinct_donor_games": n_donor,
        "n_distinct_recip_donor_game_pairs": len(set(pair_games)),
        "donor_reuse": bool(reused_donors),
        "note": (
            "Primary CI clusters by recipient game. "
            "Secondary CI clusters by recipient|donor game pair. "
            "Do not treat pairs as independent if either count < n_pairs."
        ),
    }


@dataclass
class FakeResidualForward:
    """SYNTHETIC_UNIT_TEST residual. Not Othello-GPT. Not a scientific substitute."""

    d_model: int = 8
    d_vocab: int = 61

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        self.resid_mid = rng.normal(size=(2, self.d_model)).astype(np.float64)
        self.mlp_out = rng.normal(scale=0.8, size=(2, self.d_model)).astype(np.float64)
        self.resid_post = self.resid_mid + self.mlp_out
        self.unembed = rng.normal(size=(self.d_model, self.d_vocab)).astype(np.float64)

    def cache(self, hook: str) -> np.ndarray:
        if hook.endswith("resid_mid"):
            return self.resid_mid.copy()
        if hook.endswith("resid_post"):
            return self.resid_post.copy()
        raise KeyError(hook)

    def logits_from_post(self, resid_post: np.ndarray) -> np.ndarray:
        # Final LN analogue: unit-normalize then unembed.
        x = resid_post - resid_post.mean(axis=-1, keepdims=True)
        nrm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return (x / nrm) @ self.unembed

    def run_patched(self, hook: str, values: np.ndarray) -> dict:
        mid = self.resid_mid.copy()
        post = self.resid_post.copy()
        patched = values.astype(np.float64)
        if hook.endswith("resid_mid"):
            mid = patched
            post = mid + self.mlp_out
        elif hook.endswith("resid_post"):
            post = patched
        else:
            raise KeyError(hook)
        base = self.logits_from_post(self.resid_post)
        after = self.logits_from_post(post)
        return {
            "hook": hook,
            "recip_mid": self.resid_mid.copy(),
            "recip_post": self.resid_post.copy(),
            "patched_mid": mid,
            "patched_post": post,
            "pert_norm_mid": float(np.linalg.norm(mid - self.resid_mid)),
            "pert_norm_post": float(np.linalg.norm(post - self.resid_post)),
            "logit_max_abs_diff": float(np.max(np.abs(after - base))),
            "logits_before": base,
            "logits_after": after,
        }


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cpu_generator_fill(shapes: list[tuple[int, ...]], seed: int) -> list[np.ndarray]:
    """CPU-only init used by the untrained-model control. Never device-bound."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape).astype(np.float32) for shape in shapes]


def apply_cpu_normal_to_tensors(tensors, seed: int) -> None:
    """Fill torch parameter tensors from a CPU generator, then copy to their device."""
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    with torch.no_grad():
        for p in tensors:
            drawn = torch.randn(p.shape, generator=g, dtype=torch.float32)
            p.copy_(drawn.to(device=p.device, dtype=p.dtype))
