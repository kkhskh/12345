"""08R2 state-constrained editing utilities.

No obstruction scores. No confirmation-split evaluation.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.othello_r1 import intervention_probability_metrics, kl_row
from src.othello_runutil import add_labels, board_arr
from src.toy_othello import board_hamming


SITE = "blocks.6.hook_resid_mid"
CANONICAL_CLASSES = np.array([0, 1, 2], dtype=np.int64)
CONFIRMATION_SPLIT = "08r2_08r3_confirmation"
OFFICIAL_0A_LETTER = "C / PARTIAL"
SECTION_53_STATUS = "PARTIAL"
OBSTRUCTION_READY = "NO"
R2_DIRNAME = "08r2"
STATUS_EXECUTED = "EXECUTED"
STATUS_FAILED = "FAILED"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_BLOCKED = "BLOCKED_MISSING_INPUT"
STATUS_EDITOR_C_FIT_FAILED = "EDITOR_C_FIT_FAILED"
REQUIRED_VALIDATION_FILES = (
    "08r2_probe_parity.csv",
    "08r2_optimization_traces.csv",
    "08r2_validation_pairs.json",
    "08r2_validation_state_effects.csv",
    "08r2_validation_causal_effects.csv",
    "08r2_validation_baselines.csv",
    "08r2_validation_bootstrap.csv",
    "08r2_validation_trial_ledger.csv",
    "08r2_validation_summary.json",
    "08r2_reproducibility_manifest.json",
    "08r2_confirmation_protocol.json",
    "08r2_scientific_report.md",
)


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_no_confirmation(rows: Iterable[dict], holdout_ids: set[str] | None = None) -> None:
    ids = {str(r.get("game_id", "")) for r in rows}
    bad = {i for i in ids if i.startswith("confirm_")}
    if holdout_ids:
        bad |= ids & set(holdout_ids)
    if bad:
        raise RuntimeError(f"confirmation-split games leaked into 08R2 validation: {sorted(bad)[:8]}")


class LinearBoardDecoder:
    """Exact sklearn linear-probe reconstruction. Canonical classes 0/1/2."""

    def __init__(self, pack: dict):
        models = pack["models"]
        if len(models) != 64:
            raise ValueError(f"expected 64 square classifiers, got {len(models)}")
        self.hook = str(pack.get("hook", SITE))
        self.target = str(pack.get("target", "rel"))
        self.method = str(pack.get("method", "linear"))
        self.d_model = int(np.asarray(models[0].coef_).shape[1])
        coef = np.zeros((64, 3, self.d_model), dtype=np.float64)
        intercept = np.full((64, 3), -1e6, dtype=np.float64)
        class_sets = []
        for j, clf in enumerate(models):
            if not hasattr(clf, "coef_"):
                raise ValueError(f"square {j} is not a linear classifier")
            classes = np.asarray(clf.classes_, dtype=np.int64)
            class_sets.append(tuple(int(c) for c in classes))
            w = np.asarray(clf.coef_, dtype=np.float64)
            b = np.asarray(clf.intercept_, dtype=np.float64)
            if w.ndim == 1:
                w = w.reshape(1, -1)
            if b.ndim == 0:
                b = np.asarray([b], dtype=np.float64)
            if len(classes) == 2 and w.shape[0] == 1:
                # sklearn binary: P(classes_[1]) = sigmoid(x·w + b). Embed as logits [0, s].
                w = np.vstack([np.zeros_like(w[0]), w[0]])
                b = np.array([0.0, float(b[0])], dtype=np.float64)
            if w.shape[0] != len(classes):
                raise ValueError(f"square {j} coef rows {w.shape[0]} != n_classes {len(classes)}")
            for row, cls in enumerate(classes):
                if int(cls) not in (0, 1, 2):
                    raise ValueError(f"unexpected class {cls} at square {j}")
                coef[j, int(cls)] = w[row]
                intercept[j, int(cls)] = b[row]
        self.coef = coef
        self.intercept = intercept
        self.class_sets = class_sets

    def logits_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        return np.einsum("bd,jcd->bjc", x, self.coef) + self.intercept[None, :, :]

    def proba_np(self, x: np.ndarray) -> np.ndarray:
        z = self.logits_np(x)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def predict_np(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits_np(x), axis=-1).astype(np.int64)

    def rowspace_basis(self, max_rank: int | None = None) -> np.ndarray:
        flat = self.coef.reshape(-1, self.d_model).T
        q, r = np.linalg.qr(flat, mode="reduced")
        diag = np.abs(np.diag(r)) if r.ndim == 2 else np.array([])
        keep = int((diag > 1e-8).sum()) if diag.size else q.shape[1]
        if max_rank is not None:
            keep = min(keep, int(max_rank))
        return q[:, :keep]

    def as_torch(self):
        import torch
        import torch.nn as nn

        mod = nn.Module()
        mod.coef = nn.Parameter(torch.tensor(self.coef, dtype=torch.float32), requires_grad=False)
        mod.intercept = nn.Parameter(torch.tensor(self.intercept, dtype=torch.float32), requires_grad=False)
        d = self.d_model

        def logits(x):
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return torch.einsum("bd,jcd->bjc", x, mod.coef) + mod.intercept

        def proba(x):
            z = logits(x)
            return torch.softmax(z, dim=-1)

        def predict(x):
            return torch.argmax(logits(x), dim=-1)

        def state_loss(h, target_lab, rec_lab, eta_preserve: float = 1.0):
            if h.ndim == 1:
                h = h.unsqueeze(0)
            logp = torch.log_softmax(logits(h), dim=-1)[0]
            tgt = torch.as_tensor(target_lab, device=h.device, dtype=torch.long)
            rec = torch.as_tensor(rec_lab, device=h.device, dtype=torch.long)
            change = tgt != rec
            keep = ~change
            nll = -logp[torch.arange(64, device=h.device), tgt]
            ch = nll[change].mean() if bool(change.any()) else nll.new_zeros(())
            kp = nll[keep].mean() if bool(keep.any()) else nll.new_zeros(())
            return ch + float(eta_preserve) * kp, ch, kp

        mod.logits = logits
        mod.proba = proba
        mod.predict = predict
        mod.state_loss = state_loss
        mod.d_model = d
        return mod


def sklearn_proba_pack(models, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    out = np.zeros((x.shape[0], 64, 3), dtype=np.float64)
    for j, clf in enumerate(models):
        p = clf.predict_proba(x)
        for col, cls in enumerate(np.asarray(clf.classes_)):
            out[:, j, int(cls)] = p[:, col]
    return out


def parity_report(decoder: LinearBoardDecoder, models, x: np.ndarray) -> dict:
    a = decoder.proba_np(x)
    b = sklearn_proba_pack(models, x)
    pred_a = decoder.predict_np(x)
    pred_b = np.stack([m.predict(x) for m in models], axis=1)
    return {
        "max_abs_proba": float(np.max(np.abs(a - b))),
        "mean_abs_proba": float(np.mean(np.abs(a - b))),
        "square_agree": float((pred_a == pred_b).mean()),
        "board_agree": float((pred_a == pred_b).all(axis=1).mean()),
        "n": int(x.shape[0]),
        "d_model": decoder.d_model,
        "hook": decoder.hook,
        "target": decoder.target,
    }


def project_budget(delta: np.ndarray, budget: float) -> np.ndarray:
    nrm = float(np.linalg.norm(delta))
    if nrm > float(budget) and nrm > 0:
        return delta * (float(budget) / nrm)
    return delta


def _failed_edit(reason: str, **extra) -> dict:
    row = {
        "status": STATUS_FAILED,
        "valid_edit": 0,
        "edited": None,
        "delta": None,
        "reason": reason,
        "steps_completed": 0,
        "termination": reason,
        "traces": [],
        "used_donor_activation": 0,
        "used_model_logits": 0,
    }
    row.update(extra)
    return row


def _labels_valid(lab) -> bool:
    arr = np.asarray(lab)
    return arr.shape == (64,) and np.isin(arr, (0, 1, 2)).all()


def iterative_state_edit(
    h,
    target_lab,
    rec_lab,
    decoder_torch,
    *,
    steps: int = 40,
    lr: float = 0.4,
    budget: float = 8.0,
    lambda_norm: float = 0.01,
    eta_preserve: float = 1.0,
    mu_manifold: float = 0.0,
    mean_act=None,
    pca_basis=None,
    rowspace=None,
) -> dict:
    """Real multi-step autograd editor. Fail-closed: never returns h as a successful edit."""
    import torch

    if int(steps) < 2:
        return _failed_edit("invalid_step_count", configured_steps=int(steps))
    if not _labels_valid(target_lab) or not _labels_valid(rec_lab):
        return _failed_edit("invalid_target_or_recipient_labels")
    h_np = np.asarray(h, dtype=np.float64).reshape(-1)
    if not np.isfinite(h_np).all():
        return _failed_edit("nonfinite_input_activation")
    device = h.device if torch.is_tensor(h) else torch.device("cpu")
    h0 = torch.as_tensor(h, dtype=torch.float32, device=device)
    if not torch.isfinite(h0).all():
        return _failed_edit("nonfinite_input_activation")
    if rowspace is not None:
        basis_np = np.asarray(rowspace, dtype=np.float64)
        if basis_np.ndim != 2 or basis_np.shape[0] != h_np.size:
            return _failed_edit("invalid_rowspace_shape")
        gram = basis_np.T @ basis_np
        if not np.allclose(gram, np.eye(basis_np.shape[1]), atol=1e-5):
            return _failed_edit("rowspace_not_orthonormal")
    delta = torch.zeros_like(h0, requires_grad=True)
    opt = torch.optim.SGD([delta], lr=float(lr))
    traces = []
    term = "max_steps"
    completed = 0
    last_loss = None
    for t in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        edited = h0 + delta
        if not torch.isfinite(edited).all():
            return _failed_edit("nonfinite_activation", steps_completed=completed, traces=traces)
        loss, ch, kp = decoder_torch.state_loss(edited, target_lab, rec_lab, eta_preserve=eta_preserve)
        loss = loss + float(lambda_norm) * (delta * delta).sum()
        if mu_manifold and mean_act is not None:
            centered = edited - torch.as_tensor(mean_act, dtype=torch.float32, device=device)
            if pca_basis is not None:
                basis = torch.as_tensor(pca_basis, dtype=torch.float32, device=device)
                proj = (centered @ basis) @ basis.T
                loss = loss + float(mu_manifold) * ((centered - proj) ** 2).sum()
            else:
                loss = loss + float(mu_manifold) * (centered ** 2).sum()
        if not torch.isfinite(loss):
            return _failed_edit("nonfinite_loss", steps_completed=completed, traces=traces)
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            return _failed_edit("nonfinite_grad", steps_completed=completed, traces=traces)
        gnorm = float(delta.grad.detach().norm())
        opt.step()
        with torch.no_grad():
            if rowspace is not None:
                basis = torch.as_tensor(rowspace, dtype=torch.float32, device=device)
                projected = (delta @ basis) @ basis.T
                if not torch.isfinite(projected).all():
                    return _failed_edit("nonfinite_rowspace_projection", steps_completed=completed, traces=traces)
                leftover = float(torch.norm(projected - (projected @ basis) @ basis.T))
                if leftover > 1e-4:
                    return _failed_edit("rowspace_projection_failed", steps_completed=completed, traces=traces)
                delta.copy_(projected)
            nrm = float(delta.norm())
            if nrm > float(budget) + 1e-5 and nrm > 0:
                delta.mul_(float(budget) / nrm)
            if float(delta.norm()) > float(budget) + 1e-4:
                return _failed_edit("norm_constraint_violation", steps_completed=completed, traces=traces)
        completed += 1
        last_loss = float(loss.detach())
        traces.append(
            {
                "step": t,
                "loss": last_loss,
                "change_nll": float(ch.detach()),
                "keep_nll": float(kp.detach()),
                "grad_norm": gnorm,
                "delta_norm": float(delta.detach().norm()),
            }
        )
        if t > 0 and abs(traces[-1]["loss"] - traces[-2]["loss"]) < 1e-8:
            term = "early_stop_flat"
            break
    if completed < 2:
        return _failed_edit("missing_optimization_steps", steps_completed=completed, traces=traces, termination=term)
    edited = (h0 + delta.detach()).cpu().numpy().reshape(-1)
    if not np.isfinite(edited).all():
        return _failed_edit("nonfinite_activation", steps_completed=completed, traces=traces)
    pred = decoder_torch.predict(h0 + delta.detach()).detach().cpu().numpy().reshape(-1)
    tgt = np.asarray(target_lab)
    rec = np.asarray(rec_lab)
    change = tgt != rec
    keep = ~change
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "reason": term,
        "edited": edited,
        "delta": delta.detach().cpu().numpy().reshape(-1),
        "steps_completed": completed,
        "termination": term,
        "initial_loss": traces[0]["loss"] if traces else float("nan"),
        "final_loss": last_loss if last_loss is not None else float("nan"),
        "delta_norm": float(np.linalg.norm(delta.detach().cpu().numpy())),
        "grad_norm": traces[-1]["grad_norm"] if traces else float("nan"),
        "changed_acc": float((pred[change] == tgt[change]).mean()) if change.any() else float("nan"),
        "unchanged_acc": float((pred[keep] == tgt[keep]).mean()) if keep.any() else float("nan"),
        "board_exact": float((pred == tgt).all()),
        "traces": traces,
        "used_donor_activation": 0,
        "used_model_logits": 0,
        "near_zero_edit": int(float(np.linalg.norm(delta.detach().cpu().numpy())) < 1e-6),
    }


def torch_parity_report(decoder: LinearBoardDecoder, models, x: np.ndarray) -> dict:
    """Compare the torch decoder used for optimization to sklearn, not the numpy clone."""
    import torch

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    sk = sklearn_proba_pack(models, x)
    pred_sk = np.stack([m.predict(x) for m in models], axis=1)
    mod = decoder.as_torch()
    with torch.no_grad():
        pt = mod.proba(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
        pred_t = mod.predict(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    return {
        "backend": "torch_vs_sklearn",
        "max_abs_proba": float(np.max(np.abs(pt - sk))),
        "mean_abs_proba": float(np.mean(np.abs(pt - sk))),
        "square_agree": float((pred_t == pred_sk).mean()),
        "board_agree": float((pred_t == pred_sk).all(axis=1).mean()),
        "n": int(x.shape[0]),
        "d_model": decoder.d_model,
        "hook": decoder.hook,
        "target": decoder.target,
        "class_sets": [list(s) for s in decoder.class_sets],
        "n_binary_squares": int(sum(len(s) == 2 for s in decoder.class_sets)),
        "n_multiclass_squares": int(sum(len(s) == 3 for s in decoder.class_sets)),
    }


def finite_difference_grad_check(decoder_torch, h, target_lab, rec_lab, eps: float = 1e-3, n_coord: int = 8, seed: int = 0) -> dict:
    import torch

    h0 = torch.as_tensor(h, dtype=torch.float32)
    h0 = h0.detach().clone().requires_grad_(True)
    loss, _, _ = decoder_torch.state_loss(h0, target_lab, rec_lab)
    loss.backward()
    analytic = h0.grad.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(analytic.size, size=min(n_coord, analytic.size), replace=False)
    rel = []
    for i in idx:
        hp = h0.detach().clone()
        hm = h0.detach().clone()
        hp.view(-1)[i] += eps
        hm.view(-1)[i] -= eps
        lp, _, _ = decoder_torch.state_loss(hp, target_lab, rec_lab)
        lm, _, _ = decoder_torch.state_loss(hm, target_lab, rec_lab)
        fd = float((lp - lm) / (2 * eps))
        rel.append(abs(fd - float(analytic.reshape(-1)[i])))
    return {
        "max_abs_fd_error": float(np.max(rel)) if rel else float("nan"),
        "mean_abs_fd_error": float(np.mean(rel)) if rel else float("nan"),
        "grad_finite": bool(np.isfinite(analytic).all()),
        "n_coord": int(len(idx)),
    }


def train_activation_stats(X: np.ndarray, rank: int = 32) -> dict:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    xc = X - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(int(rank), vt.shape[0])
    return {"mean": mean, "pca_basis": vt[:k].T, "singular_values": s[:k], "n": int(X.shape[0]), "rank": k}


def isolate_r2_results_dir(requested: str | Path) -> dict:
    """Always write 08R2 outputs under 08r2/. Never overwrite 08a–08i or 08R1."""
    requested = Path(requested)
    out = requested if requested.name == R2_DIRNAME else requested / R2_DIRNAME
    return {
        "requested": str(requested),
        "results_dir": str(out),
        "nested": int(out != requested),
        "reason": "08R2 writes only under 08r2/",
    }


def load_holdout_game_ids(path: str | Path) -> dict:
    from src.toy_othello import sha256_json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = {str(g["game_id"]) for g in raw.get("games", [])}
    payload = sha256_json([g.get("sha256") for g in raw.get("games", [])])
    return {
        "path": str(path),
        "split": raw.get("split"),
        "n_games": int(raw.get("n_games", len(ids))),
        "game_ids": ids,
        "manifest_sha256": raw.get("manifest_sha256"),
        "payload_sha256": payload,
        "file_sha256": sha256_file(path),
        "hash_ok": payload == raw.get("manifest_sha256"),
        "used_in_08r1_method_development": raw.get("used_in_08r1_method_development"),
    }


def confirmation_guard_text() -> str:
    return (
        "08R2 confirmation is sealed. Do not evaluate the 400-game confirmation "
        "split during implementation or validation."
    )


def fit_reusable_map(X_src, X_tgt, lab_src, lab_tgt, rank: int = 16, ridge: float = 10.0, min_pairs: int = 32) -> dict:
    if len(X_src) < int(min_pairs):
        return {
            "status": STATUS_EDITOR_C_FIT_FAILED,
            "reason": "INSUFFICIENT_TRAIN_PAIRS",
            "n": int(len(X_src)),
            "min_pairs": int(min_pairs),
        }
    D = []
    Y = []
    for xs, xt, ls, lt in zip(X_src, X_tgt, lab_src, lab_tgt):
        if np.array_equal(ls, lt):
            continue
        a = np.eye(3)[np.asarray(ls, dtype=int)].reshape(-1)
        b = np.eye(3)[np.asarray(lt, dtype=int)].reshape(-1)
        D.append(b - a)
        Y.append(np.asarray(xt, dtype=np.float64) - np.asarray(xs, dtype=np.float64))
    if len(D) < int(min_pairs):
        return {
            "status": STATUS_EDITOR_C_FIT_FAILED,
            "reason": "INSUFFICIENT_STATE_CHANGES",
            "n": int(len(D)),
            "min_pairs": int(min_pairs),
        }
    D = np.stack(D)
    Y = np.stack(Y)
    gram = D.T @ D + float(ridge) * np.eye(D.shape[1])
    W = np.linalg.solve(gram, D.T @ Y)
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    k = min(int(rank), vt.shape[0])
    Wk = (u[:, :k] * s[:k]) @ vt[:k]
    return {"status": "FIT", "W": Wk, "rank": k, "n": int(len(D)), "ridge": ridge, "reason": "FIT"}


def apply_reusable_map(W: np.ndarray, rec_x, rec_lab, tgt_lab, budget: float) -> dict:
    if W is None:
        return _failed_edit("editor_c_missing_W")
    if not _labels_valid(rec_lab) or not _labels_valid(tgt_lab):
        return _failed_edit("invalid_target_or_recipient_labels")
    a = np.eye(3)[np.asarray(rec_lab, dtype=int)].reshape(-1)
    b = np.eye(3)[np.asarray(tgt_lab, dtype=int)].reshape(-1)
    delta = (b - a) @ np.asarray(W, dtype=np.float64)
    edited = np.asarray(rec_x, dtype=np.float64).reshape(-1) + project_budget(delta, budget)
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "edited": edited,
        "delta": edited - np.asarray(rec_x, dtype=np.float64).reshape(-1),
        "delta_norm": float(np.linalg.norm(edited - np.asarray(rec_x, dtype=np.float64).reshape(-1))),
        "used_donor_activation": 0,
        "used_model_logits": 0,
        "reason": "reusable_map",
    }


def _pair_kind(rec: dict, don: dict) -> tuple[str | None, int]:
    if rec.get("terminal") or don.get("terminal"):
        return None, -1
    if int(rec["player"]) != int(don["player"]):
        return None, -1
    if rec["game_id"] == don["game_id"]:
        return None, -1
    ham = int(board_hamming(board_arr(rec), board_arr(don)))
    lr, ld = set(rec["legal"]), set(don["legal"])
    if lr != ld and ham <= 8:
        return "B_similar_board_legal_change", ham
    if lr != ld:
        return "A_legal_diff", ham
    if ham > 0:
        return "C_same_legal_diff_board", ham
    return None, ham


def pair_category_census(rows: list[dict], holdout_ids: set[str] | None = None, occupancy_slack: int = 0) -> dict:
    """Deterministic enumeration inside occupancy buckets. Does not fabricate pairs."""
    require_no_confirmation(rows, holdout_ids)
    labeled = add_labels(list(rows))
    elig = [i for i, r in enumerate(labeled) if r.get("legal") and not r.get("terminal")]
    buckets: dict[tuple, list[int]] = {}
    for i in elig:
        r = labeled[i]
        buckets.setdefault((int(r["player"]), int(r["occupied"]) // 4), []).append(i)
    counts = {"A_legal_diff": 0, "B_similar_board_legal_change": 0, "C_same_legal_diff_board": 0}
    hammings = {"A_legal_diff": [], "B_similar_board_legal_change": [], "C_same_legal_diff_board": []}
    rejections = {"same_game": 0, "same_state": 0, "cross_bucket": 0}
    found: dict[str, list[tuple[int, int, int]]] = {k: [] for k in counts}
    keys = sorted(buckets)
    for key in keys:
        idxs = buckets[key]
        neighbor = []
        if occupancy_slack:
            for dk in range(-occupancy_slack, occupancy_slack + 1):
                neighbor.extend(buckets.get((key[0], key[1] + dk), []))
            idxs = sorted(set(neighbor))
        for i in idxs:
            rec = labeled[i]
            for j in idxs:
                if i == j:
                    continue
                don = labeled[j]
                if rec["game_id"] == don["game_id"]:
                    rejections["same_game"] += 1
                    continue
                kind, ham = _pair_kind(rec, don)
                if kind is None:
                    rejections["same_state"] += 1
                    continue
                counts[kind] += 1
                hammings[kind].append(ham)
                found[kind].append((i, j, ham))
    summary = {}
    for kind, vals in hammings.items():
        summary[kind] = {
            "n_ordered_pairs": counts[kind],
            "n_recip_games": len({labeled[i]["game_id"] for i, _, _ in found[kind]}),
            "n_target_games": len({labeled[j]["game_id"] for _, j, _ in found[kind]}),
            "hamming_min": int(min(vals)) if vals else None,
            "hamming_max": int(max(vals)) if vals else None,
            "hamming_median": float(np.median(vals)) if vals else None,
        }
    return {
        "occupancy_slack_buckets": int(occupancy_slack),
        "n_eligible": len(elig),
        "rejections": rejections,
        "categories": summary,
        "found": found,
        "labeled": labeled,
    }


def r2_pair_categories(rows: list[dict], n: int, seed: int, holdout_ids: set[str] | None = None, n_b_reserve: int = 8) -> list[dict]:
    """Reserve genuine B pairs when they exist. Do not relabel A as B/C."""
    census = pair_category_census(rows, holdout_ids, occupancy_slack=0)
    labeled = census["labeled"]
    found = census["found"]
    slack_census = pair_category_census(rows, holdout_ids, occupancy_slack=1)
    primary = []
    used_recip = set()
    used_idx = set()

    def _take(kind: str, limit: int) -> None:
        for i, j, ham in found[kind]:
            if len([p for p in primary if p["r2_type"] == kind]) >= limit:
                return
            rec = labeled[i]
            if rec["game_id"] in used_recip or i in used_idx:
                continue
            don = labeled[j]
            lr, ld = set(rec["legal"]), set(don["legal"])
            primary.append(
                {
                    "recip": i,
                    "donor": j,
                    "type": kind,
                    "r2_type": kind,
                    "hamming": ham,
                    "n_became_legal": len(ld - lr),
                    "n_became_illegal": len(lr - ld),
                    "same_player": 1,
                    "recip_game": rec["game_id"],
                    "target_game": don["game_id"],
                    "used_confirmation": 0,
                    "occupancy_slack": 0,
                }
            )
            used_recip.add(rec["game_id"])
            used_idx.add(i)

    n_b = min(int(n_b_reserve), int(n), census["categories"]["B_similar_board_legal_change"]["n_recip_games"])
    _take("B_similar_board_legal_change", n_b)
    _take("A_legal_diff", int(n) - len(primary))
    n_c_avail = slack_census["categories"]["C_same_legal_diff_board"]["n_recip_games"]
    c_pairs = []
    if n_c_avail and found["C_same_legal_diff_board"]:
        _take("C_same_legal_diff_board", min(4, n_c_avail))
    elif slack_census["categories"]["C_same_legal_diff_board"]["n_ordered_pairs"]:
        # Broader bucket (±1 occupancy quartile) only if strict search found none.
        extra = slack_census["found"]["C_same_legal_diff_board"]
        labeled_s = slack_census["labeled"]
        for i, j, ham in extra:
            rec = labeled_s[i]
            if rec["game_id"] in used_recip:
                continue
            don = labeled_s[j]
            c_pairs.append(
                {
                    "recip": i,
                    "donor": j,
                    "type": "C_same_legal_diff_board",
                    "r2_type": "C_same_legal_diff_board",
                    "hamming": ham,
                    "n_became_legal": 0,
                    "n_became_illegal": 0,
                    "same_player": 1,
                    "recip_game": rec["game_id"],
                    "target_game": don["game_id"],
                    "used_confirmation": 0,
                    "occupancy_slack": 1,
                    "note": "found only after ±1 occupancy-bucket slack",
                }
            )
            used_recip.add(rec["game_id"])
            if len(c_pairs) >= 4:
                break
    primary.extend(c_pairs)
    for p in primary:
        p["census"] = {
            "A": census["categories"]["A_legal_diff"]["n_ordered_pairs"],
            "B": census["categories"]["B_similar_board_legal_change"]["n_ordered_pairs"],
            "C_strict": census["categories"]["C_same_legal_diff_board"]["n_ordered_pairs"],
            "C_slack1": slack_census["categories"]["C_same_legal_diff_board"]["n_ordered_pairs"],
        }
    wrong = []
    rng = np.random.default_rng(seed + 19)
    analysis = [p for p in primary if p["r2_type"] in {"A_legal_diff", "B_similar_board_legal_change"}]
    for p in analysis:
        rec = labeled[p["recip"]]
        intended_legal = set(labeled[p["donor"]]["legal"])
        cands = [
            i
            for i, r in enumerate(labeled)
            if r["game_id"] not in {p["recip_game"], p["target_game"]}
            and int(r["player"]) == int(rec["player"])
            and set(r["legal"]) != intended_legal
            and not r.get("terminal")
        ]
        if not cands:
            continue
        j = int(rng.choice(cands))
        wr = labeled[j]
        wrong.append(
            {
                "recip": p["recip"],
                "donor": j,
                "r2_type": "D_wrong_target",
                "type": "D_wrong_target",
                "hamming": int(board_hamming(board_arr(rec), board_arr(wr))),
                "recip_game": rec["game_id"],
                "target_game": wr["game_id"],
                "intended_target_game": p["target_game"],
                "intended_recip": p["recip"],
                "intended_donor": p["donor"],
                "same_player": 1,
                "used_confirmation": 0,
                "n_became_legal": len(set(wr["legal"]) - set(rec["legal"])),
                "n_became_illegal": len(set(rec["legal"]) - set(wr["legal"])),
            }
        )
    return primary + wrong


def pair_census_public(rows: list[dict], holdout_ids: set[str] | None = None) -> dict:
    strict = pair_category_census(rows, holdout_ids, 0)
    slack = pair_category_census(rows, holdout_ids, 1)
    return {
        "strict_same_occupancy_bucket": strict["categories"],
        "slack_one_occupancy_bucket": slack["categories"],
        "rejections_strict": strict["rejections"],
        "n_eligible": strict["n_eligible"],
        "c_available_strict": strict["categories"]["C_same_legal_diff_board"]["n_ordered_pairs"] > 0,
        "c_available_slack1": slack["categories"]["C_same_legal_diff_board"]["n_ordered_pairs"] > 0,
        "b_available": strict["categories"]["B_similar_board_legal_change"]["n_recip_games"] > 0,
    }


def decoder_state_effects(decoder: LinearBoardDecoder, h: np.ndarray, rec_lab, tgt_lab) -> dict:
    pred = decoder.predict_np(h).reshape(-1)
    tgt = np.asarray(tgt_lab, dtype=np.int64).reshape(-1)
    rec = np.asarray(rec_lab, dtype=np.int64).reshape(-1)
    change = tgt != rec
    keep = ~change
    return {
        "changed_acc": float((pred[change] == tgt[change]).mean()) if change.any() else float("nan"),
        "unchanged_acc": float((pred[keep] == tgt[keep]).mean()) if keep.any() else float("nan"),
        "board_exact": float((pred == tgt).all()),
        "square_acc": float((pred == tgt).mean()),
        "n_changed": int(change.sum()),
        "n_unchanged": int(keep.sum()),
        "pred_rel": pred,
    }


def causal_scores(pb, pe, rec_legal, tgt_legal) -> dict:
    mets = intervention_probability_metrics(pb, pe, rec_legal, tgt_legal)
    gain = mets["became_legal_d_mass"]
    loss = mets["became_illegal_d_mass"]
    mets["dP_gain"] = gain
    mets["dP_loss"] = loss
    mets["dP_gain_defined"] = int(gain == gain)
    mets["dP_loss_defined"] = int(loss == loss)
    mets["stayed_legal_defined"] = int(mets["stayed_legal_d_mass"] == mets["stayed_legal_d_mass"])
    mets["stayed_illegal_defined"] = int(mets["stayed_illegal_d_mass"] == mets["stayed_illegal_d_mass"])
    both = bool(mets["dP_gain_defined"] and mets["dP_loss_defined"])
    mets["joint_metric_eligible"] = int(both)
    mets["S_legal"] = float(gain - loss) if both else float("nan")
    mets["kl"] = kl_row(pb, pe)
    return mets


def random_same_norm_edit(h: np.ndarray, budget: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    vec = rng.normal(size=np.asarray(h).shape)
    nrm = float(np.linalg.norm(vec))
    if nrm == 0:
        return np.asarray(h, dtype=np.float64)
    return np.asarray(h, dtype=np.float64) + vec * (float(budget) / nrm)


def pca_patch(rec_h: np.ndarray, don_h: np.ndarray, basis: np.ndarray, complement: bool = False) -> dict:
    rec = np.asarray(rec_h, dtype=np.float64).reshape(-1)
    don = np.asarray(don_h, dtype=np.float64).reshape(-1)
    if rec.shape != don.shape:
        return _failed_edit("pca_shape_mismatch")
    delta = don - rec
    proj = (delta @ basis) @ basis.T
    edited = rec + (delta - proj if complement else proj)
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "edited": edited,
        "delta_norm": float(np.linalg.norm(edited - rec)),
        "configured_budget": None,
        "used_donor_activation": 1,
        "used_model_logits": 0,
        "reason": "pca_complement" if complement else "pca_donor",
    }


def attribution_sign_edit(h, target_lab, rec_lab, decoder_torch, budget: float) -> dict:
    """One-step sign of decoder state-loss gradient. Not a model-logit method."""
    import torch

    if not _labels_valid(target_lab) or not _labels_valid(rec_lab):
        return _failed_edit("invalid_target_or_recipient_labels")
    h0 = torch.as_tensor(h, dtype=torch.float32).detach().clone().requires_grad_(True)
    loss, _, _ = decoder_torch.state_loss(h0, target_lab, rec_lab)
    if not torch.isfinite(loss):
        return _failed_edit("nonfinite_loss")
    loss.backward()
    if h0.grad is None or not torch.isfinite(h0.grad).all():
        return _failed_edit("nonfinite_grad")
    g = h0.grad.detach().cpu().numpy().reshape(-1)
    vec = -np.sign(g)
    nrm = float(np.linalg.norm(vec))
    rec = np.asarray(h, dtype=np.float64).reshape(-1)
    if nrm == 0:
        return _failed_edit("zero_attribution_gradient")
    edited = rec + vec * (float(budget) / nrm)
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "edited": edited,
        "delta_norm": float(np.linalg.norm(edited - rec)),
        "used_donor_activation": 0,
        "used_model_logits": 0,
        "loss_name": "decoder_state_nll",
        "reason": "attribution_sign_decoder_state_loss",
    }


def random_edit_safe(h: np.ndarray, budget: float, seed: int) -> dict:
    rec = np.asarray(h, dtype=np.float64).reshape(-1)
    if not np.isfinite(rec).all():
        return _failed_edit("nonfinite_input_activation")
    edited = random_same_norm_edit(rec, budget, seed).reshape(-1)
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "edited": edited,
        "delta_norm": float(np.linalg.norm(edited - rec)),
        "used_donor_activation": 0,
        "used_model_logits": 0,
        "reason": "random_same_norm",
    }


def probe_direction_edit_safe(x, rec_lab, don_lab, models, budget: float) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rec_lab = np.asarray(rec_lab)
    don_lab = np.asarray(don_lab)
    if not _labels_valid(rec_lab) or not _labels_valid(don_lab):
        return _failed_edit("invalid_target_or_recipient_labels")
    delta = np.zeros_like(x)
    n = 0
    skipped = 0
    for j in range(64):
        if rec_lab[j] == don_lab[j]:
            continue
        clf = models[j]
        if not hasattr(clf, "coef_") or not hasattr(clf, "classes_"):
            return _failed_edit("probe_missing_linear_params", square=j)
        classes = np.asarray(clf.classes_)
        coef = np.asarray(clf.coef_, dtype=np.float64)
        src = np.where(classes == int(rec_lab[j]))[0]
        tgt = np.where(classes == int(don_lab[j]))[0]
        if src.size != 1 or tgt.size != 1:
            skipped += 1
            continue
        delta = delta + (coef[int(tgt[0])] - coef[int(src[0])])
        n += 1
    if n == 0:
        return _failed_edit("probe_direction_no_mapped_class_changes", skipped=skipped)
    nrm = float(np.linalg.norm(delta))
    if nrm <= 1e-12:
        return _failed_edit("probe_direction_zero_delta")
    vec = delta * (float(budget) / nrm) if nrm > float(budget) else delta
    edited = x + vec
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "edited": edited,
        "delta_norm": float(np.linalg.norm(vec)),
        "n_squares_used": n,
        "n_squares_skipped_class": skipped,
        "used_donor_activation": 0,
        "used_model_logits": 0,
        "reason": "probe_direction_class_aligned",
    }


def mean_square_edit_safe(Xtr, ytr, rec_x, rec_lab, don_lab, budget: float) -> dict:
    rec = np.asarray(rec_x, dtype=np.float64).reshape(-1)
    rec_lab = np.asarray(rec_lab)
    don_lab = np.asarray(don_lab)
    ytr = np.asarray(ytr)
    Xtr = np.asarray(Xtr)
    delta = np.zeros_like(rec)
    n = 0
    for j in range(64):
        if rec_lab[j] == don_lab[j]:
            continue
        src = ytr[:, j] == rec_lab[j]
        tgt = ytr[:, j] == don_lab[j]
        if src.sum() < 8 or tgt.sum() < 8:
            continue
        delta = delta + (Xtr[tgt].mean(0) - Xtr[src].mean(0))
        n += 1
    if n == 0:
        return _failed_edit("mean_square_insufficient_class_counts")
    nrm = float(np.linalg.norm(delta))
    if nrm <= 1e-12:
        return _failed_edit("mean_square_zero_delta")
    vec = delta * (float(budget) / nrm) if nrm > float(budget) else delta
    edited = rec + vec
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "edited": edited,
        "delta_norm": float(np.linalg.norm(vec)),
        "n_squares_used": n,
        "used_donor_activation": 0,
        "used_model_logits": 0,
        "reason": "mean_square_train_stats",
    }


def wrong_square_edit_safe(x, rec_lab, don_lab, models, budget: float, rng) -> dict:
    from src.othello_r1_eval import wrong_square_linear_edit

    out = wrong_square_linear_edit(x, rec_lab, don_lab, models, budget, rng)
    if out is None:
        return _failed_edit("wrong_square_unavailable")
    return {
        "status": STATUS_EXECUTED,
        "valid_edit": 1,
        "edited": np.asarray(out["edited"], dtype=np.float64).reshape(-1),
        "delta_norm": float(out.get("edit_norm", np.nan)),
        "wrong_square": out.get("wrong_square"),
        "used_donor_activation": 0,
        "used_model_logits": 0,
        "reason": "wrong_square_target_specificity",
    }


REQUIRED_BASELINES = (
    "full_donor_patch",
    "pca32_donor",
    "pca32_complement",
    "probe_direction",
    "mean_square",
    "random_same_norm",
    "wrong_target",
    "wrong_square",
    "wrong_donor",
    "noop",
    "attribution_sign",
    "logit_steer",
)


def outcome_bucket(board_exact: float, dP_gain, dP_loss) -> str:
    rep = board_exact >= 1.0 or (board_exact == board_exact and board_exact >= 0.9)
    beh = (dP_gain == dP_gain and dP_gain > 0) and (dP_loss == dP_loss and dP_loss < 0)
    if rep and beh:
        return "A_decode_and_behavior"
    if rep and not beh:
        return "B_probe_fooling"
    if (not rep) and beh:
        return "C_behavior_without_decode"
    return "D_neither"
