"""Controlled toy world for representation detection: modular addition.

Experiment 0B trains a small Transformer on (a + b) mod p, where the latent state is known
exactly by construction. This module provides the dataset, the model, the known cyclic latent
state, latent-state germs fit as probes, ridge transports between sites, and the latent-state
obstruction energy O_y. The point of the toy world is that a probe can be checked against
ground truth before any steering claim is made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EQUALS_TOKEN_OFFSET = 0  # the '=' token id is p, i.e. one past the last number token


@dataclass
class ModularConfig:
    p: int = 113
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1
    train_frac: float = 0.3
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    steps: int = 25000

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def equals_token(self) -> int:
        return self.p

    def to_dict(self) -> dict:
        return {
            "p": self.p,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_mlp": self.d_mlp,
            "n_layers": self.n_layers,
            "train_frac": self.train_frac,
            "seed": self.seed,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "steps": self.steps,
        }


# ----------------------------------------------------------------------------------
# Dataset: every (a, b) pair, tokenized as [a, b, '='], label (a + b) mod p
# ----------------------------------------------------------------------------------


def make_dataset(cfg: ModularConfig):
    import torch

    a = torch.arange(cfg.p).repeat_interleave(cfg.p)
    b = torch.arange(cfg.p).repeat(cfg.p)
    equals = torch.full_like(a, cfg.equals_token)
    tokens = torch.stack([a, b, equals], dim=1)
    labels = (a + b) % cfg.p
    return tokens, labels


def split_dataset(cfg: ModularConfig, tokens, labels):
    import torch

    generator = torch.Generator().manual_seed(cfg.seed)
    n = tokens.shape[0]
    perm = torch.randperm(n, generator=generator)
    n_train = int(cfg.train_frac * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return (tokens[train_idx], labels[train_idx]), (tokens[test_idx], labels[test_idx])


# ----------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------


def build_model(cfg: ModularConfig, device: str | None = None):
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tl_cfg = HookedTransformerConfig(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        d_mlp=cfg.d_mlp,
        act_fn="relu",
        normalization_type=None,
        d_vocab=cfg.p + 1,
        d_vocab_out=cfg.p,
        n_ctx=3,
        init_weights=True,
        device=device,
        seed=cfg.seed,
    )
    return HookedTransformer(tl_cfg)


def final_logits(model, tokens):
    """Logits at the '=' position, shape [batch, p]."""
    return model(tokens)[:, -1, :]


def loss_and_accuracy(logits, labels):
    import torch.nn.functional as F

    loss = F.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, accuracy


# ----------------------------------------------------------------------------------
# Known latent state: cyclic (Fourier) structure
# ----------------------------------------------------------------------------------


def fourier_basis(p: int, device: str | None = None):
    """Rows: [constant, cos(2*pi*k*i/p), sin(2*pi*k*i/p), ...], each unit norm. Shape [p, p]."""
    import torch

    rows = [torch.ones(p)]
    for k in range(1, p // 2 + 1):
        idx = torch.arange(p, dtype=torch.float32)
        rows.append(torch.cos(2 * math.pi * k * idx / p))
        rows.append(torch.sin(2 * math.pi * k * idx / p))
    basis = torch.stack(rows[:p], dim=0)
    basis = basis / basis.norm(dim=-1, keepdim=True)
    if device is not None:
        basis = basis.to(device)
    return basis


def frequency_of_basis_row(row_index: int) -> int:
    """Map a Fourier basis row index to its frequency k (row 0 is the constant, k = 0)."""
    if row_index == 0:
        return 0
    return (row_index + 1) // 2


def fourier_power_by_frequency(matrix, p: int) -> dict[int, float]:
    """Squared Fourier power of `matrix` [p, d] grouped by frequency k."""
    basis = fourier_basis(p, device=str(matrix.device))
    coeffs = basis @ matrix.float()  # [p, d]
    power = coeffs.pow(2).sum(dim=-1)
    out: dict[int, float] = {}
    for row in range(p):
        k = frequency_of_basis_row(row)
        out[k] = out.get(k, 0.0) + float(power[row])
    return out


def key_frequencies(model, p: int, top_k: int = 5) -> list[int]:
    """Frequencies carrying the most embedding power, excluding the constant term."""
    embed = model.W_E[:p].detach()
    power = fourier_power_by_frequency(embed, p)
    power.pop(0, None)
    ranked = sorted(power.items(), key=lambda kv: kv[1], reverse=True)
    return sorted(k for k, _ in ranked[:top_k])


def abstraction_strength(model, p: int, top_k: int = 5) -> float:
    """A_cyc(t): fraction of embedding Fourier power in the top-k non-constant frequencies.

    High values mean the model represents inputs through a small set of cyclic frequencies,
    which is the known generalizing abstraction for modular addition.
    """
    embed = model.W_E[:p].detach()
    power = fourier_power_by_frequency(embed, p)
    power.pop(0, None)
    total = sum(power.values())
    if total <= 0:
        return 0.0
    ranked = sorted(power.values(), reverse=True)
    return float(sum(ranked[:top_k]) / total)


def latent_targets(tokens, p: int, freqs: list[int]):
    """Ground-truth cyclic latent state of the sum: [cos, sin] per key frequency.

    This is computed from the task simulator, not from the model, so probes can be scored
    against known truth.
    """
    import torch

    a = tokens[:, 0].float()
    b = tokens[:, 1].float()
    s = (a + b) % p
    cols = []
    for k in freqs:
        angle = 2 * math.pi * k * s / p
        cols.append(torch.cos(angle))
        cols.append(torch.sin(angle))
    return torch.stack(cols, dim=-1)


def latent_input_targets(tokens, p: int, freqs: list[int]):
    """Cyclic latent state of the two inputs separately, used for early-site germs."""
    import torch

    cols = []
    for operand in (0, 1):
        v = tokens[:, operand].float()
        for k in freqs:
            angle = 2 * math.pi * k * v / p
            cols.append(torch.cos(angle))
            cols.append(torch.sin(angle))
    return torch.stack(cols, dim=-1)


def decode_sum_from_latent(latent, p: int, freqs: list[int]):
    """Recover the predicted sum from cyclic features by maximizing cosine agreement.

    Used to test whether a germ decodes the true latent state out of sample.
    """
    import torch

    candidates = torch.arange(p, dtype=torch.float32, device=latent.device)
    cols = []
    for k in freqs:
        angle = 2 * math.pi * k * candidates / p
        cols.append(torch.cos(angle))
        cols.append(torch.sin(angle))
    table = torch.stack(cols, dim=-1)  # [p, 2 * n_freqs]
    scores = latent.float() @ table.T  # [batch, p]
    return scores.argmax(dim=-1)


# ----------------------------------------------------------------------------------
# Sites and activation caching
# ----------------------------------------------------------------------------------


@dataclass
class ToySite:
    site_id: str
    hook_name: str
    component: str
    layer: int
    position: int
    head: int | None = None

    def to_row(self) -> dict:
        return {
            "site_id": self.site_id,
            "component": self.component,
            "layer": self.layer,
            "position": self.position,
            "head": -1 if self.head is None else self.head,
        }


def make_toy_sites(cfg: ModularConfig, position: int = 2, include_operand_sites: bool = True) -> list[ToySite]:
    """An ordered chain of sites through the computation at the '=' position.

    The order defines the edge set: consecutive sites are connected, so obstruction measures
    whether latent-state evidence transports coherently along the computation.
    """
    sites: list[ToySite] = []
    if include_operand_sites:
        for pos in (0, 1):
            sites.append(
                ToySite(
                    site_id=f"embed_pos{pos}",
                    hook_name="hook_embed",
                    component="embed",
                    layer=0,
                    position=pos,
                )
            )
    for layer in range(cfg.n_layers):
        sites.append(
            ToySite(
                site_id=f"L{layer}_resid_pre",
                hook_name=f"blocks.{layer}.hook_resid_pre",
                component="resid_pre",
                layer=layer,
                position=position,
            )
        )
        for head in range(cfg.n_heads):
            sites.append(
                ToySite(
                    site_id=f"L{layer}_H{head}",
                    hook_name=f"blocks.{layer}.attn.hook_z",
                    component="head",
                    layer=layer,
                    position=position,
                    head=head,
                )
            )
        sites.append(
            ToySite(
                site_id=f"L{layer}_resid_mid",
                hook_name=f"blocks.{layer}.hook_resid_mid",
                component="resid_mid",
                layer=layer,
                position=position,
            )
        )
        sites.append(
            ToySite(
                site_id=f"L{layer}_mlp_post",
                hook_name=f"blocks.{layer}.mlp.hook_post",
                component="mlp_post",
                layer=layer,
                position=position,
            )
        )
        sites.append(
            ToySite(
                site_id=f"L{layer}_resid_post",
                hook_name=f"blocks.{layer}.hook_resid_post",
                component="resid_post",
                layer=layer,
                position=position,
            )
        )
    return sites


def extract_site(cache, site: ToySite):
    acts = cache[site.hook_name]
    if site.head is not None:
        return acts[:, site.position, site.head, :]
    return acts[:, site.position, :]


def cache_site_activations(model, tokens, sites: list[ToySite], batch_size: int = 2048) -> dict[str, Any]:
    """Run the model in batches and return {site_id: [n, d]} activations."""
    import torch

    names = sorted({site.hook_name for site in sites})
    chunks: dict[str, list] = {site.site_id: [] for site in sites}
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size]
            _, cache = model.run_with_cache(batch, names_filter=lambda n: n in names)
            for site in sites:
                chunks[site.site_id].append(extract_site(cache, site).detach().cpu())
    return {site_id: torch.cat(parts, dim=0) for site_id, parts in chunks.items()}


# ----------------------------------------------------------------------------------
# Ridge fits: latent-state germs P^y_v and transports rho^y
# ----------------------------------------------------------------------------------


def fit_ridge(X, Y, ridge: float = 1.0):
    """Closed-form ridge solution W with Y ~= X @ W, plus a bias column."""
    import torch

    X = X.float()
    Y = Y.float()
    ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
    Xb = torch.cat([X, ones], dim=1)
    d = Xb.shape[1]
    eye = torch.eye(d, dtype=Xb.dtype, device=Xb.device)
    eye[-1, -1] = 0.0  # do not penalize the bias
    gram = Xb.T @ Xb + ridge * eye
    return torch.linalg.solve(gram, Xb.T @ Y)


def apply_ridge(W, X):
    import torch

    X = X.float()
    ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
    return torch.cat([X, ones], dim=1) @ W


def r_squared(pred, target) -> float:
    target = target.float()
    pred = pred.float()
    ss_res = (target - pred).pow(2).sum()
    ss_tot = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
    if float(ss_tot) == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def fit_latent_germs(
    train_acts: dict[str, Any],
    train_latent,
    sites: list[ToySite],
    ridge: float = 1.0,
) -> dict[str, Any]:
    """Fit P^y_v mapping site activations to the known latent state, on train examples only."""
    return {site.site_id: fit_ridge(train_acts[site.site_id], train_latent, ridge) for site in sites}


def germ_values(acts: dict[str, Any], germs: dict[str, Any], sites: list[ToySite]) -> dict[str, Any]:
    return {site.site_id: apply_ridge(germs[site.site_id], acts[site.site_id]) for site in sites}


def chain_edges(sites: list[ToySite]) -> list[tuple[str, str]]:
    """Consecutive sites in the ordered chain, skipping operand-position embed sites into resid."""
    ids = [site.site_id for site in sites]
    return list(zip(ids[:-1], ids[1:]))


def fit_transports(
    germ_train: dict[str, Any],
    edges: list[tuple[str, str]],
    ridge: float = 1.0,
) -> dict[tuple[str, str], Any]:
    """Fit rho^y_{u->v} on clean training examples only, as required by the framework."""
    return {(u, v): fit_ridge(germ_train[u], germ_train[v], ridge) for u, v in edges}


def identity_transports(edges: list[tuple[str, str]], dim: int) -> dict[tuple[str, str], Any]:
    """Identity-transport ablation: assumes evidence should be literally equal across sites."""
    import torch

    W = torch.cat([torch.eye(dim), torch.zeros(1, dim)], dim=0)
    return {edge: W.clone() for edge in edges}


def random_transports(
    edges: list[tuple[str, str]], dim: int, seed: int = 0, scale: float = 1.0
) -> dict[tuple[str, str], Any]:
    """Random-transport ablation: matched shape, no learned structure."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    out = {}
    for i, edge in enumerate(edges):
        mat = torch.randn(dim, dim, generator=generator) * (scale / math.sqrt(dim))
        out[edge] = torch.cat([mat, torch.zeros(1, dim)], dim=0)
    return out


def latent_obstruction(
    germ_vals: dict[str, Any],
    transports: dict[tuple[str, str], Any],
    edges: list[tuple[str, str]],
    weights: dict[tuple[str, str], float] | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """O_y(x) = sum_e w_e || rho_{u->v} s_y(u) - s_y(v) ||^2, plus its normalized form."""
    import torch

    raw = None
    norm_sq = None
    edge_terms: dict[str, Any] = {}
    for u, v in edges:
        w = 1.0 if weights is None else weights.get((u, v), 1.0)
        residual = apply_ridge(transports[(u, v)], germ_vals[u]) - germ_vals[v].float()
        term = w * residual.pow(2).sum(dim=-1)
        edge_terms[f"{u}->{v}"] = term
        raw = term if raw is None else raw + term
    for site_id, vals in germ_vals.items():
        contrib = vals.float().pow(2).sum(dim=-1)
        norm_sq = contrib if norm_sq is None else norm_sq + contrib
    assert raw is not None and norm_sq is not None
    return {"raw": raw, "normalized": raw / (norm_sq + eps), "edge_terms": edge_terms}


# ----------------------------------------------------------------------------------
# Candidate contrast and margins
# ----------------------------------------------------------------------------------


def candidate_margin(logits, correct, incorrect):
    """m_c = z_correct - z_incorrect at the '=' position."""
    import torch

    idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[idx, correct] - logits[idx, incorrect]


def hardest_incorrect(logits, labels):
    """The highest-scoring wrong answer, giving the tightest candidate contrast."""
    import torch

    masked = logits.clone()
    idx = torch.arange(logits.shape[0], device=logits.device)
    masked[idx, labels] = float("-inf")
    return masked.argmax(dim=-1)


# ----------------------------------------------------------------------------------
# Subspaces for causal intervention
# ----------------------------------------------------------------------------------


def germ_subspace(germ_W):
    """Orthonormal basis for the column space of an already-fit germ map P^y_v.

    Reusing the fitted germ guarantees the ablated subspace is exactly the one whose germ
    drives the obstruction and the repair edit, and keeps the fit on training data only.
    Projecting this basis out of an activation drives the germ's linear part to zero.
    """
    import torch

    q, _ = torch.linalg.qr(germ_W[:-1].float())  # drop the bias row
    return q


def cyclic_subspace(acts, tokens, p: int, freqs: list[int], ridge: float = 1.0):
    """Cyclic subspace fit directly from activations. Pass training activations only."""
    latent = latent_targets(tokens, p, freqs).to(acts.device)
    return germ_subspace(fit_ridge(acts, latent, ridge=ridge))


def pca_subspace(acts, n_dims: int):
    """Top-variance directions, the generic-PCA control baseline."""
    import torch

    centered = acts.float() - acts.float().mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:n_dims].T


def random_subspace(d_model: int, n_dims: int, seed: int, device: str = "cpu"):
    """Random orthonormal subspace of matched dimension, the null control."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(d_model, n_dims, generator=generator)
    q, _ = torch.linalg.qr(mat)
    return q.to(device)


def spectral_subspace(model, layer: int, n_dims: int):
    """Top singular directions of W_out, the spectral-analysis control baseline."""
    import torch

    weight = model.blocks[layer].mlp.W_out.detach().float()
    _, _, vh = torch.linalg.svd(weight, full_matrices=False)
    return vh[:n_dims].T.cpu()


def project_out(vectors, basis):
    """Remove the component of `vectors` inside the span of orthonormal `basis`."""
    basis = basis.to(device=vectors.device, dtype=vectors.float().dtype)
    coeffs = vectors.float() @ basis
    return vectors.float() - coeffs @ basis.T


def match_norm_in_subspace(delta, basis):
    """Project `delta` into `basis` and rescale to the original norm.

    Gives a control edit of the same magnitude in a different subspace, so a repair effect
    cannot be explained by edit size alone.
    """
    import torch

    delta = delta.float()
    basis = basis.to(device=delta.device, dtype=delta.dtype)
    coeffs = delta @ basis
    projected = coeffs @ basis.T
    proj_norm = projected.norm(dim=-1, keepdim=True)
    target_norm = delta.norm(dim=-1, keepdim=True)
    scale = torch.where(proj_norm > 1e-8, target_norm / (proj_norm + 1e-12), torch.zeros_like(proj_norm))
    return projected * scale


def phase_rotate_delta(acts, germ_W, angle_rad: float, n_freqs: int | None = None):
    """Norm-preserving rotation of the cyclic (cos, sin) pairs in germ space.

    This is the phase-disruption control: the germ vector keeps its Euclidean norm, but
    each frequency's phase is rotated, so the decoded modular sum is wrong. A repair that
    is specific to modular structure should beat this control; a repair that only needs
    activation energy should not.
    """
    import math
    import torch

    current = apply_ridge(germ_W, acts)
    dim = current.shape[-1]
    if n_freqs is None:
        n_freqs = dim // 2
    n_used = min(2 * n_freqs, dim - (dim % 2))
    rotated = current.clone()
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    for k in range(0, n_used, 2):
        cos_k = current[:, k]
        sin_k = current[:, k + 1]
        rotated[:, k] = c * cos_k - s * sin_k
        rotated[:, k + 1] = s * cos_k + c * sin_k
    return germ_target_delta(acts, germ_W, rotated)


def germ_target_delta(acts, germ_W, target_latent):
    """Minimum-norm activation edit that moves the germ to `target_latent`.

    The germ is s = [h, 1] @ germ_W, so the edit solves (h + delta) @ W_lin = target.
    """
    import torch

    W_lin = germ_W[:-1].float()
    current = apply_ridge(germ_W, acts)
    residual = target_latent.float().to(current.device) - current
    return residual @ torch.linalg.pinv(W_lin)


# ----------------------------------------------------------------------------------
# Intervention hooks
# ----------------------------------------------------------------------------------


def site_edit_hook(site: ToySite, delta):
    """Additive edit h_u -> h_u + delta at a single site."""

    def hook(acts, hook):  # noqa: ARG001 - TransformerLens hook signature
        if site.head is not None:
            acts[:, site.position, site.head, :] += delta.to(acts.dtype).to(acts.device)
        else:
            acts[:, site.position, :] += delta.to(acts.dtype).to(acts.device)
        return acts

    return hook


def site_ablate_hook(site: ToySite, basis):
    """Project a subspace out of a site's activation, the necessity test."""

    def hook(acts, hook):  # noqa: ARG001 - TransformerLens hook signature
        b = basis.to(acts.device).float()
        if site.head is not None:
            current = acts[:, site.position, site.head, :].float()
            acts[:, site.position, site.head, :] = (current - (current @ b) @ b.T).to(acts.dtype)
        else:
            current = acts[:, site.position, :].float()
            acts[:, site.position, :] = (current - (current @ b) @ b.T).to(acts.dtype)
        return acts

    return hook


def kl_divergence(logits_before, logits_after):
    """KL(p_before || p_after) over the output distribution, the collateral-damage cost."""
    import torch.nn.functional as F

    log_p = F.log_softmax(logits_before.float(), dim=-1)
    log_q = F.log_softmax(logits_after.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


# ----------------------------------------------------------------------------------
# Checkpoint I/O
# ----------------------------------------------------------------------------------


@dataclass
class CheckpointRecord:
    step: int
    path: str
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    extras: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {
            "step": self.step,
            "path": self.path,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
        }
        row.update(self.extras)
        return row


def checkpoint_steps(total_steps: int, n_checkpoints: int = 40) -> list[int]:
    """Log-spaced early steps plus linear coverage, so grokking is not missed."""
    if total_steps <= 0:
        return [0]
    steps = {0, total_steps - 1}
    n_log = max(1, n_checkpoints // 2)
    for i in range(n_log):
        frac = (i + 1) / n_log
        steps.add(min(total_steps - 1, int((total_steps ** frac) - 1)))
    n_lin = n_checkpoints - n_log
    for i in range(max(1, n_lin)):
        steps.add(min(total_steps - 1, int(i * total_steps / max(1, n_lin))))
    return sorted(s for s in steps if s >= 0)


def save_checkpoint(model, path: str | Path, step: int, cfg: ModularConfig) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "state_dict": model.state_dict(), "config": cfg.to_dict()}, path)


def load_checkpoint(path: str | Path, device: str | None = None) -> tuple[Any, ModularConfig, int]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModularConfig(**blob["config"])
    model = build_model(cfg, device=device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg, int(blob["step"])
