"""Independent Othello simulator and 0A shared utilities.

Tokenization and implicit-pass game generation match Li et al. / Nanda
Othello-GPT (vocab 61, token 0 = pass, center squares omitted). The
simulator itself does not depend on TransformerLens.

Pass convention (training alignment):
    Synthetic Othello-GPT games omit pass tokens. When the side to move has
    no legal placement, Li's generator returns the opponent's legal squares
    and the next recorded move is played by that opponent. Player identity
    is therefore not odd/even move parity. This module tracks side-to-move
    explicitly and exposes both standard-pass and Li-compatible play.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, -1
BOARD_N = 8
N_SQUARES = 64
PASS_SQUARE = -100
PASS_TOKEN = 0
VOCAB_SIZE = 61
N_CTX = 59
CENTER_SQUARES = (27, 28, 35, 36)  # d4, d5, e4, e5
DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
ROWS = "abcdefgh"
COLS = "12345678"

# Li / Nanda token maps: token 0 is pass; tokens 1..60 skip the four center squares.
ITOS: dict[int, int] = {0: PASS_SQUARE}
STOI: dict[int, int] = {PASS_SQUARE: PASS_TOKEN, -1: PASS_TOKEN}
_tok = 1
for _sq in range(N_SQUARES):
    if _sq in CENTER_SQUARES:
        continue
    ITOS[_tok] = _sq
    STOI[_sq] = _tok
    _tok += 1

# Nanda / TransformerLens synthetic-checkpoint smoke test (argmax next-move tokens).
NANDA_SAMPLE_INPUT = [
    20, 19, 18, 10, 2, 1, 27, 3, 41, 42, 34, 12, 4, 40, 11, 29, 43, 13, 48, 56,
    33, 39, 22, 44, 24, 5, 46, 6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37, 9,
    25, 38, 23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7,
]
NANDA_SAMPLE_OUTPUT = [
    21, 41, 40, 34, 40, 41, 3, 11, 21, 43, 40, 21, 28, 50, 33, 50, 33, 5, 33, 5,
    52, 46, 14, 46, 14, 47, 38, 57, 36, 50, 38, 15, 28, 26, 28, 59, 50, 28, 14,
    28, 28, 28, 28, 45, 28, 35, 15, 14, 30, 59, 49, 59, 15, 15, 14, 15, 8, 7, 8,
]

MODEL_SPEC = {
    "source_repo": "https://github.com/likenneth/othello_world",
    "paper": "https://arxiv.org/abs/2210.13382",
    "nanda_writeup": "https://www.neelnanda.io/mechanistic-interpretability/othello",
    "hf_repo": "NeelNanda/Othello-GPT-Transformer-Lens",
    "checkpoint_file": "synthetic_model.pth",
    "architecture": "8-layer GPT, TransformerLens HookedTransformer",
    "n_layers": 8,
    "n_heads": 8,
    "d_model": 512,
    "d_head": 64,
    "d_mlp": 2048,
    "d_vocab": VOCAB_SIZE,
    "n_ctx": N_CTX,
    "act_fn": "gelu",
    "normalization_type": "LNPre",
    "training_objective": "next-move prediction on synthetic legal Othello games",
    "training_distribution": "random legal self-play (Li et al. synthetic)",
    "tokenization": "token 0=pass; tokens 1-60 = playable squares; center D4/D5/E4/E5 omitted",
    "pass_convention": "synthetic games omit pass tokens; implicit pass by opponent-move continuation",
    "license_note": "Research checkpoints redistributed by Neel Nanda from Li et al. ICLR 2023.",
}

SITE_TEMPLATES = (
    ("resid_pre", "blocks.{L}.hook_resid_pre"),
    ("attn_out", "blocks.{L}.hook_attn_out"),
    ("resid_mid", "blocks.{L}.hook_resid_mid"),
    ("mlp_out", "blocks.{L}.hook_mlp_out"),
    ("resid_post", "blocks.{L}.hook_resid_post"),
)


def square_name(sq: int) -> str:
    if sq == PASS_SQUARE:
        return "pass"
    return f"{ROWS[sq // 8]}{COLS[sq % 8]}"


def parse_square(name: str) -> int:
    if name.lower() == "pass":
        return PASS_SQUARE
    s = name.strip().lower()
    if len(s) != 2 or s[0] not in ROWS or s[1] not in COLS:
        raise ValueError(f"Invalid square {name!r}")
    return ROWS.index(s[0]) * 8 + COLS.index(s[1])


def token_to_square(tok: int) -> int:
    return ITOS[int(tok)]


def square_to_token(sq: int) -> int:
    return STOI[int(sq)]


def moves_to_tokens(squares: Iterable[int]) -> list[int]:
    return [square_to_token(s) for s in squares]


def tokens_to_moves(tokens: Iterable[int]) -> list[int]:
    return [token_to_square(t) for t in tokens]


def initial_board() -> np.ndarray:
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3] = WHITE  # d4
    board[3, 4] = BLACK  # d5
    board[4, 3] = BLACK  # e4
    board[4, 4] = WHITE  # e5
    return board


def opponent(player: int) -> int:
    return -player


def _flips(board: np.ndarray, square: int, player: int) -> list[tuple[int, int]]:
    r, c = divmod(int(square), 8)
    if board[r, c] != EMPTY:
        return []
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        buf: list[tuple[int, int]] = []
        cr, cc = r + dr, c + dc
        while 0 <= cr < 8 and 0 <= cc < 8:
            cell = int(board[cr, cc])
            if cell == EMPTY:
                break
            if cell == player:
                out.extend(buf)
                break
            buf.append((cr, cc))
            cr += dr
            cc += dc
    return out


def legal_squares(board: np.ndarray, player: int) -> list[int]:
    return [s for s in range(N_SQUARES) if _flips(board, s, player)]


def apply_square(board: np.ndarray, square: int, player: int) -> np.ndarray:
    flips = _flips(board, square, player)
    if not flips:
        raise ValueError(f"Illegal placement {square_name(square)} for player {player}")
    out = board.copy()
    r, c = divmod(int(square), 8)
    out[r, c] = player
    for fr, fc in flips:
        out[fr, fc] = player
    return out


def occupancy(board: np.ndarray) -> int:
    return int((board != EMPTY).sum())


def board_hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())


def to_absolute_labels(board: np.ndarray) -> np.ndarray:
    """Per-square labels: 0 empty, 1 black, 2 white."""
    lab = np.zeros(64, dtype=np.int64)
    flat = board.reshape(-1)
    lab[flat == BLACK] = 1
    lab[flat == WHITE] = 2
    return lab


def to_relative_labels(board: np.ndarray, player: int) -> np.ndarray:
    """Per-square labels: 0 empty, 1 mine, 2 theirs."""
    lab = np.zeros(64, dtype=np.int64)
    flat = board.reshape(-1)
    lab[flat == player] = 1
    lab[flat == opponent(player)] = 2
    return lab


def relative_to_absolute(rel: np.ndarray, player: int) -> np.ndarray:
    abs_lab = rel.copy()
    mine = rel == 1
    theirs = rel == 2
    abs_lab[mine] = 1 if player == BLACK else 2
    abs_lab[theirs] = 2 if player == BLACK else 1
    return abs_lab


def labels_to_board(lab: np.ndarray) -> np.ndarray:
    board = np.zeros((8, 8), dtype=np.int8)
    flat = lab.reshape(8, 8)
    board[flat == 1] = BLACK
    board[flat == 2] = WHITE
    return board


SYMMETRY_NAMES = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "flip_d", "flip_ad")


def transform_square(sq: int, name: str) -> int:
    if sq == PASS_SQUARE:
        return PASS_SQUARE
    r, c = divmod(int(sq), 8)
    if name == "id":
        nr, nc = r, c
    elif name == "rot90":
        nr, nc = c, 7 - r
    elif name == "rot180":
        nr, nc = 7 - r, 7 - c
    elif name == "rot270":
        nr, nc = 7 - c, r
    elif name == "flip_h":
        nr, nc = r, 7 - c
    elif name == "flip_v":
        nr, nc = 7 - r, c
    elif name == "flip_d":
        nr, nc = c, r
    elif name == "flip_ad":
        nr, nc = 7 - c, 7 - r
    else:
        raise ValueError(name)
    return nr * 8 + nc


def transform_board(board: np.ndarray, name: str) -> np.ndarray:
    out = np.zeros_like(board)
    for sq in range(64):
        t = transform_square(sq, name)
        out[divmod(t, 8)] = board[divmod(sq, 8)]
    return out


@dataclass
class BoardState:
    """Standard Othello state: board, side to move, history of placements."""

    board: np.ndarray = field(default_factory=initial_board)
    player: int = BLACK
    history: list[int] = field(default_factory=list)
    passed_last: bool = False

    def copy(self) -> "BoardState":
        return BoardState(self.board.copy(), self.player, list(self.history), self.passed_last)

    def legal_moves(self, player: int | None = None) -> list[int]:
        return legal_squares(self.board, self.player if player is None else player)

    def must_pass(self) -> bool:
        return not self.legal_moves() and bool(self.legal_moves(opponent(self.player)))

    def terminal(self) -> bool:
        return not self.legal_moves() and not self.legal_moves(opponent(self.player))

    def play(self, square: int) -> "BoardState":
        if square == PASS_SQUARE:
            if self.legal_moves():
                raise ValueError("Pass is illegal while placements exist.")
            nxt = self.copy()
            nxt.player = opponent(self.player)
            nxt.passed_last = True
            nxt.history.append(PASS_SQUARE)
            return nxt
        nxt = self.copy()
        nxt.board = apply_square(self.board, square, self.player)
        nxt.player = opponent(self.player)
        nxt.passed_last = False
        nxt.history.append(int(square))
        return nxt

    def play_li(self, square: int) -> "BoardState":
        """Apply a Li-generator move: implicit pass if current side has no placement."""
        legal = self.legal_moves()
        if square in legal:
            return self.play(square)
        if not legal and square in self.legal_moves(opponent(self.player)):
            return self.play(PASS_SQUARE).play(square)
        raise ValueError(f"Illegal Li-compatible move {square_name(square)}")

    def apply_moves(self, squares: Iterable[int], li_compatible: bool = True) -> "BoardState":
        state = self
        for sq in squares:
            state = state.play_li(int(sq)) if li_compatible else state.play(int(sq))
        return state


def generate_game(rng: np.random.Generator, li_compatible: bool = True, max_moves: int = 60) -> list[int]:
    """Random legal self-play. Li-compatible games record placements only (no pass tokens)."""
    state = BoardState()
    recorded: list[int] = []
    for _ in range(max_moves + 8):
        if state.terminal() or len(recorded) >= max_moves:
            break
        if state.must_pass():
            if li_compatible:
                state = state.play(PASS_SQUARE)
                continue
            recorded.append(PASS_SQUARE)
            state = state.play(PASS_SQUARE)
            continue
        legal = state.legal_moves()
        move = int(legal[int(rng.integers(0, len(legal)))])
        recorded.append(move)
        state = state.play(move)
    return recorded


def replay_states(squares: list[int], li_compatible: bool = True) -> list[BoardState]:
    """States BEFORE each recorded placement, plus the terminal/final state."""
    state = BoardState()
    out = [state]
    for sq in squares:
        state = state.play_li(sq) if li_compatible else state.play(sq)
        out.append(state)
    return out


@dataclass
class PrefixRecord:
    """One model-aligned prefix.

    tokens[:t] have been consumed. Residual at position t-1 (0-index) is the
    next-move prediction site. Board is the position BEFORE the next move.
    """

    game_id: str
    split: str
    t: int
    squares: list[int]
    tokens: list[int]
    board: np.ndarray
    player: int
    raw_player: int
    legal: list[int]
    next_square: int | None
    next_token: int | None
    terminal: bool
    must_pass: bool
    occupied: int
    after_board: np.ndarray | None
    stage: str

    def board_before(self) -> np.ndarray:
        return self.board

    def side_to_move(self) -> int:
        return self.player


def game_stage(occupied: int) -> str:
    if occupied <= 16:
        return "opening"
    if occupied <= 48:
        return "middlegame"
    return "endgame"


def prefixes_from_game(game_id: str, squares: list[int], split: str, li_compatible: bool = True) -> list[PrefixRecord]:
    tokens = moves_to_tokens(squares)
    states = replay_states(squares, li_compatible=li_compatible)
    rows: list[PrefixRecord] = []
    for t in range(1, min(len(tokens), N_CTX) + 1):
        before = states[t]
        nxt = squares[t] if t < len(squares) else None
        after = states[t + 1].board if t < len(squares) else None
        raw_player = before.player
        if before.must_pass():
            token_player = opponent(raw_player)
            legal = before.legal_moves(token_player)
        else:
            token_player = raw_player
            legal = before.legal_moves()
        rows.append(
            PrefixRecord(
                game_id=game_id,
                split=split,
                t=t,
                squares=list(squares[:t]),
                tokens=list(tokens[:t]),
                board=before.board.copy(),
                player=token_player,
                raw_player=raw_player,
                legal=legal,
                next_square=nxt,
                next_token=square_to_token(nxt) if nxt is not None and nxt in STOI else (PASS_TOKEN if nxt == PASS_SQUARE else None),
                terminal=before.terminal(),
                must_pass=before.must_pass(),
                occupied=occupancy(before.board),
                after_board=None if after is None else after.copy(),
                stage=game_stage(occupancy(before.board)),
            )
        )
    return rows


def legal_mask_from_board(board: np.ndarray, player: int) -> np.ndarray:
    mask = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for sq in legal_squares(board, player):
        mask[square_to_token(sq)] = 1.0
    if not legal_squares(board, player) and legal_squares(board, opponent(player)):
        mask[PASS_TOKEN] = 1.0
    return mask


def legal_mask_tokens(legal: list[int], must_pass: bool) -> np.ndarray:
    mask = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for sq in legal:
        if sq in STOI:
            mask[square_to_token(sq)] = 1.0
    if must_pass or (not legal):
        mask[PASS_TOKEN] = 1.0
    return mask


def pad_tokens(tokens: list[int], length: int = N_CTX) -> np.ndarray:
    out = np.zeros(length, dtype=np.int64)
    n = min(len(tokens), length)
    out[:n] = tokens[:n]
    return out


def board_key(board: np.ndarray, player: int) -> tuple:
    return (player, tuple(int(v) for v in board.reshape(-1)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def default_config() -> dict:
    return {
        "seed": 0,
        "val_seed": 1,
        "test_seed": 2,
        "n_games_train": 2000,
        "n_games_val": 400,
        "n_games_test": 400,
        "max_moves": 60,
        "positions_per_game": 6,
        "min_prefix": 5,
        "max_prefix_from_end": 5,
        "li_compatible": True,
        "results_dir": "artifacts/results/toy_othello",
        "figures_dir": "artifacts/figures/toy_othello",
        "checkpoints_dir": "artifacts/checkpoints/toy_othello",
        "data_dir": "artifacts/toy_othello/data",
        "hf_repo": MODEL_SPEC["hf_repo"],
        "checkpoint_file": MODEL_SPEC["checkpoint_file"],
        "batch_size": 64,
        "cache_batch": 32,
        "probe_layers": [0, 2, 4, 5, 6, 7],
        "probe_sites": ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"],
        "primary_site": "blocks.6.hook_resid_post",
        "linear_c": 1.0,
        "mlp_hidden": 32,
        "mlp_alpha": 1e-3,
        "n_boot": 200,
        "n_pairs": 80,
        "edit_norm_budget": 8.0,
        "edit_steps": 40,
        "edit_lr": 0.4,
        "legality_gate_metric": "mean_legal_mass",
        "legality_gate_min": None,
        "device": None,
    }


def load_config(path: str | Path | None = None) -> dict:
    cfg = default_config()
    if path is None:
        candidate = Path("configs/toy_othello.json")
        path = candidate if candidate.exists() else None
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg.update(raw)
    return cfg


def save_config(path: str | Path, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def hook_name(site: str, layer: int) -> str:
    for key, tmpl in SITE_TEMPLATES:
        if key == site:
            return tmpl.format(L=layer)
    raise KeyError(site)


def all_hooks(layers: Iterable[int], sites: Iterable[str]) -> list[str]:
    return [hook_name(s, L) for L in layers for s in sites]


def load_othello_gpt(device: str | None = None, checkpoint_path: str | Path | None = None):
    """Load the frozen Nanda-converted synthetic Othello-GPT."""
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.utils import download_file_from_hf

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = HookedTransformerConfig(
        n_layers=int(MODEL_SPEC["n_layers"]),
        d_model=int(MODEL_SPEC["d_model"]),
        d_head=int(MODEL_SPEC["d_head"]),
        n_heads=int(MODEL_SPEC["n_heads"]),
        d_mlp=int(MODEL_SPEC["d_mlp"]),
        d_vocab=int(MODEL_SPEC["d_vocab"]),
        n_ctx=int(MODEL_SPEC["n_ctx"]),
        act_fn=str(MODEL_SPEC["act_fn"]),
        normalization_type=str(MODEL_SPEC["normalization_type"]),
        tokenizer_name=None,
    )
    model = HookedTransformer(cfg)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        sd = torch.load(checkpoint_path, map_location="cpu")
    else:
        sd = download_file_from_hf(MODEL_SPEC["hf_repo"], MODEL_SPEC["checkpoint_file"])
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def verify_nanda_checkpoint(model) -> dict:
    import torch

    device = next(model.parameters()).device
    x = torch.tensor([NANDA_SAMPLE_INPUT], device=device)
    with torch.no_grad():
        pred = model(x).argmax(dim=-1).cpu().tolist()[0]
    ok = pred == NANDA_SAMPLE_OUTPUT
    return {
        "nanda_argmax_match": bool(ok),
        "n_mismatch": int(sum(int(a != b) for a, b in zip(pred, NANDA_SAMPLE_OUTPUT))),
        "pred": pred,
    }


def random_untrained_othello(seed: int = 0, device: str | None = None):
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=int(MODEL_SPEC["n_layers"]),
        d_model=int(MODEL_SPEC["d_model"]),
        d_head=int(MODEL_SPEC["d_head"]),
        n_heads=int(MODEL_SPEC["n_heads"]),
        d_mlp=int(MODEL_SPEC["d_mlp"]),
        d_vocab=int(MODEL_SPEC["d_vocab"]),
        n_ctx=int(MODEL_SPEC["n_ctx"]),
        act_fn=str(MODEL_SPEC["act_fn"]),
        normalization_type=str(MODEL_SPEC["normalization_type"]),
        tokenizer_name=None,
        seed=seed,
    )
    model = HookedTransformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(generator=g)
    model.to(device)
    model.eval()
    return model


def pad_batch(token_lists: list[list[int]], device=None):
    import torch

    n = len(token_lists)
    lengths = [len(t) for t in token_lists]
    tmax = max(lengths)
    toks = torch.zeros(n, tmax, dtype=torch.long)
    for i, seq in enumerate(token_lists):
        toks[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    if device is not None:
        toks = toks.to(device)
    return toks, lengths


def run_model_logits(model, token_lists: list[list[int]], batch_size: int = 64):
    import torch

    outs = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(token_lists), batch_size):
            chunk = token_lists[start : start + batch_size]
            toks, lengths = pad_batch(chunk, device)
            logits = model(toks)
            rows = []
            for i, t in enumerate(lengths):
                rows.append(logits[i, t - 1, :].float().cpu())
            outs.append(torch.stack(rows, dim=0))
    return torch.cat(outs, dim=0)


def cache_site(model, token_lists: list[list[int]], hook: str, batch_size: int = 32):
    import torch

    outs = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(token_lists), batch_size):
            chunk = token_lists[start : start + batch_size]
            toks, lengths = pad_batch(chunk, device)
            _, cache = model.run_with_cache(toks, names_filter=lambda n, h=hook: n == h)
            h = cache[hook]
            rows = [h[i, t - 1, :].float().cpu() for i, t in enumerate(lengths)]
            outs.append(torch.stack(rows, dim=0))
    return torch.cat(outs, dim=0)


def patch_site(model, token_lists: list[list[int]], hook: str, values, batch_size: int = 32):
    """Replace `hook` at the last real token of each sequence with `values`."""
    import torch

    outs = []
    device = next(model.parameters()).device
    values = values.to(device)

    def make_hook(start, end, lengths):
        chunk = values[start:end]

        def fn(act, hook):
            out = act.clone()
            for i, t in enumerate(lengths):
                out[i, t - 1, :] = chunk[i].to(dtype=out.dtype)
            return out

        return fn

    with torch.no_grad():
        for start in range(0, len(token_lists), batch_size):
            chunk = token_lists[start : start + batch_size]
            toks, lengths = pad_batch(chunk, device)
            end = start + len(chunk)
            logits = model.run_with_hooks(toks, fwd_hooks=[(hook, make_hook(start, end, lengths))])
            rows = [logits[i, t - 1, :].float().cpu() for i, t in enumerate(lengths)]
            outs.append(torch.stack(rows, dim=0))
    return torch.cat(outs, dim=0)


def softmax_np(logits) -> np.ndarray:
    import torch

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def legal_metrics(logits, legal_masks: np.ndarray) -> dict:
    import torch

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits.float(), dim=-1)
    mask = torch.tensor(legal_masks, dtype=torch.float32)
    pred = probs.argmax(dim=-1)
    legal_idx = pred.unsqueeze(-1)
    top1 = mask.gather(1, legal_idx).squeeze(-1)
    legal_mass = (probs * mask).sum(dim=-1)
    illegal_mass = 1.0 - legal_mass
    legal_logit = (logits.float() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    illegal_logit = (logits.float() * (1.0 - mask)).sum(dim=-1) / (1.0 - mask).sum(dim=-1).clamp_min(1.0)
    return {
        "top1_legal": float(top1.mean()),
        "mean_legal_mass": float(legal_mass.mean()),
        "mean_illegal_mass": float(illegal_mass.mean()),
        "mean_legal_illegal_margin": float((legal_logit - illegal_logit).mean()),
        "n": int(logits.shape[0]),
    }


def bootstrap_by_game(values: np.ndarray, game_ids: list[str], n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    by: dict[str, list[float]] = {}
    for v, g in zip(values, game_ids):
        by.setdefault(g, []).append(float(v))
    games = list(by.keys())
    if not games:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        draw = rng.choice(games, size=len(games), replace=True)
        chunk = [float(np.mean(by[g])) for g in draw]
        means.append(float(np.mean(chunk)))
    arr = np.asarray(means)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def subsample_prefixes(rows: list[PrefixRecord], k: int, seed: int, min_prefix: int = 5, max_from_end: int = 5) -> list[PrefixRecord]:
    rng = np.random.default_rng(seed)
    by: dict[str, list[PrefixRecord]] = {}
    for r in rows:
        if r.t < min_prefix:
            continue
        if r.terminal or r.must_pass:
            continue
        if not r.legal:
            continue
        by.setdefault(r.game_id, []).append(r)
    out: list[PrefixRecord] = []
    for gid, items in by.items():
        items = [r for r in items if r.t <= max(len(r.tokens), 1)]
        if max_from_end > 0:
            tmax = max(r.t for r in items) if items else 0
            items = [r for r in items if r.t <= tmax - max_from_end or tmax <= max_from_end]
        if not items:
            continue
        take = min(k, len(items))
        idx = rng.choice(len(items), size=take, replace=False)
        out.extend(items[int(i)] for i in idx)
    return out
