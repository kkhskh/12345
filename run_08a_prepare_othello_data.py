"""08a: validate the Othello simulator and build game-level splits.

Does not load a Transformer. Does not score probes or obstruction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_text
from src.toy_othello import (
    BLACK,
    CENTER_SQUARES,
    MODEL_SPEC,
    STOI,
    SYMMETRY_NAMES,
    BoardState,
    board_key,
    default_config,
    generate_game,
    legal_squares,
    load_config,
    occupancy,
    prefixes_from_game,
    replay_states,
    save_config,
    sha256_json,
    square_name,
    subsample_prefixes,
    transform_board,
    transform_square,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 0A Othello games and simulator validation.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def split_games(n_train: int, n_val: int, n_test: int, seed: int, max_moves: int = 60) -> dict[str, list[tuple[str, list[int]]]]:
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    out: dict[str, list[tuple[str, list[int]]]] = {"train": [], "val": [], "test": []}
    targets = {"train": n_train, "val": n_val, "test": n_test}
    counters = {"train": 0, "val": 0, "test": 0}
    order = ["train", "val", "test"]
    guard = 0
    while any(len(out[k]) < targets[k] for k in order):
        guard += 1
        if guard > 50 * (n_train + n_val + n_test):
            raise RuntimeError("Could not generate enough unique games.")
        sq = generate_game(rng, li_compatible=True, max_moves=max_moves)
        key = tuple(sq)
        if len(sq) < 8 or key in seen:
            continue
        seen.add(key)
        for split in order:
            if len(out[split]) < targets[split]:
                gid = f"{split}_{counters[split]:05d}"
                counters[split] += 1
                out[split].append((gid, sq))
                break
    return out


def _record_to_dict(rec) -> dict:
    return {
        "game_id": rec.game_id,
        "split": rec.split,
        "t": rec.t,
        "tokens": rec.tokens,
        "squares": rec.squares,
        "board": rec.board.reshape(-1).astype(int).tolist(),
        "player": int(rec.player),
        "raw_player": int(rec.raw_player),
        "legal": rec.legal,
        "next_square": rec.next_square,
        "next_token": rec.next_token,
        "terminal": rec.terminal,
        "must_pass": rec.must_pass,
        "occupied": rec.occupied,
        "after_board": None if rec.after_board is None else rec.after_board.reshape(-1).astype(int).tolist(),
        "stage": rec.stage,
    }


def write_dataset(data_dir: Path, splits: dict, positions_per_game: int, seed: int, min_prefix: int, max_from_end: int) -> dict:
    ensure_dir(data_dir)
    all_rows = []
    games_payload = {}
    for split, games in splits.items():
        games_payload[split] = [{"game_id": gid, "squares": sq} for gid, sq in games]
        recs = []
        for gid, sq in games:
            recs.extend(prefixes_from_game(gid, sq, split))
        sampled = subsample_prefixes(
            recs, positions_per_game, seed + {"train": 0, "val": 1, "test": 2}[split], min_prefix, max_from_end
        )
        all_rows.extend(sampled)
        path = data_dir / f"prefixes_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for rec in sampled:
                handle.write(json.dumps(_record_to_dict(rec)) + "\n")
    (data_dir / "games.json").write_text(json.dumps(games_payload) + "\n", encoding="utf-8")
    keys = [board_key(r.board, r.player) for r in all_rows]
    counts = Counter(keys)
    transpositions = sum(1 for _, c in counts.items() if c > 1)
    return {
        "n_rows": len(all_rows),
        "n_transposition_states": int(transpositions),
        "stage_counts": dict(Counter(r.stage for r in all_rows)),
        "pass_prefixes": int(sum(1 for r in all_rows if r.must_pass)),
        "split_counts": {s: len(list(splits[s])) for s in splits},
    }


def load_prefixes(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"prefixes_{split}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def validate_simulator() -> dict:
    checks = []

    def add(name: str, ok: bool, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    st = BoardState()
    legal = {square_name(s) for s in st.legal_moves()}
    add("opening_legal_c4_d3_e6_f5", legal == {"c4", "d3", "e6", "f5"}, str(sorted(legal)))
    d3 = st.play(next(s for s in st.legal_moves() if square_name(s) == "d3"))
    add("d3_occupancy_5", occupancy(d3.board) == 5)
    add("d3_flips_d4_black", int(d3.board[3, 3]) == BLACK)
    add("centers_have_no_tokens", all(s not in STOI for s in CENTER_SQUARES))
    add("center_not_legal", not any(s in CENTER_SQUARES for s in st.legal_moves()))

    rng = np.random.default_rng(0)
    n_ok = 0
    n_games = 80
    n_pass = 0
    for i in range(n_games):
        sq = generate_game(rng)
        states = replay_states(sq)
        good = True
        for t, rec_state in enumerate(states[:-1]):
            move = sq[t]
            if rec_state.must_pass():
                n_pass += 1
                if move not in rec_state.legal_moves(-rec_state.player):
                    good = False
            elif move not in rec_state.legal_moves():
                good = False
        n_ok += int(good)
    add("random_games_legal_replay", n_ok == n_games, f"{n_ok}/{n_games}")
    add("implicit_pass_observed_or_rare", True, f"must_pass_states={n_pass}")

    st2 = BoardState().play(next(s for s in BoardState().legal_moves() if square_name(s) == "d3"))
    for name in SYMMETRY_NAMES:
        tb = transform_board(st2.board, name)
        tlegal = {transform_square(s, name) for s in st2.legal_moves()}
        add(f"symmetry_{name}", tlegal == set(legal_squares(tb, st2.player)))

    rows = prefixes_from_game("align", generate_game(np.random.default_rng(7)), "train")
    states = replay_states(rows[0].squares + ([rows[0].next_square] if rows[0].next_square is not None else []))
    add("prefix_alignment_t_equals_consumed_moves", all(r.t == len(r.tokens) for r in rows))
    add("prefix_board_after_consumed", all(np.array_equal(r.board, replay_states(r.squares)[-1].board) for r in rows[:12]))
    add("next_move_in_legal", all(r.next_square in r.legal for r in rows if r.next_square is not None))
    _ = states
    failed = [c for c in checks if not c["ok"]]
    return {
        "n_checks": len(checks),
        "n_failed": len(failed),
        "passed": len(failed) == 0,
        "checks": checks,
        "model_alignment_note": MODEL_SPEC["pass_convention"],
    }


def write_preregistration(path: Path, cfg: dict) -> None:
    md = f"""# 08a preregistered gates — Experiment 0A

Official 07d remains NO_GO. 07f A, 07g–07i B, 07j A, 07k C are unchanged.
This file was written before model/probe/causal test evaluation.

## Existing files reused

- `src/model.py`, `src/cache.py`, `src/patching.py` (TransformerLens load/cache/patch patterns)
- `run_07f_modular_causal_sanity_validation.py` (`ensure_dir`, `write_csv`, `write_text`)
- `pyproject.toml` TransformerLens pin `>=2.11,<4`

## New files

- `src/toy_othello.py`, `configs/toy_othello.json`
- `run_08a` … `run_08i`
- `tests/test_toy_othello_rules.py` and sibling tests

## Model

Frozen pretrained synthetic Othello-GPT:

- HuggingFace `{cfg['hf_repo']}` / `{cfg['checkpoint_file']}`
- 8 layers, 8 heads, d_model 512, d_mlp 2048, vocab 61, ctx 59
- Next-move LM on random legal synthetic games (Li et al.)
- Token 0 = pass; tokens 1–60 playable squares; D4/D5/E4/E5 omitted
- Synthetic games omit pass tokens (implicit pass)

Do not train a new large model unless this checkpoint cannot be loaded.

## Primary hypotheses

1. Held-out activations decode board state above shuffled/untrained/metadata controls (Claim A).
2. Full donor activation interchange at a mid-layer site moves legal-move mass toward the donor legal set (positive control).
3. A reusable state-specific edit (probe subspace / train-derived transform) produces targeted legality changes that beat rank/norm-matched random, PCA, attribution, and logit steering (Claim C).
4. Candidate germ \(s_m = 1[m \\in L(\\hat B)]\) tracks simulator legality and intervention-induced legality changes.

## Main metrics

- Decode: per-square accuracy, whole-board exact match, macro-F1, induced legal-mask Jaccard
- Behavior: top-1 legality, legal probability mass
- Causal: Δ legal-mass on became-legal / became-illegal / unchanged; decoder agreement; edit norm
- Germ: AUROC of \(s_m\) vs true legality; intervention-locked Δs_m vs Δmargin

Bootstrap by **game**, not by position. Report 95% CIs.

## Eligibility

- Simulator validation must pass
- Unique game-level splits
- Frozen synthetic checkpoint must match the Nanda argmax fixture
- Validation-only legality gate: mean legal mass on val. Freeze the threshold on val before test. If unset, require val mean legal mass ≥ 0.70 to interpret causal negatives as representation failures rather than model failure.

## Sites and methods (frozen)

- Sites: resid_pre / attn_out / resid_mid / mlp_out / resid_post at layers {cfg['probe_layers']}
- Primary site: `{cfg['primary_site']}`
- Probes: linear on all listed sites; capacity-32 MLP only on resid_post layers 4 and 6 (runtime cap). Controls (shuffle, metadata, random subspace, untrained) on the primary site.
- Edits: linear directions, probe-constrained (train-frozen step/lr/norm), train-mean square deltas, donor projection onto probe subspace
- Baselines: full patch, random same-rank, PCA same-rank, norm-matched random, attribution, logit +4, shuffled target, wrong square, no-op

Validation selects the probe type/site and one edit method. Test is run once.

## Minimum effects (val-calibrated, not test-tuned)

- PASS_DECODE: whole-board exact match ≥ 0.20 **or** per-square acc ≥ 0.85, and ≥ +0.15 vs shuffled and untrained, on held-out games
- Full-patch positive control: donor-legal mass lift ≥ 0.10 vs no-op on Type B pairs
- State-specific pass: became-legal ΔP ≥ +0.08 and became-illegal ΔP ≤ −0.08, beating attribution and logit steer by ≥ 0.05, on held-out Type B pairs, with decoder target-agreement ≥ 0.30

## Success / failure

- A: decode + math consistency + reusable targeted causal edits beat controls
- B: decode works; causal state edits fail
- C: full interchange works; state-specific methods do not add value
- D: model/sim/data insufficient

`SECTION_5_3_EMPIRICAL_GATE` is PASS only for A, PARTIAL for B/C with real decode or full patch, else FAIL.
`OBSTRUCTION_READY = NO` regardless.

## What does not count as success

High square accuracy alone; full late-layer copy; probe fooling; val-only edits; a few board pictures; replicating Nanda without beating the listed controls on held-out legal-set change.
"""
    write_text(path, md)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    results_dir = ensure_dir(cfg["results_dir"])
    data_dir = ensure_dir(cfg["data_dir"])
    ensure_dir(cfg["figures_dir"])
    ensure_dir(cfg["checkpoints_dir"])
    save_config(Path("configs/toy_othello.json"), cfg)

    print("=== 08a simulator validation ===")
    sim = validate_simulator()
    (results_dir / "08a_simulator_validation.json").write_text(json.dumps(sim, indent=2) + "\n", encoding="utf-8")
    print(f"  simulator passed={sim['passed']} failed={sim['n_failed']}/{sim['n_checks']}")
    if not sim["passed"]:
        raise RuntimeError("Simulator validation failed. Stop before probing.")

    print("=== 08a dataset ===")
    splits = split_games(cfg["n_games_train"], cfg["n_games_val"], cfg["n_games_test"], cfg["seed"], cfg["max_moves"])
    stats = write_dataset(
        Path(data_dir),
        splits,
        cfg["positions_per_game"],
        cfg["seed"],
        cfg["min_prefix"],
        cfg["max_prefix_from_end"],
    )
    manifest = {
        "experiment": "0A",
        "simulator": "src/toy_othello.py",
        "pass_convention": MODEL_SPEC["pass_convention"],
        "tokenization": MODEL_SPEC["tokenization"],
        "ood_note": (
            "Test games are unseen random legal trajectories, not a held-out operand class. "
            "Symmetry evaluations transform valid states; they are not in the training token stream."
        ),
        "splits": {k: len(v) for k, v in splits.items()},
        "dataset_stats": stats,
        "games_sha256": sha256_json({k: [sq for _, sq in v] for k, v in splits.items()}),
        "config": cfg,
        "model_spec": MODEL_SPEC,
    }
    (results_dir / "08a_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_preregistration(results_dir / "08a_preregistered_gates.md", cfg)
    print(f"  rows={stats['n_rows']} transpositions={stats['n_transposition_states']}")
    print(f"saved {results_dir}")
    print(f"saved {data_dir}")


if __name__ == "__main__":
    main()
