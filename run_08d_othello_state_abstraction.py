"""08d: mathematical consistency of the selected state abstraction. Not causal."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_runutil import add_labels, board_arr, induced_legal, load_split, tokens_of, write_json
from src.toy_othello import (
    cache_site,
    labels_to_board,
    load_config,
    load_othello_gpt,
    relative_to_absolute,
    to_absolute_labels,
    to_relative_labels,
    transform_board,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08d state-abstraction consistency.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _pred_abs(models, X, rows, target: str) -> np.ndarray:
    pred = np.stack([m.predict(X) for m in models], axis=1)
    if target == "rel":
        out = []
        for i, r in enumerate(rows):
            out.append(relative_to_absolute(pred[i], int(r["player"])))
        return np.stack(out, axis=0)
    return pred


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    results_dir = ensure_dir(cfg["results_dir"])
    fig_dir = ensure_dir(cfg["figures_dir"])
    probe_path = Path(cfg["checkpoints_dir"]) / "08c_selected_probe.pkl"
    if not probe_path.exists():
        write_json(
            results_dir / "08d_status.json",
            {
                "status": "BLOCKED_MISSING_ARTIFACT",
                "artifact": str(probe_path),
                "reason": "08c_selected_probe.pkl is required. No Gaussian or arbitrary probe substitute.",
            },
        )
        raise SystemExit(f"08d BLOCKED_MISSING_ARTIFACT: {probe_path}")
    with probe_path.open("rb") as handle:
        pack = pickle.load(handle)
    test = add_labels(load_split(Path(cfg["data_dir"]), "test"))
    try:
        model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=args.checkpoint)
    except Exception as exc:
        write_json(results_dir / "08d_status.json", {"status": "BLOCKED", "reason": str(exc)})
        print(f"BLOCKED: {exc}")
        return
    X = cache_site(model, tokens_of(test), pack["hook"], cfg["cache_batch"]).numpy()
    pred = _pred_abs(pack["models"], X, test, pack["target"])
    true = np.stack([r["abs_lab"] for r in test])

    from src.othello_r1_eval import dist_stats
    from src.toy_othello import SYMMETRY_NAMES, BoardState, labels_to_board, legal_squares, transform_board, transform_square

    rows = []
    n_games = len({r["game_id"] for r in test})
    # player-relative invertibility
    if pack["target"] == "rel":
        rel = np.stack([m.predict(X) for m in pack["models"]], axis=1)
        recon = np.stack([relative_to_absolute(rel[i], int(test[i]["player"])) for i in range(len(test))])
        rows.append(
            {
                "test": "player_relative_roundtrip",
                **dist_stats((recon == pred).mean(axis=1)),
                "n_games": n_games,
                "left": "relative_to_absolute(decoded_rel)",
                "right": "decoded absolute labels",
                "used_probe": 1,
                "used_model_activations": 1,
                "valid": 1,
            }
        )

    # Historical tautology kept only as an excluded row. Not evidence.
    rows.append(
        {
            "test": "legalizer_self_consistent",
            "mean": float("nan"),
            "n": 0,
            "n_games": 0,
            "valid": 0,
            "status": "NOT_APPLICABLE",
            "left": "removed",
            "right": "removed",
            "used_probe": 0,
            "used_model_activations": 0,
            "note": "Tautological induced_legal(pred) vs itself. Excluded from 08R1 evidence.",
        }
    )

    j2 = []
    exact = []
    for i, r in enumerate(test):
        a, b = set(induced_legal(pred[i], int(r["player"]))), set(r["legal"])
        j2.append(len(a & b) / len(a | b) if a | b else 1.0)
        exact.append(float((pred[i] == r["abs_lab"]).all()))
    rows.append(
        {
            "test": "decoded_vs_true_legal_jaccard",
            **dist_stats(j2),
            "n_games": n_games,
            "left": "induced_legal(decoded_board)",
            "right": "simulator legal set on true board",
            "used_probe": 1,
            "used_model_activations": 1,
            "valid": 1,
        }
    )
    rows.append(
        {
            "test": "whole_board_exact",
            **dist_stats(exact),
            "n_games": n_games,
            "left": "decoded absolute labels",
            "right": "true absolute labels",
            "used_probe": 1,
            "used_model_activations": 1,
            "valid": 1,
        }
    )

    by_game = defaultdict(list)
    for i, r in enumerate(test):
        by_game[r["game_id"]].append(i)
    trans = []
    for items in by_game.values():
        items = sorted(items, key=lambda i: test[i]["t"])
        for a, b in zip(items, items[1:]):
            if test[b]["t"] != test[a]["t"] + 1:
                continue
            if test[a]["after_board"] is None:
                continue
            dec_a = labels_to_board(pred[a])
            st = BoardState(dec_a, int(test[a]["player"]))
            move = test[a]["next_square"]
            if move is None or move not in st.legal_moves():
                continue
            stepped = st.play(move).board
            trans.append(float((to_absolute_labels(stepped) == pred[b]).mean()))
    rows.append(
        {
            "test": "transition_decoded_F_vs_next",
            **dist_stats(trans),
            "n_games": n_games,
            "left": "simulator.play(decoded_board_t, true_next_square)",
            "right": "independently decoded board at t+1",
            "used_probe": 1,
            "used_model_activations": 1,
            "valid": 1,
        }
    )

    keys = defaultdict(list)
    for i, r in enumerate(test):
        keys[(int(r["player"]), tuple(r["board"]))].append(i)
    same = []
    for idxs in keys.values():
        if len(idxs) < 2:
            continue
        same.append(float((pred[idxs[0]] == pred[idxs[1]]).mean()))
    rows.append(
        {
            "test": "same_state_decode_agreement",
            **dist_stats(same),
            "n_games": n_games,
            "left": "decoded labels at history A",
            "right": "decoded labels at history B, same (player, board)",
            "used_probe": 1,
            "used_model_activations": 1,
            "valid": 1,
        }
    )

    sym = []
    for i, r in enumerate(test[:200]):
        board = labels_to_board(pred[i])
        player = int(r["player"])
        base_legal = set(legal_squares(board, player))
        for name in SYMMETRY_NAMES:
            tb = transform_board(board, name)
            tlegal = {transform_square(s, name) for s in base_legal}
            sym.append(int(tlegal == set(legal_squares(tb, player))))
    rows.append(
        {
            "test": "decoded_board_symmetry_legal",
            **dist_stats(sym),
            "n_games": min(n_games, 200),
            "left": "transform(decoded legal set)",
            "right": "legal_squares(transform(decoded board))",
            "used_probe": 1,
            "used_model_activations": 1,
            "valid": 1,
            "note": "Property of the decoded board. Not a new model forward.",
        }
    )

    write_csv(results_dir / "08d_state_abstraction_consistency.csv", rows)
    write_json(
        results_dir / "08d_abstraction_spec.json",
        {
            "R_v": f"64 independent {pack['method']} classifiers on {pack['hook']}",
            "domain": "residual (or named site) at last prefix token",
            "codomain": "8x8 labels empty/black/white (via player-relative if selected)",
            "player_dependence": pack["target"] == "rel",
            "discarded": "move order, disc age, bluffing, value",
            "selected": {k: pack[k] for k in pack if k != "models"},
            "probe_path": str(probe_path),
            "used_probe": 1,
            "used_model_activations": 1,
            "legalizer_self_consistent_is_evidence": 0,
            "note": "Consistency is not causal participation. legalizer_self_consistent is excluded.",
        },
    )
    make_fig(fig_dir, rows)
    print("08d consistency rows", len(rows))


def make_fig(fig_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ys = [r.get("mean", r.get("mean_square_agree", float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(rows)), ys)
    ax.set_xticks(range(len(rows)), [r["test"][:22] for r in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("08d state abstraction consistency")
    fig.tight_layout()
    fig.savefig(fig_dir / "08d_state_abstraction_consistency.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
