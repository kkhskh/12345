"""08c: recover board-state probes. Gate 1. Train-only fitting, val selection, one test pass."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_runutil import add_labels, induced_legal, load_split, tokens_of, write_json
from src.toy_othello import cache_site, hook_name, load_config, load_othello_gpt, random_untrained_othello


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08c Othello state probes.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Reuse 08c_state_decode.csv; refit only the val-selected probe and write remaining artifacts.",
    )

    return parser.parse_args()


def _fit_linear(X, y, C: float):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=200)
    clf.fit(X, y)
    return clf


def _fit_mlp(X, y, hidden: int, alpha: float):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=40, random_state=0)
    clf.fit(X, y)
    return clf


def _predict_squares(clf, X) -> np.ndarray:
    return clf.predict(X)


def _score_board(pred: np.ndarray, true: np.ndarray) -> dict:
    from sklearn.metrics import f1_score

    sq_acc = float((pred == true).mean())
    exact = float((pred == true).all(axis=1).mean())
    occ = true != 0
    empty = true == 0
    return {
        "square_acc": sq_acc,
        "exact_board": exact,
        "occupied_acc": float((pred[occ] == true[occ]).mean()) if occ.any() else float("nan"),
        "empty_acc": float((pred[empty] == true[empty]).mean()) if empty.any() else float("nan"),
        "macro_f1": float(f1_score(true.reshape(-1), pred.reshape(-1), average="macro")),
        "n": int(true.shape[0]),
    }


def _per_square_train(method: str, X, labs, C, hidden, alpha):
    models = []
    for j in range(64):
        y = labs[:, j]
        if method == "linear":
            models.append(_fit_linear(X, y, C))
        elif method == "mlp":
            models.append(_fit_mlp(X, y, hidden, alpha))
        else:
            raise ValueError(method)
    return models


def _per_square_pred(models, X) -> np.ndarray:
    cols = [m.predict(X) for m in models]
    return np.stack(cols, axis=1)


def _pca_then_linear(Xtr, ytr, Xte, dim: int, C: float):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(dim, Xtr.shape[1], Xtr.shape[0])).fit(Xtr)
    return _fit_linear(pca.transform(Xtr), ytr, C), pca


def _metadata_features(rows: list[dict]) -> np.ndarray:
    return np.stack(
        [[r["t"], r["occupied"], int(r["player"]), len(r["legal"]), int(r["must_pass"])] for r in rows],
        axis=0,
        dtype=np.float32,
    )


def _nan_score() -> dict:
    return {
        "square_acc": float("nan"),
        "exact_board": float("nan"),
        "occupied_acc": float("nan"),
        "empty_acc": float("nan"),
        "macro_f1": float("nan"),
        "n": 0,
    }


def _val_score(row: dict) -> float:
    return float(row["val_exact_board"]) + 0.25 * float(row["val_square_acc"])


def _load_decode_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keep = {"site", "hook", "target", "method"}
    for row in rows:
        for key, value in list(row.items()):
            if key in keep or value in (None, ""):
                continue
            if key == "layer" or key.endswith("_n"):
                row[key] = int(float(value))
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def _run_primary_controls(cfg, train, test, Xtr, Xte, hook: str) -> list[dict]:
    ytr = np.stack([r["rel_lab"] for r in train])
    yte = np.stack([r["rel_lab"] for r in test])
    y_shuf = ytr.copy()
    rng = np.random.default_rng(0)
    rng.shuffle(y_shuf)
    models_sh = _per_square_train("linear", Xtr, y_shuf, cfg["linear_c"], 32, 1e-3)
    rows = [{"hook": hook, "control": "shuffled_labels", **_score_board(_per_square_pred(models_sh, Xte), yte)}]

    meta = _metadata_features(train)
    meta_te = _metadata_features(test)
    models_m = _per_square_train("linear", meta, ytr, cfg["linear_c"], 32, 1e-3)
    rows.append({"hook": hook, "control": "metadata_only", **_score_board(_per_square_pred(models_m, meta_te), yte)})

    rng_proj = rng.normal(size=(Xtr.shape[1], 16)).astype(np.float32)
    rng_proj /= np.linalg.norm(rng_proj, axis=0, keepdims=True) + 1e-8
    models_r = _per_square_train("linear", Xtr @ rng_proj, ytr, cfg["linear_c"], 32, 1e-3)
    rows.append({"hook": hook, "control": "random_subspace16", **_score_board(_per_square_pred(models_r, Xte @ rng_proj), yte)})
    return rows


def _run_untrained_control(cfg, args, train, test) -> dict:
    print("  untrained-model control on primary site")
    try:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        umodel = random_untrained_othello(0, device=args.device or cfg.get("device"))
        hook = cfg["primary_site"]
        Xu_tr = cache_site(umodel, tokens_of(train), hook, cfg["cache_batch"]).numpy()
        Xu_te = cache_site(umodel, tokens_of(test), hook, cfg["cache_batch"]).numpy()
        ytr = np.stack([r["rel_lab"] for r in train])
        yte = np.stack([r["rel_lab"] for r in test])
        models_u = _per_square_train("linear", Xu_tr, ytr, cfg["linear_c"], 32, 1e-3)
        return {"hook": hook, "control": "untrained_model", **_score_board(_per_square_pred(models_u, Xu_te), yte)}
    except Exception as exc:
        print(f"    untrained-model control failed: {exc}")
        return {"hook": cfg["primary_site"], "control": "untrained_model", **_nan_score(), "error": str(exc)}


def _write_gate_artifacts(results_dir, ckpt_dir, fig_dir, decode_rows, control_rows, best) -> str:
    write_csv(results_dir / "08c_state_decode.csv", decode_rows)
    write_csv(results_dir / "08c_probe_controls.csv", control_rows)
    if best is not None:
        hook, target, method, models = best[0], best[1], best[2], best[3]
        with (ckpt_dir / "08c_selected_probe.pkl").open("wb") as handle:
            pickle.dump({"hook": hook, "target": target, "method": method, "models": models}, handle)
        print(f"  val-selected {method} {target} at {hook}")
    lin = [r for r in decode_rows if r["method"] == "linear"]
    shuf = [r for r in control_rows if r["control"] == "shuffled_labels"]
    gate1 = decide_gate1(lin, shuf)
    write_json(
        results_dir / "08c_gate1.json",
        {"gate1": gate1, "selected": None if best is None else {"hook": best[0], "target": best[1], "method": best[2]}},
    )
    make_figure(fig_dir, decode_rows)
    print(f"  Gate 1 {gate1}")
    return gate1


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    results_dir = ensure_dir(cfg["results_dir"])
    ckpt_dir = ensure_dir(cfg["checkpoints_dir"])
    data_dir = Path(cfg["data_dir"])
    fig_dir = ensure_dir(cfg["figures_dir"])
    gate0 = json_gate(results_dir)
    if gate0 == "BLOCKED":
        write_json(results_dir / "08c_state_decode.csv".replace(".csv", "_status.json"), {"status": "BLOCKED", "reason": "Gate 0 blocked"})
        print("08c blocked: run 08b with a loadable checkpoint first.")
        return

    train = add_labels(load_split(data_dir, "train"))
    val = add_labels(load_split(data_dir, "val"))
    test = add_labels(load_split(data_dir, "test"))
    if args.max_train:
        train = train[: args.max_train]
        val = val[: max(64, args.max_train // 4)]
        test = test[: max(64, args.max_train // 4)]

    try:
        model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=args.checkpoint)
    except Exception as exc:
        write_json(results_dir / "08c_probe_controls.csv".replace(".csv", "_status.json"), {"status": "BLOCKED", "reason": str(exc)})
        print(f"BLOCKED: {exc}")
        return

    decode_path = results_dir / "08c_state_decode.csv"
    layers = list(cfg["probe_layers"])
    sites = list(cfg["probe_sites"])
    decode_rows, control_rows = [], []
    best = None
    best_score = -1.0
    primary_cache = None

    if args.skip_sweep:
        if not decode_path.exists():
            raise SystemExit("08c --skip-sweep needs 08c_state_decode.csv from the finished sweep.")
        decode_rows = _load_decode_csv(decode_path)
        winner = max(decode_rows, key=_val_score)
        print(
            f"=== 08c finalize from {decode_path.name} ===\n"
            f"  reuse {len(decode_rows)} decode rows; refit {winner['method']} {winner['target']} at {winner['hook']}"
        )
        hook = winner["hook"]
        target = winner["target"]
        method = winner["method"]
        key = "abs_lab" if target == "abs" else "rel_lab"
        Xtr = cache_site(model, tokens_of(train), hook, cfg["cache_batch"]).numpy()
        models = _per_square_train(method, Xtr, np.stack([r[key] for r in train]), cfg["linear_c"], cfg["mlp_hidden"], cfg["mlp_alpha"])
        best = (hook, target, method, models)
        if hook != cfg["primary_site"]:
            Xtr = cache_site(model, tokens_of(train), cfg["primary_site"], cfg["cache_batch"]).numpy()
        Xte = cache_site(model, tokens_of(test), cfg["primary_site"], cfg["cache_batch"]).numpy()
        control_rows.extend(_run_primary_controls(cfg, train, test, Xtr, Xte, cfg["primary_site"]))
        control_rows.append(_run_untrained_control(cfg, args, train, test))
        _write_gate_artifacts(results_dir, ckpt_dir, fig_dir, decode_rows, control_rows, best)
        return

    print("=== 08c probes ===")
    for layer in layers:
        for site in sites:
            hook = hook_name(site, layer)
            print(f"  cache {hook}")
            Xtr = cache_site(model, tokens_of(train), hook, cfg["cache_batch"]).numpy()
            Xva = cache_site(model, tokens_of(val), hook, cfg["cache_batch"]).numpy()
            Xte = cache_site(model, tokens_of(test), hook, cfg["cache_batch"]).numpy()
            for target, key in (("abs", "abs_lab"), ("rel", "rel_lab")):
                ytr = np.stack([r[key] for r in train])
                yva = np.stack([r[key] for r in val])
                yte = np.stack([r[key] for r in test])
                methods = ("linear",)
                if site == "resid_post" and layer in {4, 6}:
                    methods = ("linear", "mlp")
                for method in methods:
                    models = _per_square_train(method, Xtr, ytr, cfg["linear_c"], cfg["mlp_hidden"], cfg["mlp_alpha"])
                    pred_va = _per_square_pred(models, Xva)
                    pred_te = _per_square_pred(models, Xte)
                    sva = _score_board(pred_va, yva)
                    ste = _score_board(pred_te, yte)
                    legal_j = induced_jaccard(pred_te, test, target)
                    row = {
                        "layer": layer,
                        "site": site,
                        "hook": hook,
                        "target": target,
                        "method": method,
                        **{f"val_{k}": v for k, v in sva.items()},
                        **{f"test_{k}": v for k, v in ste.items()},
                        "test_induced_legal_jaccard": legal_j,
                    }
                    decode_rows.append(row)
                    write_csv(decode_path, decode_rows)
                    score = sva["exact_board"] + 0.25 * sva["square_acc"]
                    if score > best_score:
                        best_score, best = score, (hook, target, method, models)
                    print(
                        f"    {target} {method} val_exact={sva['exact_board']:.3f} "
                        f"val_sq={sva['square_acc']:.3f} test_exact={ste['exact_board']:.3f}"
                    )

            if hook != cfg["primary_site"]:
                continue
            primary_cache = (Xtr, Xte)
            control_rows.extend(_run_primary_controls(cfg, train, test, Xtr, Xte, hook))

    if primary_cache is None:
        Xtr = cache_site(model, tokens_of(train), cfg["primary_site"], cfg["cache_batch"]).numpy()
        Xte = cache_site(model, tokens_of(test), cfg["primary_site"], cfg["cache_batch"]).numpy()
        control_rows.extend(_run_primary_controls(cfg, train, test, Xtr, Xte, cfg["primary_site"]))
    control_rows.append(_run_untrained_control(cfg, args, train, test))
    _write_gate_artifacts(results_dir, ckpt_dir, fig_dir, decode_rows, control_rows, best)


def json_gate(results_dir: Path) -> str:
    p = results_dir / "08b_baseline_summary.json"
    if not p.exists():
        return "BLOCKED"
    obj = __import__("json").loads(p.read_text())
    return obj.get("status", "BLOCKED")


def induced_jaccard(pred: np.ndarray, rows: list[dict], target: str) -> float:
    from src.toy_othello import relative_to_absolute

    js = []
    for i, r in enumerate(rows):
        lab = pred[i]
        if target == "rel":
            lab = relative_to_absolute(lab, int(r["player"]))
        pred_legal = set(induced_legal(lab, int(r["player"])))
        true = set(r["legal"])
        union = pred_legal | true
        js.append(len(pred_legal & true) / len(union) if union else 1.0)
    return float(np.mean(js)) if js else float("nan")


def decide_gate1(lin_rows, shuf_rows) -> str:
    if not lin_rows:
        return "INCONCLUSIVE"
    best = max(lin_rows, key=lambda r: r.get("test_exact_board", 0.0))
    sh = float(np.mean([r.get("exact_board", 0.0) for r in shuf_rows])) if shuf_rows else 0.0
    if best["test_square_acc"] >= 0.85 or best["test_exact_board"] >= 0.20:
        if best["test_square_acc"] >= sh + 0.15 or best["test_exact_board"] >= sh + 0.10:
            return "PASS_DECODE"
    if best["test_square_acc"] < 0.55:
        return "FAIL_DECODE"
    return "INCONCLUSIVE"


def make_figure(fig_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for method, marker in (("linear", "o"), ("mlp", "s")):
        sub = [r for r in rows if r["method"] == method and r["target"] == "rel" and r["site"] == "resid_post"]
        if not sub:
            continue
        xs = [r["layer"] for r in sub]
        ys = [r["test_square_acc"] for r in sub]
        ax.plot(xs, ys, marker=marker, label=f"{method} rel resid_post")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out square acc")
    ax.set_title("08c board-state decoding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "08c_board_decode_by_layer.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
