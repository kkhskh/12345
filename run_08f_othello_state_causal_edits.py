"""08f: state-specific edits vs full patch / attribution / logit steer. Gate 2–3."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_runutil import add_labels, load_split, mean_prob_on, pair_candidates, tokens_of, write_json
from src.toy_othello import cache_site, load_config, load_othello_gpt, patch_site, run_model_logits, softmax_np, square_to_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08f causal state edits.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _dP(pa, pb, squares) -> float:
    a = mean_prob_on(pa, squares, square_to_token)
    b = mean_prob_on(pb, squares, square_to_token)
    if a != a or b != b:
        return float("nan")
    return a - b


def _effects(name, pb, pa, rec, don, extra) -> dict:
    became = list(set(don["legal"]) - set(rec["legal"]))
    left = list(set(rec["legal"]) - set(don["legal"]))
    stay = list(set(rec["legal"]) & set(don["legal"]))
    row = {
        "method": name,
        "became_legal_dP": _dP(pa, pb, became),
        "became_illegal_dP": _dP(pa, pb, left),
        "stayed_legal_dP": _dP(pa, pb, stay),
        "donor_legal_mass_after": mean_prob_on(pa, don["legal"], square_to_token),
        "donor_legal_mass_before": mean_prob_on(pb, don["legal"], square_to_token),
        "type": extra.get("type"),
        "hamming": extra.get("hamming"),
        "edit_norm": extra.get("edit_norm", float("nan")),
        "decoder_target_agree": extra.get("decoder_target_agree", float("nan")),
    }
    return row


def probe_predict(models, x: np.ndarray) -> np.ndarray:
    return np.stack([int(m.predict(x.reshape(1, -1))[0]) for m in models], axis=0)


def linear_edit(x, rec_lab, don_lab, models, budget: float) -> np.ndarray:
    """Method A: add target-minus-source class directions for disagreed squares."""
    delta = np.zeros_like(x)
    n = 0
    for j in range(64):
        if rec_lab[j] == don_lab[j]:
            continue
        clf = models[j]
        if not hasattr(clf, "coef_"):
            continue
        coef = clf.coef_
        if coef.shape[0] < 3:
            continue
        delta = delta + (coef[int(don_lab[j])] - coef[int(rec_lab[j])])
        n += 1
    if n == 0:
        return x
    vec = delta
    nrm = float(np.linalg.norm(vec))
    if nrm > budget:
        vec = vec * (budget / nrm)
    return x + vec


def mean_square_transform(Xtr, ytr, y_target_shift, rec_x, rec_lab, don_lab, budget: float) -> np.ndarray:
    """Method C: train-mean activation delta for each square class change."""
    delta = np.zeros_like(rec_x)
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
        return rec_x
    vec = delta
    nrm = float(np.linalg.norm(vec))
    if nrm > budget:
        vec = vec * (budget / nrm)
    return rec_x + vec


def subspace_project(rec_h, don_h, models) -> tuple[np.ndarray, np.ndarray]:
    """Method D: transfer probe-row space component of the donor residual."""
    dirs = []
    for clf in models:
        if hasattr(clf, "coef_"):
            dirs.append(clf.coef_)
    if not dirs:
        return rec_h, don_h
    B = np.concatenate(dirs, axis=0).T  # d x (64*C)
    q, _ = np.linalg.qr(B)
    q = q[:, : min(32, q.shape[1])]
    def proj(v):
        return (v @ q) @ q.T
    state = proj(don_h)
    comp = rec_h - proj(rec_h) + state
    return comp, rec_h - proj(rec_h) + proj(rec_h) * 0 + (rec_h - proj(rec_h))


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
        write_json(results_dir / "08f_status.json", {"status": "BLOCKED", "reason": "08c probe missing"})
        print("08f blocked: run 08c first.")
        return
    with probe_path.open("rb") as handle:
        pack = pickle.load(handle)
    train = add_labels(load_split(Path(cfg["data_dir"]), "train"))
    test = add_labels(load_split(Path(cfg["data_dir"]), "test"))
    try:
        model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=args.checkpoint)
    except Exception as exc:
        write_json(results_dir / "08f_status.json", {"status": "BLOCKED", "reason": str(exc)})
        print(f"BLOCKED: {exc}")
        return

    hook = pack["hook"]
    pairs = [p for p in pair_candidates(test, cfg["n_pairs"], cfg["seed"] + 21) if p["n_became_legal"] + p["n_became_illegal"] > 0]
    if len(pairs) < 8:
        pairs = pair_candidates(test, cfg["n_pairs"], cfg["seed"] + 21)
    rec_rows = [test[p["recip"]] for p in pairs]
    don_rows = [test[p["donor"]] for p in pairs]
    rec_tok = tokens_of(rec_rows)
    don_tok = tokens_of(don_rows)
    rec_h = cache_site(model, rec_tok, hook, cfg["cache_batch"]).numpy()
    don_h = cache_site(model, don_tok, hook, cfg["cache_batch"]).numpy()
    Xtr = cache_site(model, tokens_of(train), hook, cfg["cache_batch"]).numpy()
    ytr = np.stack([r["rel_lab"] if pack["target"] == "rel" else r["abs_lab"] for r in train])
    base = run_model_logits(model, rec_tok, cfg["batch_size"])
    pb = softmax_np(base)
    budget = float(cfg["edit_norm_budget"])
    models = pack["models"]

    methods = {}
    methods["full_patch"] = don_h
    methods["no_op"] = rec_h
    rng = np.random.default_rng(0)
    rand = rec_h + rng.normal(size=rec_h.shape).astype(np.float32)
    rand = rec_h + (rand - rec_h) * (budget / (np.linalg.norm(rand - rec_h, axis=1, keepdims=True) + 1e-8))
    methods["norm_matched_random"] = rand
    # PCA same-energy
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(32, rec_h.shape[1], rec_h.shape[0])).fit(Xtr)
    rec_p = pca.transform(rec_h)
    don_p = pca.transform(don_h)
    methods["pca32_patch"] = pca.inverse_transform(don_p)
    methods["pca32_complement"] = rec_h - pca.inverse_transform(rec_p) + pca.inverse_transform(don_p)

    lin = np.stack(
        [
            linear_edit(rec_h[i], rec_rows[i]["rel_lab" if pack["target"] == "rel" else "abs_lab"], don_rows[i]["rel_lab" if pack["target"] == "rel" else "abs_lab"], models, budget)
            for i in range(len(pairs))
        ]
    )
    methods["linear_board_dirs"] = lin
    mean_ed = np.stack(
        [
            mean_square_transform(
                Xtr,
                ytr,
                None,
                rec_h[i],
                rec_rows[i]["rel_lab" if pack["target"] == "rel" else "abs_lab"],
                don_rows[i]["rel_lab" if pack["target"] == "rel" else "abs_lab"],
                budget,
            )
            for i in range(len(pairs))
        ]
    )
    methods["train_mean_square_delta"] = mean_ed
    proj = []
    comp = []
    for i in range(len(pairs)):
        p, c = subspace_project(rec_h[i], don_h[i], models)
        proj.append(p)
        comp.append(c)
    methods["donor_probe_subspace"] = np.stack(proj)
    methods["donor_complement"] = np.stack(comp)
    shuf = don_h.copy()
    rng.shuffle(shuf)
    methods["shuffled_donor"] = shuf

    # logit steer baseline: +4 on donor-legal tokens
    z_steer = base.clone()
    for i, d in enumerate(don_rows):
        for sq in d["legal"]:
            if sq in {27, 28, 35, 36} or sq == -100:
                continue
            z_steer[i, square_to_token(sq)] += 4.0

    # attribution: sign of d legal_mass / dh approximated by mean donor-recip * |T| proxy = residual itself
    attr = rec_h + np.sign(don_h - rec_h) * (budget / 32.0)
    methods["sign_attribution"] = attr

    rows = []
    spec = []
    for name, H in methods.items():
        patched = patch_site(model, rec_tok, hook, torch.tensor(H, dtype=torch.float32), cfg["cache_batch"])
        pa = softmax_np(patched)
        dec_agree = []
        for i in range(len(pairs)):
            pred = probe_predict(models, H[i])
            tgt = don_rows[i]["rel_lab" if pack["target"] == "rel" else "abs_lab"]
            dec_agree.append(float((pred == tgt).mean()))
            extra = {
                "type": pairs[i]["type"],
                "hamming": pairs[i]["hamming"],
                "edit_norm": float(np.linalg.norm(H[i] - rec_h[i])),
                "decoder_target_agree": dec_agree[-1],
            }
            rows.append(_effects(name, pb[i], pa[i], rec_rows[i], don_rows[i], extra))
        print(f"  {name} became_legal dP={np.nanmean([r['became_legal_dP'] for r in rows if r['method']==name]):+.3f}")

    pa_s = softmax_np(z_steer)
    for i in range(len(pairs)):
        rows.append(_effects("logit_steer_plus4", pb[i], pa_s[i], rec_rows[i], don_rows[i], {"type": pairs[i]["type"], "hamming": pairs[i]["hamming"], "edit_norm": 0.0}))

    write_csv(results_dir / "08f_state_edit_effects.csv", rows)
    # specificity summary
    for name in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == name]
        spec.append(
            {
                "method": name,
                "mean_became_legal_dP": float(np.nanmean([r["became_legal_dP"] for r in sub])),
                "mean_became_illegal_dP": float(np.nanmean([r["became_illegal_dP"] for r in sub])),
                "mean_stayed_legal_dP": float(np.nanmean([r["stayed_legal_dP"] for r in sub])),
                "mean_decoder_agree": float(np.nanmean([r["decoder_target_agree"] for r in sub])),
                "mean_edit_norm": float(np.nanmean([r["edit_norm"] for r in sub])),
                "n": len(sub),
            }
        )
    write_csv(results_dir / "08f_causal_specificity.csv", spec)
    write_json(results_dir / "08f_status.json", {"status": "OK", "hook": hook, "n_pairs": len(pairs)})
    make_fig(fig_dir, spec)
    print(f"saved {results_dir / '08f_causal_specificity.csv'}")


def make_fig(fig_dir: Path, spec: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [s["method"] for s in spec]
    ax.bar(range(len(names)), [s["mean_became_legal_dP"] for s in spec])
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("became-legal ΔP")
    ax.set_title("08f causal specificity vs baselines")
    fig.tight_layout()
    fig.savefig(fig_dir / "08f_causal_baselines.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
