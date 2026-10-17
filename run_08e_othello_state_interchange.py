"""08e: full activation interchange positive control. Gate 2 start."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_runutil import add_labels, load_split, mean_prob_on, pair_candidates, tokens_of, write_json
from src.toy_othello import (
    cache_site,
    hook_name,
    load_config,
    load_othello_gpt,
    patch_site,
    run_model_logits,
    softmax_np,
    square_to_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08e full activation interchange.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _mass(probs, squares) -> float:
    return mean_prob_on(probs, squares, square_to_token)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    results_dir = ensure_dir(cfg["results_dir"])
    fig_dir = ensure_dir(cfg["figures_dir"])
    test = add_labels(load_split(Path(cfg["data_dir"]), "test"))
    try:
        model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=args.checkpoint)
    except Exception as exc:
        write_json(results_dir / "08e_status.json", {"status": "BLOCKED", "reason": str(exc)})
        print(f"BLOCKED: {exc}")
        return

    pairs = pair_candidates(test, cfg["n_pairs"], cfg["seed"] + 17)
    if len(pairs) < 8:
        write_json(results_dir / "08e_status.json", {"status": "BLOCKED", "reason": "too few pairs"})
        print("08e blocked: too few matched pairs")
        return
    rec_i = [p["recip"] for p in pairs]
    don_i = [p["donor"] for p in pairs]
    rec_rows = [test[i] for i in rec_i]
    don_rows = [test[i] for i in don_i]
    rec_tok = tokens_of(rec_rows)
    base = run_model_logits(model, rec_tok, cfg["batch_size"])
    noop = patch_site(model, rec_tok, cfg["primary_site"], cache_site(model, rec_tok, cfg["primary_site"], cfg["cache_batch"]), cfg["cache_batch"])
    noop_rel = float((noop - base).abs().max())
    print(f"  no-op max logit delta={noop_rel:.3e}")

    layers = list(cfg["probe_layers"])
    rows = []
    for layer in layers:
        for site in ("resid_mid", "resid_post"):
            hook = hook_name(site, layer)
            don_h = cache_site(model, tokens_of(don_rows), hook, cfg["cache_batch"])
            rec_h = cache_site(model, rec_tok, hook, cfg["cache_batch"])
            patched = patch_site(model, rec_tok, hook, don_h, cfg["cache_batch"])
            pb = softmax_np(base)
            pa = softmax_np(patched)
            for k, p in enumerate(pairs):
                r, d = rec_rows[k], don_rows[k]
                became = list(set(d["legal"]) - set(r["legal"]))
                left = list(set(r["legal"]) - set(d["legal"]))
                stay = list(set(r["legal"]) & set(d["legal"]))
                rows.append(
                    {
                        "hook": hook,
                        "type": p["type"],
                        "hamming": p["hamming"],
                        "donor_legal_mass_before": _mass(pb[k], d["legal"]),
                        "donor_legal_mass_after": _mass(pa[k], d["legal"]),
                        "recip_legal_mass_before": _mass(pb[k], r["legal"]),
                        "recip_legal_mass_after": _mass(pa[k], r["legal"]),
                        "became_legal_dP": _mass(pa[k], became) - _mass(pb[k], became),
                        "became_illegal_dP": _mass(pa[k], left) - _mass(pb[k], left),
                        "stayed_legal_dP": _mass(pa[k], stay) - _mass(pb[k], stay),
                        "kl": float((pb[k] * (np.log(pb[k] + 1e-12) - np.log(pa[k] + 1e-12))).sum()),
                        "patch_norm": float(torch.norm(don_h[k] - rec_h[k]).item()),
                    }
                )
            sub = [r for r in rows if r["hook"] == hook]
            lift = float(np.nanmean([r["donor_legal_mass_after"] - r["donor_legal_mass_before"] for r in sub]))
            print(f"  {hook} donor-legal lift={lift:+.3f} n={len(sub)}")

    write_csv(results_dir / "08e_full_interchange.csv", rows)
    write_json(results_dir / "08e_status.json", {"status": "OK", "n_pairs": len(pairs), "noop_max_delta": noop_rel})
    make_fig(fig_dir, rows)
    print(f"saved {results_dir / '08e_full_interchange.csv'}")


def make_fig(fig_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hooks = sorted({r["hook"] for r in rows})
    ys = []
    for h in hooks:
        sub = [r for r in rows if r["hook"] == h]
        ys.append(float(np.nanmean([r["donor_legal_mass_after"] - r["donor_legal_mass_before"] for r in sub])))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(hooks)), ys)
    ax.set_xticks(range(len(hooks)), hooks, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("donor-legal mass lift")
    ax.set_title("08e full interchange")
    fig.tight_layout()
    fig.savefig(fig_dir / "08e_full_interchange.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
