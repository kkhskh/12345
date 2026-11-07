#!/usr/bin/env python3
"""Generate Path 1 figures from reported_numbers.json. Does not load Drive NPZ files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUM = json.loads((ROOT / "reported_numbers.json").read_text())
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def figure1_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    boxes = [
        (0.25, 2.2, 2.3, 1.8, "Residual\n$h\\in\\mathbb{R}^{512}$"),
        (3.2, 4.2, 2.6, 1.5, "Rowspace $P_R h$\n(this probe reads)"),
        (3.2, 0.4, 2.6, 1.5, "Nullspace $P_N h$\n($W P_N=0$)"),
        (6.6, 4.2, 3.9, 1.5, "Frozen board probe\n64$\\times$3 probabilities"),
        (6.6, 1.8, 3.9, 2.0, "Later layers of the\nfrozen model\nnext-move softmax, $Y=G-Q$"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center")
    ax.annotate("", xy=(3.2, 4.9), xytext=(2.55, 3.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.2, 1.1), xytext=(2.55, 2.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.6, 4.95), xytext=(5.8, 4.95), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.6, 3.2), xytext=(5.8, 4.7), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.6, 2.4), xytext=(5.8, 1.15), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Figure 1 (schematic). Both components can affect the output; only the rowspace is read by this probe.")
    save(fig, "fig1_probe_geometry_schematic")


def _forest(ax, rows, xlabel, title, note):
    y = np.arange(len(rows))[::-1]
    for i, r in enumerate(rows):
        yi = y[i]
        ax.plot(r["ci"], [yi, yi], color="0.35", lw=1.5)
        ax.plot(r["mean"], yi, "o", color="C0", ms=6)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    fig = ax.figure
    fig.text(0.01, 0.02, note, fontsize=8, va="bottom", wrap=True)


def figure3_fixed_js() -> None:
    fj = NUM["fixed_js"]
    rows = [
        {
            "label": "correct − wrong (recorded)",
            "mean": fj["delta_wrong_mean"]["value"],
            "ci": fj["delta_wrong_mean"]["ci95"],
            "xmax_label": 0.04,
        },
        {
            "label": "correct − random (recorded)",
            "mean": fj["delta_random_mean"]["value"],
            "ci": fj["delta_random_mean"]["ci95"],
            "xmax_label": 0.04,
        },
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    _forest(
        ax,
        rows,
        xlabel="Paired ΔY at JS = 0.01 (probability-mass units)",
        title="Figure 3. Recorded fixed-JS target-specificity (n=48 pairs; same SPEC sample)",
        note="Means and CIs from reported_numbers.json; pair-level Y also in verified_data/fixed_js_verified_data.csv (n=48 pairs, 336 interventions). Equal JS ≠ equal L2. Original COST is a separate inconclusive run.",
    )
    ax.set_xlim(-0.005, 0.045)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig3_fixed_js_contrasts")


def figure4_nst() -> None:
    nst = NUM["nst"]
    rows = [
        {
            "label": "learned − wrong",
            "mean": nst["y_learned_minus_wrong"]["mean"],
            "ci": nst["y_learned_minus_wrong"]["ci95"],
            "xmax_label": 0.035,
        },
        {
            "label": "learned − random",
            "mean": nst["y_learned_minus_random"]["mean"],
            "ci": nst["y_learned_minus_random"]["ci95"],
            "xmax_label": 0.05,
        },
        {
            "label": "learned − mean-state",
            "mean": nst["y_learned_minus_mean_state"]["mean"],
            "ci": nst["y_learned_minus_mean_state"]["ci95"],
            "xmax_label": 0.035,
        },
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    _forest(
        ax,
        rows,
        xlabel="Paired ΔY at JS = 0.01",
        title="Figure 4. Donor-free NST contrasts (n=48; pasted Colab pooled stdout)",
        note="Mean Y_learned = %.3f (still negative). Donor-free at evaluation, not at training. Gradient baseline JS-ok on 7/48 is omitted from this full-sample panel."
        % nst["mean_Y_learned"]["value"],
    )
    ax.set_xlim(-0.005, 0.05)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, "fig4_nst_contrasts")


def figure5_limits() -> None:
    fr = NUM["fresh_symmetric"]
    vnf = NUM["vnf"]
    sem = NUM["sem"]["mean_level_correction"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    ax = axes[0]
    labs = [r"$A_T$", r"$A_W$"]
    means = [fr["a_t_mean"]["value"], fr["a_w_mean"]["value"]]
    los = [fr["a_t_mean"]["ci95"][0], fr["a_w_mean"]["ci95"][0]]
    his = [fr["a_t_mean"]["ci95"][1], fr["a_w_mean"]["ci95"][1]]
    ax.errorbar([0, 1], means, yerr=[np.array(means) - los, his - np.array(means)], fmt="o", capsize=4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks([0, 1], labs)
    ax.set_ylabel("Mean advantage (Y units)")
    ax.set_title("FRESH natural-anchor FULL, n=32 triplets")
    ax.set_ylim(-0.01, 0.05)

    ax = axes[1]
    labs = [r"$I_T$", r"$I_W$"]
    means = [vnf["i_t"]["value"], vnf["i_w"]["value"]]
    los = [vnf["i_t"]["ci95"][0], vnf["i_w"]["ci95"][0]]
    his = [vnf["i_t"]["ci95"][1], vnf["i_w"]["ci95"][1]]
    ax.errorbar([0, 1], means, yerr=[np.array(means) - los, his - np.array(means)], fmt="o", capsize=4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axhline(0.005, color="C1", ls="--", lw=0.9, label="prespecified +0.005")
    ax.set_xticks([0, 1], labs)
    ax.set_title("VNF interaction, n=32 triplets")
    ax.set_ylim(-0.002, 0.007)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 5. Limits: switching gate failed; factorial interaction unsupported")
    fig.text(
        0.5,
        0.02,
        "FRESH both-positive 13/32 = 0.406 < 0.60 gate (SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED). "
        "VNF: TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED. "
        "STAGE share of FULL−EditorA = %.1f%% (mean ratio)."
        % (100 * sem["stage_share_of_full_minus_editor_a"]["value"]),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    save(fig, "fig5_limits_fresh_vnf")


def load_fixed_js_rows() -> list[dict]:
    path = ROOT / "verified_data" / "fixed_js_verified_data.csv"
    with path.open() as handle:
        return list(csv.DictReader(handle))


def figure2_from_verified_csv() -> None:
    rows = load_fixed_js_rows()
    probe = np.array([float(r["probe_max_abs"]) for r in rows])
    js = np.array([float(r["js"]) for r in rows])
    tv = np.array([float(r["tv"]) for r in rows])
    methods = [r["method"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax = axes[0]
    ax.scatter(probe, js - 0.01, s=12, alpha=0.45, c="C0")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Max |Δ probe probability| (64×3)")
    ax.set_ylabel("JS minus 0.01")
    ax.set_title("Probe held; JS at 0.01")
    ax = axes[1]
    colors = {"correct": "C0", "wrong": "C1", "random": "C2"}
    for m in ("correct", "wrong", "random"):
        mask = [method == m for method in methods]
        ax.scatter(probe[mask], tv[mask], s=12, alpha=0.5, label=m, c=colors[m])
    ax.set_xscale("log")
    ax.set_xlabel("Max |Δ probe probability| (64×3)")
    ax.set_ylabel("Total variation of output")
    ax.set_title("Outputs still move")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 2. Fixed-JS probe preservation vs output change (336 interventions, 48 pairs)")
    fig.text(
        0.5,
        0.01,
        "Unit of inference is the pair, not the intervention. Largest probe change 1.19e-14; max |JS-0.01|=1.23e-6. "
        "Source: verified_data/fixed_js_verified_data.csv.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    save(fig, "fig2_probe_preservation")


if __name__ == "__main__":
    figure1_schematic()
    figure2_from_verified_csv()
    figure3_fixed_js()
    figure4_nst()
    figure5_limits()
    print("wrote figures in", FIG)
