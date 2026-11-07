"""Layer C -> D: draw figures/ from analysis/*.csv, analysis/results.json and tables/*.csv.

Usage (from anywhere):
    MPLBACKEND=Agg python scripts/reproduce_figures.py
Writes figures/fig1..fig5 as PDF and PNG plus figures/figure_provenance.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1lib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG = p1lib.PKG / "figures"
COLORS = {"correct": "#1b6ca8", "wrong": "#d1495b", "random": "#7a7a7a", "learned": "#1b6ca8"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def read_rows(path: Path) -> list[dict]:
    p1lib.require(path.is_file(), f"figure input missing: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    p1lib.require(len(rows) > 0, f"figure input empty: {path}")
    return rows


def save(fig, name: str, inputs: list[Path], prov: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("pdf", "png"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else None)
        outs.append(p)
    plt.close(fig)
    prov[name] = {
        "inputs": {str(p.relative_to(p1lib.PKG)): p1lib.sha256_file(p) for p in inputs},
        "outputs": [f"figures/{p.name}" for p in outs],
        "script": "scripts/reproduce_figures.py",
    }


def box(ax, xy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="#333333", lw=0.9))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8.2)


def arrow(ax, a, b, text=None, off=(0, 0.03), style="-|>", color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, lw=1.0, color=color, linestyle=ls))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=7.8, color=color)


def figure1(prov: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top, h = 0.74, 0.20
    box(ax, (0.00, top), 0.18, h, "natural recipient\nactivation $h_R$", "#eef3f8")
    box(ax, (0.33, top), 0.20, h, "Editor A reference\nactivation $h_E$", "#e3eef9")
    box(ax, (0.67, top), 0.33, h, "patched activation $h_E + t\\,\\delta$\nwith $\\delta$ in the nullspace of $W$", "#fdf1e6")
    arrow(ax, (0.19, top + h / 2), (0.325, top + h / 2))
    ax.text(0.2575, top + h / 2 + 0.025, "Editor A", ha="center", va="bottom", fontsize=7.8)
    arrow(ax, (0.54, top + h / 2), (0.665, top + h / 2))
    ax.text(0.6025, top + h / 2 + 0.025, "add $t\\,\\delta$", ha="center", va="bottom", fontsize=7.8)
    mid, mh = 0.38, 0.19
    box(ax, (0.00, mid), 0.24, mh, "frozen linear probe:\n64×3 board probabilities", "#eef3f8")
    box(ax, (0.345, mid), 0.30, mh, "rowspace component $P_R h$\n(124 dimensions)", "#e3eef9")
    box(ax, (0.72, mid), 0.28, mh, "nullspace component $P_N h$\n(388 dimensions)", "#fdf1e6")
    arrow(ax, (0.78, top), (0.52, mid + mh))
    arrow(ax, (0.86, top), (0.86, mid + mh))
    ax.text(0.67, 0.64, "decompose", ha="center", fontsize=7.8, color="#333333")
    arrow(ax, (0.34, mid + mh / 2), (0.245, mid + mh / 2))
    ax.text(0.2925, mid - 0.12, "read by\nthe probe", ha="center", va="bottom", fontsize=7.5)
    box(ax, (0.37, 0.03), 0.63, 0.17, "frozen later layers  →  next-move distribution (61 tokens)", "#f2f2f2")
    arrow(ax, (0.495, mid), (0.495, 0.20), color="#1b6ca8")
    arrow(ax, (0.86, mid), (0.86, 0.20), color="#c9752f")
    ax.text(0.68, 0.265, "both components feed the later layers", ha="center", fontsize=7.8)
    ax.text(0.12, 0.17, "The probe's readout at $h_E + t\\,\\delta$\nequals its readout at $h_E$,\nbut not at $h_R$.", ha="center", va="center", fontsize=7.8, style="italic")
    save(fig, "fig1_probe_geometry", [], prov)


def strip(ax, xs, ys, color, jitter=0.09, seed=0, size=11):
    rng = np.random.default_rng(seed)
    ax.scatter(xs + rng.uniform(-jitter, jitter, len(ys)), ys, s=size, color=color, alpha=0.75, lw=0)


def figure2(prov: dict) -> None:
    path = p1lib.ANALYSIS / "fixed_js_pairs.csv"
    rows = read_rows(path)
    p1lib.require(len(rows) == 48, f"figure 2 expects 48 pairs, found {len(rows)}")
    cats = [("correct", "probe_max_correct", "tv_correct"), ("wrong", "probe_max_wrong", "tv_wrong_mean"), ("random", "probe_max_random", "tv_random_mean")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for i, (cat, pcol, tcol) in enumerate(cats):
        pv = np.array([float(r[pcol]) for r in rows])
        tv = np.array([float(r[tcol]) for r in rows])
        p1lib.require(np.all(pv > 0) and np.all(np.isfinite(tv)), f"figure 2 invalid values for {cat}")
        strip(axes[0], np.full(48, i), pv, COLORS[cat], seed=i)
        strip(axes[1], np.full(48, i), tv, COLORS[cat], seed=10 + i)
        axes[0].hlines(np.exp(np.mean(np.log(pv))), i - 0.25, i + 0.25, color="black", lw=1.4)
        axes[1].hlines(tv.mean(), i - 0.25, i + 0.25, color="black", lw=1.4)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-17, 1e-12)
    axes[0].set_ylabel("max |Δ probe probability|\nvs Editor A reference")
    axes[0].set_title("A. Probe readout at $h_E + t\\delta$ vs $h_E$")
    axes[1].set_ylabel("TV distance of next-move\ndistribution vs Editor A reference")
    axes[1].set_title("B. Output change at JS = 0.01 from $h_E$")
    axes[1].set_ylim(0, None)
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels(["correct", "wrong\n(3 per pair)", "random\n(3 per pair)"])
        ax.set_xlim(-0.6, 2.6)
    fig.text(0.01, -0.02, "One point per recipient–target pair (n = 48). Bars: geometric mean (A), mean (B).", fontsize=7.5, color="#444444")
    fig.tight_layout()
    save(fig, "fig2_probe_and_output_by_category", [path], prov)


def forest(ax, labels, pts, los, his, color, floor=None, xlabel="mean difference in Y"):
    y = np.arange(len(labels))[::-1]
    ax.errorbar(pts, y, xerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)], fmt="o", color=color, ms=4, capsize=2.5, lw=1.1)
    ax.axvline(0, color="black", lw=0.8)
    if floor is not None:
        ax.axvline(floor, color="#999999", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)


def t3_lookup() -> tuple[dict, Path]:
    path = p1lib.PKG / "tables" / "table3_recomputed_contrasts.csv"
    rows = read_rows(path)
    return {(r["study"], r["slice"], r["metric"]): r for r in rows}, path


def figure3(prov: dict) -> None:
    ppath = p1lib.ANALYSIS / "fixed_js_pairs.csv"
    rows = read_rows(ppath)
    t3, tpath = t3_lookup()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (col, met, color, lab) in enumerate([("delta_specific", "delta_specific", COLORS["wrong"], "correct − wrong"), ("delta_random", "delta_random", COLORS["random"], "correct − random")]):
        v = np.array([float(r[col]) for r in rows])
        strip(axes[0], np.full(len(v), i), v, color, seed=20 + i, size=12)
        r = t3[("fixed_js", "pooled", met)]
        pt, lo, hi = float(r["point"]), float(r["ci95_low"]), float(r["ci95_high"])
        axes[0].errorbar([i + 0.3], [pt], yerr=[[pt - lo], [hi - pt]], fmt="s", color="black", ms=4, capsize=3, lw=1.2)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["correct\nminus wrong", "correct\nminus random"])
    axes[0].set_xlim(-0.5, 1.6)
    axes[0].set_ylabel("paired difference in Y")
    axes[0].set_title("A. Per-pair contrasts (n = 48)")
    slices = [("pooled", "all pairs (48)"), ("A", "category A (32)"), ("B", "category B (16)"), ("exact_editor", "Editor A exact (22)"), ("nonexact_editor", "Editor A not exact (26)")]
    y = np.arange(len(slices))[::-1]
    for off, met, color, lab in [(0.15, "delta_specific", COLORS["wrong"], "vs wrong"), (-0.15, "delta_random", COLORS["random"], "vs random")]:
        pts = [float(t3[("fixed_js", s, met)]["point"]) for s, _ in slices]
        los = [float(t3[("fixed_js", s, met)]["ci95_low"]) for s, _ in slices]
        his = [float(t3[("fixed_js", s, met)]["ci95_high"]) for s, _ in slices]
        axes[1].errorbar(pts, y + off, xerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)], fmt="o", color=color, ms=4, capsize=2.5, lw=1.1, label=f"correct {lab}")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([lab for _, lab in slices])
    axes[1].set_xlabel("mean difference in Y (95% CI)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, frameon=False)
    axes[1].set_title("B. Subgroups")
    fig.tight_layout()
    save(fig, "fig3_fixed_js_contrasts", [ppath, tpath], prov)


def figure4(prov: dict) -> None:
    ppath = p1lib.ANALYSIS / "nst_pairs.csv"
    rows = read_rows(ppath)
    t3, tpath = t3_lookup()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    comps = [("wrong", "wrong-target map"), ("random", "random nullspace"), ("mean", "stage-mean"), ("editor", "Editor A reference")]
    slices = [("pooled", "all (48)", "#1b6ca8", 0.2), ("exact", "exact (24)", "#5aa469", 0.0), ("nonexact", "not exact (24)", "#c9752f", -0.2)]
    y = np.arange(len(comps))[::-1]
    for s, lab, color, off in slices:
        pts = [float(t3[("nst", s, m)]["point"]) for m, _ in comps]
        los = [float(t3[("nst", s, m)]["ci95_low"]) for m, _ in comps]
        his = [float(t3[("nst", s, m)]["ci95_high"]) for m, _ in comps]
        axes[0].errorbar(pts, y + off, xerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)], fmt="o", color=color, ms=3.8, capsize=2.2, lw=1.0, label=lab)
    axes[0].axvline(0, color="black", lw=0.8)
    floor = p1lib.PROTOCOL_CONSTANTS["nst_floor"][0]
    axes[0].plot([floor, floor], [y[0] - 0.35, y[0] + 0.35], color="#999999", ls="--", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"learned − {lab}" for _, lab in comps])
    axes[0].set_xlabel("mean difference in Y (95% CI)")
    axes[0].set_title("A. Learned map vs comparisons")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False, title="pairs, by whether Editor A decoded the exact board", title_fontsize=7.5)
    cols = [("cos_learned_oracle", "oracle"), ("cos_learned_mean_state", "stage-mean"), ("cos_learned_wrong", "wrong-target")]
    for i, (c, lab) in enumerate(cols):
        v = np.array([float(r[c]) for r in rows])
        strip(axes[1], np.full(len(v), i), v, ["#1b6ca8", "#5aa469", "#d1495b"][i], seed=30 + i)
        axes[1].hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1.4)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([lab.replace("-", "\n") for _, lab in cols])
    axes[1].set_ylabel("cosine with learned direction")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("B. Direction similarity (n = 48)")
    fig.tight_layout()
    save(fig, "fig4_nst_transfer", [ppath, tpath], prov)


def figure5(prov: dict) -> None:
    rpath = p1lib.ANALYSIS / "results.json"
    p1lib.require(rpath.is_file(), "analysis/results.json missing")
    arch = json.loads(rpath.read_text(encoding="utf-8"))["archival"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    labs = [("fresh.A_T", "$A_T$"), ("fresh.A_W", "$A_W$")]
    forest(axes[0], [l for _, l in labs], [arch[k]["value"] for k, _ in labs], [arch[k]["ci95"][0] for k, _ in labs],
           [arch[k]["ci95"][1] for k, _ in labs], "#1b6ca8", floor=p1lib.PROTOCOL_CONSTANTS["fresh_floor"][0])
    cells = [("fresh.cells_both", "both +"), ("fresh.cells_t_only", "$A_T$ only"), ("fresh.cells_w_only", "$A_W$ only"), ("fresh.cells_neither", "neither")]
    txt = "\n".join(f"{lab}: {int(arch[k]['value'])}" for k, lab in cells)
    axes[0].text(0.97, 0.04, "both positive in " + txt.split("\n")[0].split(": ")[1] + " of 32 triplets\n" + ", ".join(txt.split("\n")[1:]), transform=axes[0].transAxes, fontsize=7.4, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", lw=0.6))
    axes[0].set_ylim(-1.5, 1.6)
    axes[0].set_title("A. Target switching (FRESH, 32 triplets)")
    labs = [("vnf.I_T", "$I_T$"), ("vnf.I_W", "$I_W$")]
    forest(axes[1], [l for _, l in labs], [arch[k]["value"] for k, _ in labs], [arch[k]["ci95"][0] for k, _ in labs],
           [arch[k]["ci95"][1] for k, _ in labs], "#c9752f", floor=p1lib.PROTOCOL_CONSTANTS["vnf_floor"][0], xlabel="interaction in Y")
    axes[1].set_ylim(-1.5, 1.6)
    axes[1].set_title("B. Visible × nullspace (VNF, 32 triplets)")
    fig.text(0.01, -0.03, "Archival values from the original console output; dashed line: preregistered 0.005 floor.", fontsize=7.5, color="#444444")
    fig.tight_layout(w_pad=4.0)
    save(fig, "fig5_completeness_tests", [rpath], prov)


def main() -> int:
    prov: dict = {}
    figure1(prov)
    figure2(prov)
    figure3(prov)
    figure4(prov)
    figure5(prov)
    for name, entry in prov.items():
        entry["outputs_sha256"] = {o: p1lib.sha256_file(p1lib.PKG / o) for o in entry["outputs"]}
    env = {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}
    out = {"environment": env, "figures": prov}
    (FIG / "figure_provenance.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(prov)} figures to {FIG}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except p1lib.AnalysisError as exc:
        print(f"FIGURES FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
