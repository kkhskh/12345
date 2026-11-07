"""Analysis layer (B) for the Path 1 package. Reads only data/raw and data/archival.

Every quantity is recomputed from saved arrays: next-move logits (61 tokens), 64x3 probe
probabilities, 512-d intervention directions, the frozen PCA/ridge map, and simulator legal sets.
Nothing here imports the research repository.

Activations named in this package:
  h_R  natural recipient activation (cell "baseline" in SPEC, "unedited" in NST)
  h_E  Editor A reference activation: Editor A applied to h_R (cell "editor_A")
  h_E + t*delta  final nullspace-patched activation (cells correct_null, wrong_null_j, ...)
Probe invariance and the JS calibration are both measured against h_E.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
RAW = PKG / "data" / "raw"
ARCHIVAL = PKG / "data" / "archival"
ANALYSIS = PKG / "analysis"

CENTER_SQUARES = (27, 28, 35, 36)
JS_EPS = 1e-12
JS_TARGET = 0.01
JS_TOL = 2e-4
PROBE_ATOL_F32 = 1e-5
N_BOOT = 2000
BOOT_SEED = 82
RANK_TOL = 1e-8


# Preregistered constants used in figures; each is checked against its protocol text by
# scripts/verify_paper_claims.py.
PROTOCOL_CONSTANTS = {
    "fresh_floor": (0.005, "08r2_symmetric_target_fresh_preregistration.md", "Mean \\(A_T\\ge 0.005\\) and \\(A_W\\ge 0.005\\)"),
    "fresh_both_positive_gate": (0.60, "08r2_symmetric_target_fresh_preregistration.md", "both arms positive on ≥ 60% of complete triplets"),
    "vnf_floor": (0.005, "08r2_visible_null_factorial_preregistration.md", "Mean `I_T` and `I_W` ≥ 0.005"),
    "fixed_js_floor": (0.02, "08r2_fiber_cost_fixed_js_preregistration.md", "Mean \\(\\Delta_{\\mathrm{wrong}}\\ge 0.02\\)"),
    "nst_floor": (0.01, "08r2_null_state_transfer_preregistration.md", "Mean \\(Y_{\\mathrm{learned}}-Y_{\\mathrm{wrong}}\\ge 0.01\\)"),
    "spec_floor": (0.05, "08r2_fiber_spec_preregistration.md", "Mean \\(\\Delta_{\\mathrm{specific}}\\ge 0.05\\)"),
    "cost_floor": (0.05, "08r2_fiber_cost_preregistration.md", "Mean \\(\\Delta_{\\mathrm{JS}}\\ge 0.05\\)"),
    "sem_floor": (0.01, "08r2_state_semantics_preregistration.md", "Incremental advantage \\(+0.01\\) with paired-bootstrap CI lower bound"),
    "fresh_coverage": (29, "08r2_symmetric_target_fresh_preregistration.md", "Coverage ≥ 29/32"),
}


class AnalysisError(RuntimeError):
    pass


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AnalysisError(msg)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- primitives

def softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _smooth(x, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) + eps
    return x / x.sum()


def _kl(a, b, eps: float) -> float:
    a, b = _smooth(a, eps), _smooth(b, eps)
    return float(np.sum(a * (np.log(a) - np.log(b))))


def js_divergence(p, q, eps: float = JS_EPS) -> float:
    """Jensen-Shannon divergence in nats, with the original runs' smoothing.

    p and q are smoothed (add eps, renormalize), and each KL term smooths its two arguments
    again, exactly as the experiment code did when it calibrated to JS = 0.01.
    """
    p, q = _smooth(p, eps), _smooth(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps)


def total_variation(p, q) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def square_to_token(square: int) -> int:
    """Token 0 is pass; tokens 1..60 index the 60 non-center squares in row-major order."""
    s = int(square)
    require(0 <= s < 64, f"square out of range: {s}")
    require(s not in CENTER_SQUARES, f"center square has no move token: {s}")
    return s + 1 - sum(c < s for c in CENTER_SQUARES)


def y_components(probs, became_legal_squares, became_illegal_squares) -> dict:
    p = np.asarray(probs, dtype=np.float64)
    require(p.shape == (61,), f"expected 61 move probabilities, got {p.shape}")
    gl = [square_to_token(s) for s in became_legal_squares]
    il = [square_to_token(s) for s in became_illegal_squares]
    require(len(gl) > 0 and len(il) > 0, "Y requires nonempty became-legal and became-illegal sets")
    G = float(p[gl].sum())
    Q = float(p[il].sum())
    return {"G": G, "Q": Q, "Y": G - Q}


def cluster_bootstrap(values, clusters, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Percentile bootstrap over clusters (mean of cluster means).

    Clusters are taken in order of first appearance and resampled with
    numpy.random.default_rng(seed).choice(..., replace=True), as in the original runs.
    """
    vals = np.asarray(values, dtype=np.float64)
    require(vals.size == len(clusters) and vals.size > 0, "bootstrap input size mismatch or empty")
    require(np.all(np.isfinite(vals)), "bootstrap input contains non-finite values")
    by: dict[str, list[float]] = {}
    for v, c in zip(vals, clusters):
        by.setdefault(str(c), []).append(float(v))
    keys = list(by.keys())
    cm = np.array([np.mean(by[k]) for k in keys])
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(keys))
    means = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        means[b] = cm[rng.choice(idx, size=len(keys), replace=True)].mean()
    return {
        "n": int(vals.size), "n_clusters": len(keys), "point": float(vals.mean()),
        "boot_mean": float(means.mean()), "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)), "n_boot": int(n_boot), "seed": int(seed),
    }


def read_csv(path: Path) -> list[dict]:
    require(Path(path).is_file(), f"missing raw file: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    require(Path(path).is_file(), f"missing raw file: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict:
    require(Path(path).is_file(), f"missing raw file: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fnum(x) -> float:
    return float(x) if x not in (None, "") else float("nan")


def verify_raw_hashes() -> dict:
    """Check every shipped raw file against data/provenance_manifest.json."""
    man = read_json(PKG / "data" / "provenance_manifest.json")
    n = 0
    for study, entry in man["archives"].items():
        for m in entry["members"]:
            p = PKG / m["package_path"]
            require(p.is_file(), f"raw file listed in provenance manifest is missing: {m['package_path']}")
            got = sha256_file(p)
            require(got == m["sha256"], f"raw file hash mismatch: {m['package_path']} {got} != {m['sha256']}")
            n += 1
    for item in man["console_excerpts"] + man["protocols"]:
        p = PKG / item["package_path"]
        require(p.is_file(), f"missing provenance file: {item['package_path']}")
        require(sha256_file(p) == item["sha256"], f"hash mismatch: {item['package_path']}")
        n += 1
    return {"files_checked": n, "ok": 1}


def _slice_stats(rows: list[dict], metrics: dict[str, tuple[str, int]], slice_offset: int = 0) -> dict:
    """Seeds follow the original runners: BOOT_SEED + slice_offset + metric offset.

    SPEC and fixed-JS used slice_offset 0 for every slice; NST used 0/10/20/30/40 for
    pooled/A/B/exact/nonexact.
    """
    out = {"n": len(rows)}
    clusters = [r["recip_game"] for r in rows]
    for name, (col, seed_offset) in metrics.items():
        vals = [r[col] for r in rows]
        out[name] = cluster_bootstrap(vals, clusters, N_BOOT, BOOT_SEED + slice_offset + seed_offset)
        out[name]["wins"] = int(sum(v > 0 for v in vals))
        out[name]["median"] = float(np.median(vals))
    return out


# ---------------------------------------------------------------- FIBER geometry

def recompute_fiber() -> dict:
    d = RAW / "fiber"
    svd = read_json(d / "08r2_fiber_svd_report.json")
    s = np.asarray(svd["singular_values"], dtype=np.float64)
    m, n = svd["shape"]
    rank = int(np.sum(s > RANK_TOL * s.max()))
    pn = load_npz(RAW / "nst" / "null_state_pca_ridge.npz")["P_N"]
    ev = np.linalg.eigvalsh(0.5 * (pn + pn.T))
    geom = {
        "contrast_rows": int(m), "d_model": int(n), "rank_from_singular_values": rank,
        "nullspace_dim": int(n - rank), "sigma_max": float(s.max()), "sigma_min_kept": float(s[s > RANK_TOL * s.max()].min()),
        "P_N_trace": float(np.trace(pn)), "P_N_rank_eig_gt_half": int(np.sum(ev > 0.5)),
        "P_N_idempotence_err": float(np.abs(pn @ pn - pn).max()), "P_N_symmetry_err": float(np.abs(pn - pn.T).max()),
        "P_N_eig_dist_from_0_or_1": float(np.minimum(np.abs(ev), np.abs(ev - 1)).max()),
        "source_singular_values": "data/raw/fiber/08r2_fiber_svd_report.json", "source_P_N": "data/raw/nst/null_state_pca_ridge.npz",
    }
    L = load_npz(d / "08r2_fiber_full_logits.npz")
    P = load_npz(d / "08r2_fiber_probe_probabilities.npz")
    pairs = read_csv(d / "08r2_fiber_matched_pairs.csv")
    target_counts = {}
    for r in pairs:
        target_counts[r["target_game"]] = target_counts.get(r["target_game"], 0) + 1
    rows = []
    for r in pairs:
        k = f"{r['recip_game']}__{r['target_game']}"
        for cell in ("H00", "H01", "H10", "H11"):
            require(f"{k}__{cell}" in L, f"FIBER missing logits {k}__{cell}")
        p00, p01 = softmax(L[f"{k}__H00"]), softmax(L[f"{k}__H01"])
        rows.append({
            "recip_game": r["recip_game"], "target_game": r["target_game"], "r2_type": r["r2_type"],
            "probe_max_abs_H01_vs_H00": float(np.abs(P[f"{k}__H01"] - P[f"{k}__H00"]).max()),
            "js_H00_H01": js_divergence(p00, p01),
            "js_H10_H11": js_divergence(softmax(L[f"{k}__H10"]), softmax(L[f"{k}__H11"])),
            "archived_js_00_01": fnum(r["js_00_01"]),
        })
    probe = np.array([x["probe_max_abs_H01_vs_H00"] for x in rows])
    js = np.array([x["js_H00_H01"] for x in rows])
    demo = int(np.sum((probe <= PROBE_ATOL_F32) & (js >= 1e-3)))
    anchor = read_csv(d / "08r2_fiber_natural_anchor.csv")
    def anchor_mean(family: str, eps: float, col: str, absolute: bool) -> float:
        xs = [float(r[col]) for r in anchor if r["family"] == family and float(r["eps"]) == eps]
        require(len(xs) > 0, f"natural-anchor rows missing: {family} eps={eps}")
        return float(np.mean(np.abs(xs) if absolute else xs))
    fact = read_csv(d / "08r2_fiber_factorial_interventions.csv")
    y_cell = {}
    for cell in ("H00", "H10", "H01", "H11"):
        ys = [float(r["Y"]) for r in fact if r["cell"] == cell]
        require(len(ys) == 32, f"FIBER factorial cell {cell} has {len(ys)} rows")
        y_cell[cell] = float(np.mean(ys))
    by_pair = {}
    for r in fact:
        by_pair.setdefault((r["recip_game"], r["target_game"]), {})[r["cell"]] = float(r["Y"])
    interaction = [v["H11"] - v["H10"] - v["H01"] + v["H00"] for v in by_pair.values()]
    require(len(interaction) == 32, f"FIBER factorial pairs: {len(interaction)}")
    verified = [float(r["I_recomputed"]) for r in read_csv(d / "08r2_fiber_specificity_pair_verification.csv")]
    require(abs(float(np.mean(interaction)) - float(np.mean(verified))) < 1e-9, "factorial interaction disagrees with the saved pair verification")
    return {
        "geometry": geom,
        "pairs": {
            "n_pairs": len(rows), "n_A": sum(x["r2_type"].startswith("A") for x in rows), "n_B": sum(x["r2_type"].startswith("B") for x in rows),
            "probe_max_abs_max": float(probe.max()), "mean_js_H00_H01": float(js.mean()), "min_js_H00_H01": float(js.min()),
            "n_probe_invariant_and_js_ge_0.001": demo,
            "max_pairs_one_target": int(max(target_counts.values())),
            "max_abs_js_vs_archive": float(np.max(np.abs(js - np.array([x["archived_js_00_01"] for x in rows])))),
        },
        "natural_anchor": {
            "evidence": "means of saved per-row scores; the natural-anchor logits are not in the archive",
            "donor_abs_dY_l2_1": anchor_mean("donor_null_direction", 1.0, "dY_vs_donor", True),
            "random_abs_dY_l2_1": anchor_mean("random_nullspace", 1.0, "dY_vs_donor", True),
            "donor_abs_dY_l2_4": anchor_mean("donor_null_direction", 4.0, "dY_vs_donor", True),
            "donor_js_l2_4": anchor_mean("donor_null_direction", 4.0, "js_vs_donor", False),
        },
        "factorial": {
            "evidence": "means of saved per-cell Y; H00 is the Editor A reference on the FIBER pairs",
            "mean_Y": y_cell,
            "mean_I": float(np.mean(interaction)),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------- shared pair metadata

def _spec_legal() -> dict:
    return read_json(RAW / "spec" / "fiber_spec_target_legal_sets.json")


def _pair_order(rows: list[dict]) -> list[tuple[str, str]]:
    seen, order = set(), []
    for r in rows:
        key = (r["recip_game"], r["target_game"])
        if r.get("recip_game") and key not in seen:
            seen.add(key)
            order.append(key)
    return order


# ---------------------------------------------------------------- SPEC (L2-matched)

def recompute_spec() -> dict:
    d = RAW / "spec"
    L = load_npz(d / "fiber_spec_full_logits.npz")
    P = load_npz(d / "fiber_spec_full_probe_probabilities.npz")
    V = load_npz(d / "fiber_spec_nullspace_vectors.npz")
    legal = _spec_legal()
    csvrows = read_csv(d / "fiber_spec_interventions.csv")
    meta = {(r["recip_game"], r["target_game"]): r for r in csvrows if r["cell"] == "editor_A"}
    order = [k for k in _pair_order([r for r in csvrows if r["cell"] == "pair_contrast"])]
    require(len(order) == 48, f"SPEC expected 48 pair-contrast rows, found {len(order)}")
    pairs, cells_y, js_by = [], {c: [] for c in ("editor_A", "baseline", "correct_null", "correct_rowspace", "full_donor", "wrong_null", "random_null")}, {c: [] for c in ("correct_null", "wrong_null", "random_null")}
    probe_max = 0.0
    l2 = {"correct": [], "wrong": [], "random": []}
    for rg, tg in order:
        k = f"{rg}__{tg}"
        e = legal[f"{rg}|{tg}"]
        ref = softmax(L[f"{k}__editor_A"])
        def Y(cell):
            return y_components(softmax(L[f"{k}__{cell}"]), e["became_legal"], e["became_illegal"])
        yc = Y("correct_null")
        yw = [Y(f"wrong_null_{j}") for j in range(3)]
        yr = [Y(f"random_null_{j}") for j in range(3)]
        cells_y["editor_A"].append(Y("editor_A")["Y"])
        cells_y["baseline"].append(Y("baseline")["Y"])
        cells_y["correct_null"].append(yc["Y"])
        cells_y["correct_rowspace"].append(Y("correct_rowspace")["Y"])
        cells_y["full_donor"].append(Y("full_donor")["Y"])
        cells_y["wrong_null"].append(float(np.mean([y["Y"] for y in yw])))
        cells_y["random_null"].append(float(np.mean([y["Y"] for y in yr])))
        js_by["correct_null"].append(js_divergence(ref, softmax(L[f"{k}__correct_null"])))
        js_by["wrong_null"].append(float(np.mean([js_divergence(ref, softmax(L[f"{k}__wrong_null_{j}"])) for j in range(3)])))
        js_by["random_null"].append(float(np.mean([js_divergence(ref, softmax(L[f"{k}__random_null_{j}"])) for j in range(3)])))
        for c in ["correct_null"] + [f"wrong_null_{j}" for j in range(3)] + [f"random_null_{j}" for j in range(3)]:
            probe_max = max(probe_max, float(np.abs(P[f"{k}__{c}"] - P[f"{k}__editor_A"]).max()))
        l2["correct"].append(float(np.linalg.norm(V[f"{k}__correct"])))
        l2["wrong"] += [float(np.linalg.norm(V[f"{k}__wrong_{j}"])) for j in range(3)]
        l2["random"] += [float(np.linalg.norm(V[f"{k}__random_{j}"])) for j in range(3)]
        pairs.append({
            "recip_game": rg, "target_game": tg, "r2_type": meta[(rg, tg)]["r2_type"],
            "delta_specific": yc["Y"] - float(np.mean([y["Y"] for y in yw])),
            "delta_random": yc["Y"] - float(np.mean([y["Y"] for y in yr])),
        })
    ab = {"pooled": pairs, "A": [p for p in pairs if p["r2_type"].startswith("A")], "B": [p for p in pairs if p["r2_type"].startswith("B")]}
    slices = {name: _slice_stats(rows, {"delta_specific": ("delta_specific", 0), "delta_random": ("delta_random", 1)}) for name, rows in ab.items()}
    return {
        "slices": slices,
        "mean_Y": {c: float(np.mean(v)) for c, v in cells_y.items()},
        "mean_js_vs_editor_A": {c: float(np.mean(v)) for c, v in js_by.items()},
        "mean_l2": {c: float(np.mean(v)) for c, v in l2.items()},
        "max_rel_l2_mismatch_wrong_random_vs_correct": float(max(
            abs(np.linalg.norm(V[f"{rg}__{tg}__{c}"]) / np.linalg.norm(V[f"{rg}__{tg}__correct"]) - 1)
            for rg, tg in order for c in [f"wrong_{j}" for j in range(3)] + [f"random_{j}" for j in range(3)])),
        "probe_max_abs_vs_editor_A": probe_max,
        "wrong_donor_linked_pairs": _wrong_donor_link(order)[0],
        "wrong_donor_largest_component": _wrong_donor_link(order)[1],
        "rows": pairs,
    }


def _wrong_donor_link(order: list[tuple[str, str]]) -> tuple[int, int]:
    """Pairs joined when they use the same wrong-donor game. Returns (pairs in a non-singleton component, largest component)."""
    manifest = read_json(RAW / "spec" / "fiber_spec_pair_manifest.json")
    by_pair = {(p["recip_game"], p["target_game"]): w for p, w in zip(manifest["pairs"], manifest["wrong_donors"])}
    parent = {i: i for i in range(len(order))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners: dict[str, int] = {}
    for i, key in enumerate(order):
        for donor in by_pair[key]["donors"]:
            j = owners.setdefault(donor["game_id"], i)
            parent[find(i)] = find(j)
    sizes: dict[int, int] = {}
    for i in range(len(order)):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    linked = sum(n for n in sizes.values() if n > 1)
    return linked, max(sizes.values())


# ---------------------------------------------------------------- fixed-JS (restored COST)

FJ_CELLS = ["correct_null"] + [f"wrong_null_{j}" for j in range(3)] + [f"random_null_{j}" for j in range(3)]


def _category(cell: str) -> str:
    return cell.split("_")[0]


def recompute_fixed_js() -> dict:
    d = RAW / "fixed_js"
    L = load_npz(d / "fixed_js_full_logits.npz")
    P = load_npz(d / "fixed_js_probe_probabilities.npz")
    SL = load_npz(RAW / "spec" / "fiber_spec_full_logits.npz")
    SP = load_npz(RAW / "spec" / "fiber_spec_full_probe_probabilities.npz")
    V = load_npz(RAW / "spec" / "fiber_spec_nullspace_vectors.npz")
    legal = _spec_legal()
    ivs = read_csv(d / "fixed_js_interventions.csv")
    cal = {(r["recip_game"], r["target_game"], r["cell"]): r for r in read_csv(d / "fixed_js_calibrations.csv")}
    meta = {(r["recip_game"], r["target_game"]): r for r in ivs if r["cell"] == "editor_A"}
    archived = {(r["recip_game"], r["target_game"], r["cell"]): r for r in ivs if r["cell"] in FJ_CELLS}
    order = _pair_order([r for r in ivs if r["cell"] == "editor_A"])
    require(len(order) == 48, f"fixed-JS expected 48 pairs, found {len(order)}")

    inter, pairs = [], []
    editor_identity = 0.0
    for rg, tg in order:
        k = f"{rg}__{tg}"
        e = legal[f"{rg}|{tg}"]
        ref_logits = L[f"{k}__editor_A"]
        editor_identity = max(editor_identity, float(np.abs(ref_logits - SL[f"{k}__editor_A"]).max()))
        p_E = softmax(ref_logits)
        p_R = softmax(SL[f"{k}__baseline"])
        probe_E = P[f"{k}__editor_A"]
        per = {}
        for cell in FJ_CELLS:
            key = f"{k}__{cell}"
            require(key in L and key in P, f"fixed-JS missing arrays for {key}")
            p = softmax(L[key])
            yc = y_components(p, e["became_legal"], e["became_illegal"])
            c = cal[(rg, tg, cell)]
            require(c["ok"] == "1", f"fixed-JS calibration not ok for {key}")
            vkey = f"{k}__" + cell.replace("_null", "")
            t = float(c["t"])
            row = {
                "recip_game": rg, "target_game": tg, "r2_type": meta[(rg, tg)]["r2_type"], "cell": cell, "category": _category(cell),
                "t": t, "l2": t * float(np.linalg.norm(V[vkey])), "archived_l2": fnum(c["l2"]),
                "js_vs_editor_A": js_divergence(p_E, p), "js_vs_natural_recipient": js_divergence(p_R, p),
                "tv_vs_editor_A": total_variation(p_E, p),
                "probe_max_abs_vs_editor_A": float(np.abs(P[key] - probe_E).max()),
                "probe_max_abs_vs_natural_recipient": float(np.abs(P[key] - SP[f"{k}__baseline"]).max()),
                **yc,
                "archived_Y": fnum(archived[(rg, tg, cell)]["Y"]), "archived_js": fnum(archived[(rg, tg, cell)]["js"]),
            }
            inter.append(row)
            per[cell] = row
        yw = [per[f"wrong_null_{j}"] for j in range(3)]
        yr = [per[f"random_null_{j}"] for j in range(3)]
        cc = per["correct_null"]
        pairs.append({
            "recip_game": rg, "target_game": tg, "r2_type": meta[(rg, tg)]["r2_type"],
            "board_exact_editor": int(float(meta[(rg, tg)]["board_exact_editor"]) == 1.0),
            "Y_correct": cc["Y"], "Y_wrong_mean": float(np.mean([r["Y"] for r in yw])), "Y_random_mean": float(np.mean([r["Y"] for r in yr])),
            "delta_specific": cc["Y"] - float(np.mean([r["Y"] for r in yw])),
            "delta_random": cc["Y"] - float(np.mean([r["Y"] for r in yr])),
            "dG_wrong": cc["G"] - float(np.mean([r["G"] for r in yw])),
            "dQ_wrong": cc["Q"] - float(np.mean([r["Q"] for r in yw])),
            "probe_max_correct": cc["probe_max_abs_vs_editor_A"],
            "probe_max_wrong": max(r["probe_max_abs_vs_editor_A"] for r in yw),
            "probe_max_random": max(r["probe_max_abs_vs_editor_A"] for r in yr),
            "tv_correct": cc["tv_vs_editor_A"],
            "tv_wrong_mean": float(np.mean([r["tv_vs_editor_A"] for r in yw])),
            "tv_random_mean": float(np.mean([r["tv_vs_editor_A"] for r in yr])),
            "l2_correct": cc["l2"], "l2_wrong_mean": float(np.mean([r["l2"] for r in yw])), "l2_random_mean": float(np.mean([r["l2"] for r in yr])),
        })

    def sl(rows):
        return _slice_stats(rows, {"delta_specific": ("delta_specific", 0), "delta_random": ("delta_random", 1)})
    slices = {
        "pooled": sl(pairs),
        "A": sl([p for p in pairs if p["r2_type"].startswith("A")]),
        "B": sl([p for p in pairs if p["r2_type"].startswith("B")]),
        "exact_editor": sl([p for p in pairs if p["board_exact_editor"]]),
        "nonexact_editor": sl([p for p in pairs if not p["board_exact_editor"]]),
    }
    arr = lambda col, cat=None: np.array([r[col] for r in inter if cat is None or r["category"] == cat])
    summary = {
        "n_pairs": len(pairs), "n_interventions": len(inter),
        "editor_A_logits_identical_to_spec": editor_identity,
        "max_abs_js_minus_target": float(np.max(np.abs(arr("js_vs_editor_A") - JS_TARGET))),
        "max_abs_js_vs_archive": float(np.max(np.abs(arr("js_vs_editor_A") - arr("archived_js")))),
        "max_abs_Y_vs_archive": float(np.max(np.abs(arr("Y") - arr("archived_Y")))),
        "max_rel_l2_vs_archive": float(np.max(np.abs(arr("l2") / arr("archived_l2") - 1))),
        "probe_max_abs_vs_editor_A": float(arr("probe_max_abs_vs_editor_A").max()),
        "probe_max_abs_vs_natural_recipient_min": float(arr("probe_max_abs_vs_natural_recipient").min()),
        "mean_js_vs_natural_recipient": float(arr("js_vs_natural_recipient").mean()),
        "mean_Y": {c: float(arr("Y", c).mean()) for c in ("correct", "wrong", "random")},
        "mean_l2": {c: float(arr("l2", c).mean()) for c in ("correct", "wrong", "random")},
        "mean_tv_vs_editor_A": {c: float(arr("tv_vs_editor_A", c).mean()) for c in ("correct", "wrong", "random")},
        "pair_mean_tv_vs_editor_A": {
            "correct": float(np.mean([p["tv_correct"] for p in pairs])),
            "wrong": float(np.mean([p["tv_wrong_mean"] for p in pairs])),
            "random": float(np.mean([p["tv_random_mean"] for p in pairs])),
        },
        "mean_dG_wrong": float(np.mean([p["dG_wrong"] for p in pairs])),
        "mean_dQ_wrong": float(np.mean([p["dQ_wrong"] for p in pairs])),
        "n_A": sum(p["r2_type"].startswith("A") for p in pairs), "n_B": sum(p["r2_type"].startswith("B") for p in pairs),
        "n_exact_editor": sum(p["board_exact_editor"] for p in pairs),
    }
    return {"slices": slices, "summary": summary, "pairs": pairs, "interventions": inter}


# ---------------------------------------------------------------- NST (donor-free map)

NST_CELLS = ("learned", "wrong_target", "mean_state", "random", "oracle", "gradient")


def symbolic_features(state: dict) -> np.ndarray:
    rel = np.asarray(state["rel_lab"], dtype=np.int64)
    require(rel.size == 64 and rel.min() >= 0 and rel.max() <= 2, "rel_lab must be 64 entries in {0,1,2}")
    onehot = np.eye(3)[rel].reshape(-1)
    legal = np.zeros(64)
    legal[list(map(int, state["legal_squares"]))] = 1.0
    player = 1.0 if int(state["player"]) == 1 else 0.0
    return np.concatenate([onehot, legal, [player], [int(state["occupied"]) / 64.0]])


def recompute_nst() -> dict:
    d = RAW / "nst"
    L = load_npz(d / "null_state_full_logits.npz")
    P = load_npz(d / "null_state_probe_probabilities.npz")
    D = load_npz(d / "null_state_intervention_directions.npz")
    LD = load_npz(d / "null_state_learned_directions.npz")
    M = load_npz(d / "null_state_pca_ridge.npz")
    names = read_json(d / "null_state_feature_names.json")["names"]
    require(len(names) == M["W"].shape[0] == 258, "feature dimension mismatch with ridge map")
    stage_means = read_json(d / "null_state_stage_means.json")
    legal = read_json(d / "null_state_target_legal_sets.json")
    ivs = read_csv(d / "null_state_interventions.csv")
    meta = {(r["recip_game"], r["target_game"]): r for r in ivs if r["cell"] == "pair_contrast"}
    order = _pair_order([r for r in ivs if r["cell"] == "pair_contrast"])
    require(len(order) == 48, f"NST expected 48 pairs, found {len(order)}")
    PN = M["P_N"]

    def zhat(b):
        return PN @ (M["mu"] + (b @ M["W"] + M["c"]) @ M["V"].T)

    pairs, js_err, probe_max, map_err, mean_err, ld_err = [], 0.0, 0.0, 0.0, 0.0, 0.0
    cos_lo, cos_lm, cos_lw, js_R, js_RE = [], [], [], [], []
    gradient_n = 0
    for rg, tg in order:
        k = f"{rg}__{tg}"
        e = legal[f"{rg}|{tg}"]
        bl, bi = e["became_legal_squares"], e["became_illegal_squares"]
        pE = softmax(L[f"{k}__editor_A"])
        pR = softmax(L[f"{k}__unedited"])
        ys = {}
        for cell in NST_CELLS + ("editor_A", "unedited"):
            key = f"{k}__{cell}"
            if key not in L:
                require(cell == "gradient", f"NST missing logits for {key}")
                continue
            p = softmax(L[key])
            ys[cell] = y_components(p, bl, bi)
            if cell in NST_CELLS:
                js_err = max(js_err, abs(js_divergence(pE, p) - JS_TARGET))
                probe_max = max(probe_max, float(np.abs(P[key] - P[f"{k}__editor_A"]).max()))
        gradient_n += int("gradient" in ys)
        js_R.append(js_divergence(pR, softmax(L[f"{k}__learned"])))
        js_RE.append(js_divergence(pR, pE))
        dl, dw, dm, do = D[f"{k}__learned"], D[f"{k}__wrong_target"], D[f"{k}__mean_state"], D[f"{k}__oracle"]
        bt, bw = symbolic_features(e["target"]), symbolic_features(e["wrong_target"])
        pred = zhat(bt) - zhat(bw)
        map_err = max(map_err, float(np.abs((dl - dw) - pred).max() / np.abs(pred).max()))
        skey = f"{int(e['target']['player'])}_{int(e['target']['occupied']) // 4}"
        require(skey in stage_means, f"stage mean missing for {skey}")
        pred_m = zhat(bt) - PN @ np.asarray(stage_means[skey], dtype=np.float64)
        mean_err = max(mean_err, float(np.abs((dl - dm) - pred_m).max() / np.abs(pred_m).max()))
        ld_err = max(ld_err, float(np.abs(LD[k] - dl).max()))
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        cos_lo.append(cos(dl, do)); cos_lm.append(cos(dl, dm)); cos_lw.append(cos(dl, dw))
        m = meta[(rg, tg)]
        pairs.append({
            "recip_game": rg, "target_game": tg, "r2_type": m["r2_type"],
            "board_exact_editor": int(float(m["board_exact_editor"]) == 1.0),
            **{f"Y_{c}": (ys[c]["Y"] if c in ys else float("nan")) for c in NST_CELLS + ("editor_A", "unedited")},
            "G_learned": ys["learned"]["G"], "Q_learned": ys["learned"]["Q"], "G_editor_A": ys["editor_A"]["G"], "Q_editor_A": ys["editor_A"]["Q"],
            "delta_wrong": ys["learned"]["Y"] - ys["wrong_target"]["Y"],
            "delta_random": ys["learned"]["Y"] - ys["random"]["Y"],
            "delta_mean": ys["learned"]["Y"] - ys["mean_state"]["Y"],
            "delta_editor": ys["learned"]["Y"] - ys["editor_A"]["Y"],
            "cos_learned_oracle": cos_lo[-1], "cos_learned_mean_state": cos_lm[-1], "cos_learned_wrong": cos_lw[-1],
            "archived_delta_wrong": fnum(m["delta_learned_wrong"]),
        })
    metrics = {"wrong": ("delta_wrong", 0), "random": ("delta_random", 1), "mean": ("delta_mean", 2), "editor": ("delta_editor", 3)}
    slices = {
        "pooled": _slice_stats(pairs, metrics, 0),
        "A": _slice_stats([p for p in pairs if p["r2_type"].startswith("A")], metrics, 10),
        "B": _slice_stats([p for p in pairs if p["r2_type"].startswith("B")], metrics, 20),
        "exact": _slice_stats([p for p in pairs if p["board_exact_editor"]], metrics, 30),
        "nonexact": _slice_stats([p for p in pairs if not p["board_exact_editor"]], metrics, 40),
    }
    col = lambda c: np.array([p[c] for p in pairs])
    corr = lambda a, b: float(np.corrcoef(col(a), col(b))[0, 1])
    corr_resid = lambda a, b: float(np.corrcoef(col(a) - col("Y_editor_A"), col(b) - col("Y_editor_A"))[0, 1])
    diag = read_json(d / "null_state_train_diagnostics.json")
    summary = {
        "n_pairs": len(pairs), "gradient_reached_tolerance": gradient_n,
        "max_abs_js_minus_target": js_err, "probe_max_abs_vs_editor_A": probe_max,
        "learned_minus_wrong_vs_map_rel_err": map_err, "learned_minus_mean_state_vs_map_rel_err": mean_err,
        "learned_directions_file_vs_intervention_file": ld_err,
        "mean_Y": {c: float(np.nanmean(col(f"Y_{c}"))) for c in NST_CELLS + ("editor_A", "unedited")},
        "mean_dG_vs_editor_A": float(np.mean(col("G_learned") - col("G_editor_A"))),
        "mean_dQ_vs_editor_A": float(np.mean(col("Q_learned") - col("Q_editor_A"))),
        "n_Y_learned_positive": int(np.sum(col("Y_learned") > 0)),
        "mean_cos_learned_oracle": float(np.mean(cos_lo)), "mean_cos_learned_mean_state": float(np.mean(cos_lm)), "mean_cos_learned_wrong": float(np.mean(cos_lw)),
        "pearson_Y_learned_oracle": corr("Y_learned", "Y_oracle"),
        "pearson_Y_learned_editor_A": corr("Y_learned", "Y_editor_A"),
        "pearson_Y_learned_wrong": corr("Y_learned", "Y_wrong_target"),
        "pearson_Y_learned_unedited": corr("Y_learned", "Y_unedited"),
        "pearson_residual_learned_oracle": corr_resid("Y_learned", "Y_oracle"),
        "pearson_residual_learned_wrong": corr_resid("Y_learned", "Y_wrong_target"),
        "mean_js_learned_vs_natural_recipient": float(np.mean(js_R)), "mean_js_editor_A_vs_natural_recipient": float(np.mean(js_RE)),
        "max_abs_delta_wrong_vs_archive": float(np.max(np.abs(col("delta_wrong") - col("archived_delta_wrong")))),
        "n_exact_editor": int(col("board_exact_editor").sum()),
        "random_minus_editor": float(np.mean(col("Y_random") - col("Y_editor_A"))),
        "random_minus_editor_wins": int(np.sum(col("Y_random") > col("Y_editor_A"))),
        "recorded_train_r2": float(diag["train_r2"]), "recorded_heldout_r2": float(diag["heldout_unused_val_residual"]["r2"]),
        "recorded_cv_k": int(diag["best"]["k"]), "recorded_cv_alpha": float(diag["best"]["alpha"]),
        "map_k": int(M["k"]), "map_alpha": float(M["alpha"]),
    }
    return {"slices": slices, "summary": summary, "pairs": pairs}


# ---------------------------------------------------------------- archival + mixed

def load_archival() -> dict:
    a = read_json(ARCHIVAL / "archival_values.json")
    out = {}
    for v in a["values"]:
        ex = (ARCHIVAL / "console_excerpts" / v["excerpt"]).read_text(encoding="utf-8")
        require(v["excerpt_contains"] in ex, f"archival value {v['id']} not found verbatim in {v['excerpt']}")
        out[v["id"]] = v
    return out


def stage_share(nst: dict, arch: dict) -> dict:
    """Mean-level arithmetic: STAGE-EditorA = (FULL-EditorA) - (FULL-STAGE)."""
    full_editor = nst["slices"]["pooled"]["editor"]["point"]
    via_legal = arch["sem.full_minus_legal"]["value"] + arch["sem.legal_minus_stage"]["value"]
    via_board = arch["sem.full_minus_board"]["value"] + arch["sem.board_minus_stage"]["value"]
    stage_editor = full_editor - via_legal
    return {
        "full_minus_editor_recomputed": full_editor,
        "full_minus_editor_console": arch["sem.full_minus_editor"]["value"],
        "full_minus_stage_via_legal": via_legal, "full_minus_stage_via_board": via_board,
        "decomposition_gap": abs(via_legal - via_board),
        "stage_minus_editor": stage_editor, "share": stage_editor / full_editor,
        "evidence": "FULL-EditorA recomputed from NST raw arrays; FULL-STAGE from SEM console values (archival).",
    }


def compute_all() -> dict:
    raw = verify_raw_hashes()
    fiber = recompute_fiber()
    spec = recompute_spec()
    fj = recompute_fixed_js()
    nst = recompute_nst()
    arch = load_archival()
    share = stage_share(nst, arch)
    spec_delta = {(r["recip_game"], r["target_game"]): r["delta_specific"] for r in spec["rows"]}
    both = [((spec_delta[k], r["delta_specific"])) for r in fj["pairs"] if (k := (r["recip_game"], r["target_game"])) in spec_delta]
    fj["summary"]["pearson_delta_specific_vs_spec"] = float(np.corrcoef([a for a, _ in both], [b for _, b in both])[0, 1])
    orig = {
        "spec": read_csv(RAW / "spec" / "fiber_spec_bootstrap.csv"),
        "fixed_js": read_csv(RAW / "fixed_js" / "fixed_js_bootstrap.csv"),
        "nst": read_csv(RAW / "nst" / "null_state_bootstrap.csv"),
    }
    letters = {
        "spec": read_json(RAW / "spec" / "fiber_spec_completion_status.json")["verdict"],
        "fixed_js": read_json(RAW / "fixed_js" / "fixed_js_completion_status.json")["verdict"],
        "nst": read_json(RAW / "nst" / "null_state_completion_status.json")["verdict"],
    }
    return {"raw_check": raw, "fiber": fiber, "spec": spec, "fixed_js": fj, "nst": nst, "stage_share": share,
            "archival": {k: {kk: vv for kk, vv in v.items() if kk != "excerpt_contains"} for k, v in arch.items()},
            "original_run_bootstrap": orig, "raw_letters": letters}


def strip_rows(res: dict) -> dict:
    """Summary view without per-row lists (rows are written to CSV separately)."""
    out = json.loads(json.dumps(res, default=float))
    out["fiber"].pop("rows", None)
    out["spec"].pop("rows", None)
    out["fixed_js"].pop("pairs", None)
    out["fixed_js"].pop("interventions", None)
    out["nst"].pop("pairs", None)
    return out


def get_path(d: dict, dotted: str):
    cur = d
    for part in dotted.split("/"):
        require(isinstance(cur, dict) and part in cur, f"missing analysis key: {dotted}")
        cur = cur[part]
    return cur


def is_close(a: float, b: float, tol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol
