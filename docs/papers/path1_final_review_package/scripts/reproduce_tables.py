"""Layer B -> C: build tables/ from analysis/results.json and data/archival only.

No number in this file is a claimed result. Every cell is read from the analysis output or the
archival register, then formatted. Also writes audits/recomputation_report.md.

Usage:
    python scripts/reproduce_tables.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1lib  # noqa: E402

TABLES = p1lib.PKG / "tables"
AUDITS = p1lib.PKG / "audits"


def signed(x: float, nd: int = 4) -> str:
    s = f"{x:+.{nd}f}"
    return s.replace("-", "\u2212")


def plain(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}".replace("-", "\u2212")


def interval(lo: float, hi: float, nd: int = 4) -> str:
    return f"[{plain(lo, nd)}, {plain(hi, nd)}]"


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"{m}e\u2212{abs(int(e))}" if int(e) < 0 else f"{m}e{int(e)}"


def write(name: str, header: list[str], rows: list[list]) -> Path:
    p1lib.require(len(rows) > 0, f"table {name} would be empty")
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def load() -> tuple[dict, dict]:
    res_path = p1lib.ANALYSIS / "results.json"
    p1lib.require(res_path.is_file(), "analysis/results.json missing; run scripts/recompute_analysis.py first")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    arch = {k: v for k, v in res["archival"].items()}
    return res, arch


def main() -> int:
    res, arch = load()
    fj, spec, nst = res["fixed_js"], res["spec"], res["nst"]
    g = p1lib.get_path

    def rec(study, label, unit, contrast, ci):
        return [label, unit, contrast, signed(ci["point"]), interval(ci["ci95_low"], ci["ci95_high"]), "Recomputed"]

    def archv(label, unit, contrast, key):
        v = arch[key]
        p1lib.require(v.get("ci95") is not None, f"archival value {key} lacks an interval")
        return [label, unit, contrast, signed(v["value"]), interval(*v["ci95"]), "Archival"]

    t1 = [
        rec("spec", "SPEC (matched L2)", "48 pairs", "correct − wrong", g(spec, "slices/pooled/delta_specific")),
        rec("fj", "Matched JS", "48 pairs", "correct − wrong", g(fj, "slices/pooled/delta_specific")),
        rec("fj", "Matched JS", "48 pairs", "correct − random", g(fj, "slices/pooled/delta_random")),
        rec("nst", "NST (donor-free)", "48 pairs", "learned − wrong", g(nst, "slices/pooled/wrong")),
        rec("nst", "NST (donor-free)", "48 pairs", "learned − Editor A", g(nst, "slices/pooled/editor")),
        archv("SEM ablation", "48 pairs", "FULL − LEGAL_ONLY", "sem.full_minus_legal"),
        archv("FRESH switching", "32 triplets", "A_T", "fresh.A_T"),
        archv("FRESH switching", "32 triplets", "A_W", "fresh.A_W"),
        archv("VNF factorial", "32 triplets", "I_T", "vnf.I_T"),
        archv("VNF factorial", "32 triplets", "I_W", "vnf.I_W"),
    ]
    write("table1_study_overview.csv", ["Study", "Unit", "Contrast", "Mean", "95% interval", "Evidence"], t1)

    inter = list(csv.DictReader(open(p1lib.ANALYSIS / "fixed_js_interventions.csv", encoding="utf-8")))
    t2 = []
    for cat in ("correct", "wrong", "random"):
        rows = [r for r in inter if r["category"] == cat]
        p1lib.require(len(rows) > 0, f"no fixed-JS interventions in category {cat}")
        probe = max(float(r["probe_max_abs_vs_editor_A"]) for r in rows)
        t2.append([cat, signed(fj["summary"]["mean_Y"][cat]).replace("+", ""), plain(fj["summary"]["mean_l2"][cat], 3),
                   plain(fj["summary"]["mean_tv_vs_editor_A"][cat]), sci(probe)])
    write("table2_fixed_js_by_category.csv", ["Category", "Mean Y", "Mean L2", "Mean TV from Editor A", "Max probe change"], t2)

    t3 = []
    orig = {s: {(o["slice"], o["metric"]): o for o in rows} for s, rows in res["original_run_bootstrap"].items()}
    for study, metrics in (("spec", ("delta_specific", "delta_random")), ("fixed_js", ("delta_specific", "delta_random")), ("nst", ("wrong", "random", "mean", "editor"))):
        for sl, stats in res[study]["slices"].items():
            for m in metrics:
                ci = stats[m]
                o = orig[study].get((sl, m))
                p1lib.require(o is not None, f"original-run bootstrap row missing for {study} {sl} {m}")
                t3.append([study, sl, m, stats["n"], f"{ci['point']:.8f}", f"{ci['ci95_low']:.8f}", f"{ci['ci95_high']:.8f}",
                           f"{ci['wins']}/{stats['n']}", f"{float(o['ci95_low']):.8f}", f"{float(o['ci95_high']):.8f}",
                           f"{max(abs(float(o['ci95_low']) - ci['ci95_low']), abs(float(o['ci95_high']) - ci['ci95_high'])):.2e}"])
    write("table3_recomputed_contrasts.csv",
          ["study", "slice", "metric", "n", "point", "ci95_low", "ci95_high", "wins", "original_ci95_low", "original_ci95_high", "max_abs_ci_diff"], t3)

    t4 = [[k, v["study"], v["quantity"], repr(v["value"]), "" if v.get("ci95") is None else f"{v['ci95'][0]!r}; {v['ci95'][1]!r}",
           v.get("n_boot", ""), f"data/archival/console_excerpts/{v['excerpt']}"] for k, v in sorted(arch.items())]
    write("table4_archival_values.csv", ["id", "study", "quantity", "value", "ci95", "n_boot", "source_excerpt"], t4)

    sh = res["stage_share"]
    t5 = [
        ["FULL − Editor A", f"{sh['full_minus_editor_recomputed']:.6f}", "Recomputed (NST arrays)"],
        ["FULL − Editor A (SEM console)", f"{sh['full_minus_editor_console']:.6f}", "Archival"],
        ["FULL − STAGE_ONLY via LEGAL_ONLY", f"{sh['full_minus_stage_via_legal']:.6f}", "Archival sum"],
        ["FULL − STAGE_ONLY via BOARD_ONLY", f"{sh['full_minus_stage_via_board']:.6f}", "Archival sum"],
        ["STAGE_ONLY − Editor A", f"{sh['stage_minus_editor']:.6f}", "Derived"],
        ["Stage share", f"{sh['share']:.6f}", "Derived"],
    ] + [[f"Stage share, {lab}", f"{arch[k]['value']:.6f}", "Archival"] for lab, k in
         (("category A", "sem.share_A"), ("category B", "sem.share_B"), ("exact board", "sem.share_exact"), ("nonexact board", "sem.share_nonexact"))]
    write("table5_stage_share.csv", ["quantity", "value", "evidence"], t5)

    fb = res["fiber"]
    checks = [
        ["raw files matching provenance hashes", res["raw_check"]["files_checked"], "all", 1],
        ["probe contrast rank from singular values", fb["geometry"]["rank_from_singular_values"], "124", int(fb["geometry"]["rank_from_singular_values"] == 124)],
        ["P_N trace", f"{fb['geometry']['P_N_trace']:.6f}", "388", int(abs(fb["geometry"]["P_N_trace"] - 388) < 1e-9)],
        ["FIBER max probe change H01 vs H00", f"{fb['pairs']['probe_max_abs_max']:.3e}", "<= 1e-5", int(fb["pairs"]["probe_max_abs_max"] <= 1e-5)],
        ["SPEC max probe change vs Editor A", f"{spec['probe_max_abs_vs_editor_A']:.3e}", "<= 1e-5", int(spec["probe_max_abs_vs_editor_A"] <= 1e-5)],
        ["fixed-JS max |JS - 0.01| vs Editor A", f"{fj['summary']['max_abs_js_minus_target']:.3e}", "<= 2e-4", int(fj["summary"]["max_abs_js_minus_target"] <= 2e-4)],
        ["fixed-JS max probe change vs Editor A", f"{fj['summary']['probe_max_abs_vs_editor_A']:.3e}", "<= 1e-5", int(fj["summary"]["probe_max_abs_vs_editor_A"] <= 1e-5)],
        ["fixed-JS Editor A logits identical to SPEC", f"{fj['summary']['editor_A_logits_identical_to_spec']:.1e}", "0", int(fj["summary"]["editor_A_logits_identical_to_spec"] == 0.0)],
        ["fixed-JS max |Y - archived Y|", f"{fj['summary']['max_abs_Y_vs_archive']:.3e}", "<= 1e-6", int(fj["summary"]["max_abs_Y_vs_archive"] <= 1e-6)],
        ["NST max |JS - 0.01| vs Editor A", f"{nst['summary']['max_abs_js_minus_target']:.3e}", "<= 2e-4", int(nst["summary"]["max_abs_js_minus_target"] <= 2e-4)],
        ["NST max probe change vs Editor A", f"{nst['summary']['probe_max_abs_vs_editor_A']:.3e}", "<= 1e-5", int(nst["summary"]["probe_max_abs_vs_editor_A"] <= 1e-5)],
        ["NST learned-wrong equals map difference (rel err)", f"{nst['summary']['learned_minus_wrong_vs_map_rel_err']:.3e}", "<= 1e-10", int(nst["summary"]["learned_minus_wrong_vs_map_rel_err"] <= 1e-10)],
        ["NST learned-mean equals map minus stage mean (rel err)", f"{nst['summary']['learned_minus_mean_state_vs_map_rel_err']:.3e}", "<= 1e-10", int(nst["summary"]["learned_minus_mean_state_vs_map_rel_err"] <= 1e-10)],
        ["max bootstrap CI diff vs original runs", f"{max(float(r[-1]) for r in t3):.3e}", "<= 1e-6", int(max(float(r[-1]) for r in t3) <= 1e-6)],
        ["stage-share decomposition gap", f"{sh['decomposition_gap']:.3e}", "<= 1e-6", int(sh["decomposition_gap"] <= 1e-6)],
    ]
    write("table6_integrity_checks.csv", ["check", "value", "requirement", "pass"], checks)
    failed = [c[0] for c in checks if c[3] != 1]
    p1lib.require(not failed, f"integrity checks failed: {failed}")

    AUDITS.mkdir(parents=True, exist_ok=True)
    lines = ["# Recomputation report", "",
             "Generated by `scripts/reproduce_tables.py` from `analysis/results.json`. Do not edit by hand.", "",
             "## Integrity checks", "", "| check | value | requirement | pass |", "|---|---|---|---|"]
    lines += [f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} |" for c in checks]
    lines += ["", "## Contrasts recomputed from raw arrays, with the original run's intervals", "",
              "| study | slice | metric | n | point | 95% CI (recomputed) | wins | 95% CI (original run) | max CI diff |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | [{r[5]}, {r[6]}] | {r[7]} | [{r[8]}, {r[9]}] | {r[10]} |" for r in t3]
    lines += ["", "## Archival values (not recomputable here)", "",
              "These are read from verbatim console excerpts. Their arrays were not found.", "",
              "| id | value | 95% CI | source |", "|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[3]} | {r[4]} | `{r[6]}` |" for r in t4]
    (AUDITS / "recomputation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(list(TABLES.glob('*.csv')))} tables and audits/recomputation_report.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except p1lib.AnalysisError as exc:
        print(f"TABLES FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
