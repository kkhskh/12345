"""Layer D: check every claim in manuscript/paper_master.md against its evidence.

Run from anywhere:
    python scripts/verify_paper_claims.py

What it does (all checks must pass; exit code 1 otherwise):
  1. Recomputes every analysis quantity in memory from data/raw (p1lib.compute_all) and checks
     that analysis/results.json and the analysis CSVs are identical to the fresh recomputation.
  2. Regenerates all tables into a temporary directory and checks tables/ is byte-identical.
  3. Checks the manuscript's pipe tables against tables/table1 and tables/table2.
  4. Evaluates audits/claim_evidence_ledger.csv: each row's context must occur in the manuscript
     and its value must match the recomputed, recorded, archival, or protocol source; every
     numeric token in the manuscript (outside tables, code spans and the reference list) must be
     covered by a ledger row.
  5. Checks official result letters verbatim, forbidden phrases, figure and table numbering,
     protocol constants, archival excerpt hashes, figure provenance, and the TeX/PDF build record.
  6. Checks the package contains no sealed-confirmation file and no model checkpoint.
It never writes inside the package.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1lib  # noqa: E402

PKG = p1lib.PKG
MD = PKG / "manuscript" / "paper_master.md"
LEDGER = PKG / "audits" / "claim_evidence_ledger.csv"

LETTERS = [
    "08R2_VALIDATION_COMPLETE — NEGATIVE", "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED", "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED",
    "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE", "DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED", "LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY",
    "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED", "SYMMETRIC_TARGET_CENSUS_INFEASIBLE",
    "TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED", "0A = C / PARTIAL", "SECTION_5_3 = PARTIAL", "OBSTRUCTION_READY = NO",
]
FORBIDDEN = {
    "unedited recipient": "JS and probe references are the Editor A reference activation, not the unedited recipient",
    "import src": "package code must not import the research repository",
    "from src": "package code must not import the research repository",
    "plotting_scripts/": "obsolete directory from the superseded draft",
    "verified_data/": "obsolete directory from the superseded draft",
    "fig2_probe_preservation": "obsolete Figure 2 variant",
}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".pkl"}

failures: list[str] = []
passes: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    (passes if ok else failures).append(name + ("" if ok or not detail else f": {detail}"))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ 1. recomputation
def deep_compare(a, b, path="", tol=1e-12) -> list[str]:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k} present in only one side")
            else:
                out += deep_compare(a[k], b[k], f"{path}/{k}", tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += deep_compare(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        fa, fb = float(a), float(b)
        if not ((math.isnan(fa) and math.isnan(fb)) or abs(fa - fb) <= tol * max(1.0, abs(fa), abs(fb))):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def recompute() -> dict:
    full = p1lib.compute_all()
    fresh = p1lib.strip_rows(full)
    saved = json.loads((p1lib.ANALYSIS / "results.json").read_text(encoding="utf-8"))
    diffs = deep_compare(fresh, saved)
    check(not diffs, "analysis/results.json equals in-memory recomputation from data/raw", "; ".join(diffs[:5]))
    rac = importlib.import_module("recompute_analysis")
    with tempfile.TemporaryDirectory() as td:
        rac.write_csvs(full, Path(td))
        for p in sorted(Path(td).glob("*.csv")):
            q = p1lib.ANALYSIS / p.name
            check(q.is_file() and q.read_bytes() == p.read_bytes(), f"analysis/{p.name} equals recomputation")
    return fresh


# ------------------------------------------------------------------ 2. tables
def tables_regenerate() -> None:
    rt = importlib.import_module("reproduce_tables")
    with tempfile.TemporaryDirectory() as td:
        old_t, old_a = rt.TABLES, rt.AUDITS
        rt.TABLES, rt.AUDITS = Path(td) / "tables", Path(td) / "audits"
        try:
            rt.main()
        finally:
            rt.TABLES, rt.AUDITS = old_t, old_a
        gen = sorted((Path(td) / "tables").glob("*.csv"))
        check(len(gen) > 0, "tables regenerate")
        for p in gen:
            q = PKG / "tables" / p.name
            check(q.is_file() and q.read_bytes() == p.read_bytes(), f"tables/{p.name} is reproducible")
        extra = {p.name for p in (PKG / "tables").glob("*")} - {p.name for p in gen}
        check(not extra, "tables/ has no unreproducible files", str(sorted(extra)))
        rep = Path(td) / "audits" / "recomputation_report.md"
        check(rep.read_bytes() == (PKG / "audits" / "recomputation_report.md").read_bytes(), "audits/recomputation_report.md is reproducible")


def md_tables(md: str) -> list[list[list[str]]]:
    out, cur = [], []
    for line in md.splitlines():
        if line.strip().startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [[r for r in t if not set("".join(r)) <= set("-: ")] for t in out]


def norm_cell(s: str) -> str:
    return re.sub(r"\\\((.*?)\\\)", r"\1", s).replace("-", "\u2212").strip()


def manuscript_tables(md: str) -> None:
    ts = md_tables(md)
    check(len(ts) == 2, "manuscript has exactly two pipe tables", str(len(ts)))
    for t, name in zip(ts, ("table1_study_overview.csv", "table2_fixed_js_by_category.csv")):
        rows = list(csv.reader(open(PKG / "tables" / name, encoding="utf-8")))
        a = [[norm_cell(c) for c in r] for r in t]
        b = [[norm_cell(c) for c in r] for r in rows]
        check(a == b, f"manuscript table matches tables/{name}", f"{a[:2]} vs {b[:2]}" if a != b else "")


# ------------------------------------------------------------------ 4. ledger
TOKEN_RE = re.compile(
    r"(?<![\w.^_\\])(?:10\^\{[−-]?\d+\}|[−-]?\d+(?:\.\d+)?(?:\\times\s?10\^\{[−-]?\d+\})?%?)(?![\w])")


def parse_token(tok: str) -> tuple[float, float]:
    """Return (value, half unit of the last displayed digit)."""
    t = tok.replace("−", "-").replace("\\times", "e").replace(" ", "")
    pct = t.endswith("%")
    t = t.rstrip("%")
    m = re.fullmatch(r"(-?)10\^\{(-?\d+)\}", t)
    if m:
        e = int(m.group(2))
        return float(f"{m.group(1)}1e{e}"), 0.5 * 10.0 ** e
    m = re.fullmatch(r"(-?\d+(?:\.(\d+))?)(?:e10\^\{(-?\d+)\})?", t)
    if not m:
        raise ValueError(f"unparseable token {tok!r}")
    mant, dec, exp = m.group(1), m.group(2) or "", int(m.group(3) or 0)
    return float(mant) * 10.0 ** exp, 0.5 * 10.0 ** (exp - len(dec))


def ledger_env(res: dict) -> dict:
    arch = res["archival"]
    sl = {s: res[s]["slices"] for s in ("spec", "fixed_js", "nst")}
    orig = {s: {(o["slice"], o["metric"]): o for o in rows} for s, rows in res["original_run_bootstrap"].items()}
    names = {"lo": "ci95_low", "hi": "ci95_high", "point": "point", "wins": "wins", "n": "n"}
    src = (HERE / "p1lib.py").read_text(encoding="utf-8")

    def R(path):
        return p1lib.get_path(res, path)

    def CI(study, s, metric, field):
        return sl[study][s][metric][names[field]]

    def A(i):
        p1lib.require(i in arch, f"unknown archival id {i}")
        return arch[i]["value"]

    def AL(i):
        return arch[i]["ci95"][0]

    def AH(i):
        return arch[i]["ci95"][1]

    def AN(i, f):
        return arch[i][f]

    def CONST(name):
        return getattr(p1lib, name)

    def PROTO(fname, text):
        p = PKG / "data" / "protocols" / fname
        p1lib.require(p.is_file(), f"protocol file missing: {fname}")
        return text in p.read_text(encoding="utf-8")

    def SRC(text):
        return text in src

    def MAXBOOTDIFF():
        m = 0.0
        for study, slices in sl.items():
            for s, stats in slices.items():
                for metric, ci in stats.items():
                    if not isinstance(ci, dict):
                        continue
                    o = orig[study].get((s, metric))
                    p1lib.require(o is not None, f"original bootstrap row missing: {study} {s} {metric}")
                    m = max(m, abs(float(o["ci95_low"]) - ci["ci95_low"]), abs(float(o["ci95_high"]) - ci["ci95_high"]))
        return m

    def PROBESHAPE():
        z = p1lib.load_npz(p1lib.RAW / "fixed_js" / "fixed_js_probe_probabilities.npz")
        shapes = {tuple(v.shape[-2:]) for v in z.values()}
        p1lib.require(len(shapes) == 1, f"inconsistent probe shapes {shapes}")
        return shapes.pop()

    def LOGITDIM():
        z = p1lib.load_npz(p1lib.RAW / "fixed_js" / "fixed_js_full_logits.npz")
        dims = {v.shape[-1] for v in z.values()}
        p1lib.require(len(dims) == 1, f"inconsistent logit widths {dims}")
        return dims.pop()

    def FEATDIM():
        return len(p1lib.read_json(p1lib.RAW / "nst" / "null_state_feature_names.json")["names"])

    def JSSMOOTHDIFF():
        L = p1lib.load_npz(p1lib.RAW / "fixed_js" / "fixed_js_full_logits.npz")
        worst = 0.0
        for k, v in L.items():
            ref = p1lib.softmax(L[k.rsplit("__", 1)[0] + "__editor_A"])
            p, q = p1lib._smooth(ref, p1lib.JS_EPS), p1lib._smooth(p1lib.softmax(v), p1lib.JS_EPS)
            m = 0.5 * (p + q)
            single = 0.5 * float(np.sum(p * np.log(p / m))) + 0.5 * float(np.sum(q * np.log(q / m)))
            worst = max(worst, abs(single - p1lib.js_divergence(ref, p1lib.softmax(v))))
        return worst

    def JSSMOOTHDIFF():
        L = p1lib.load_npz(p1lib.RAW / "fixed_js" / "fixed_js_full_logits.npz")
        worst = 0.0
        for k, v in L.items():
            ref = p1lib.softmax(L[k.rsplit("__", 1)[0] + "__editor_A"])
            p, q = p1lib._smooth(ref, p1lib.JS_EPS), p1lib._smooth(p1lib.softmax(v), p1lib.JS_EPS)
            m = 0.5 * (p + q)
            single = 0.5 * float(np.sum(p * np.log(p / m))) + 0.5 * float(np.sum(q * np.log(q / m)))
            worst = max(worst, abs(single - p1lib.js_divergence(ref, p1lib.softmax(v))))
        return worst

    def TOKENMAP_OK():
        toks = [p1lib.square_to_token(s) for s in range(64) if s not in p1lib.CENTER_SQUARES]
        return sorted(toks) == list(range(1, 61))

    env = {k: v for k, v in locals().items() if k[0].isupper()}
    env.update({"abs": abs, "min": min, "max": max, "all": all, "round": round, "tuple": tuple})
    return env


def strip_for_scan(md: str) -> str:
    body = md.split("\n## References\n")[0] + "\n" + md.split("\n## Appendix A.")[1]
    body = body.replace("08R2", "____")
    lines = []
    for line in body.splitlines():
        if line.strip().startswith("|") or line.startswith("# ") or line.startswith("**Authors") or line.startswith("**Status"):
            lines.append(" " * len(line))
            continue
        lines.append(line)
    body = "\n".join(lines)
    body = re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), body)
    body = re.sub(r"\]\(\.\./figures/[^)]*\)", lambda m: " " * len(m.group(0)), body)
    body = re.sub(r"(Figure|Table|Appendix) \d+", lambda m: " " * len(m.group(0)), body)
    return body


def ledger(md: str, res: dict) -> None:
    rows = list(csv.DictReader(open(LEDGER, encoding="utf-8")))
    check(len(rows) > 0, "claim ledger is not empty")
    env = ledger_env(res)
    scan = strip_for_scan(md)
    occ = [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(scan)]
    covered = set()
    for row in rows:
        cid, ctx, tok, mode, expr = row["claim_id"], row["context"], row["token"], row["mode"], row["expr"]
        starts = [m.start() for m in re.finditer(re.escape(ctx), scan)]
        if not starts:
            check(False, f"ledger {cid}: context occurs in manuscript", repr(ctx))
            continue
        if tok and tok != "*":
            check(tok in ctx, f"ledger {cid}: token inside context", f"{tok!r} not in {ctx!r}")
        for s in starts:
            for a, b, t in occ:
                if a >= s and b <= s + len(ctx) and (tok == "*" or t.replace("−", "-") == tok):
                    covered.add(a)
        if mode == "context":
            check(row["evidence"] == "CONTEXT", f"ledger {cid}: context-only rows are labeled CONTEXT")
            continue
        try:
            val = eval(expr, {"__builtins__": {}, **env})  # noqa: S307 - expressions come from the package ledger
        except Exception as exc:  # report, never hide
            check(False, f"ledger {cid}: expression evaluates", f"{type(exc).__name__}: {exc}")
            continue
        if mode == "true":
            check(val is True, f"ledger {cid}: {row['note'] or ctx[:50]}", f"expr returned {val!r}")
            continue
        shown, half = parse_token(tok)
        v = float(val)
        if mode == "eq":
            ok = v == shown
        elif mode == "round":
            ok = abs(v - shown) <= half * (1 + 1e-9) + 1e-15
        elif mode == "atmost":
            ok = v <= shown
        elif mode == "atleast":
            ok = v >= shown
        else:
            ok = False
        check(ok, f"ledger {cid}: {tok} ({row['evidence']})", f"source value {v!r}, mode {mode}")
    years = {t for _, _, t in occ if re.fullmatch(r"20[12]\d", t)}
    refs = md.split("\n## References\n")[1].split("\n## Appendix A.")[0]
    for y in sorted(years):
        check(f"({y})" in refs, f"citation year {y} appears in the reference list")
    missing = [(a, t) for a, _, t in occ if a not in covered and not re.fullmatch(r"20[12]\d", t)]
    detail = "; ".join(f"{t!r} at …{scan[max(0, a - 30):a + 15].strip()!r}" for a, t in missing[:12])
    check(not missing, f"every numeric token in the manuscript is covered by the ledger ({len(occ)} tokens)", f"{len(missing)} uncovered: {detail}")


# ------------------------------------------------------------------ 5. text, numbering, build
def text_checks(md: str, res: dict) -> None:
    for L in LETTERS:
        check(f"`{L}`" in md, f"letter verbatim in manuscript: {L}")
    for study, letter in res["raw_letters"].items():
        check(f"`{letter}`" in md, f"raw completion letter for {study} appears in manuscript")
    fiber_letter = p1lib.read_json(p1lib.RAW / "fiber" / "08r2_fiber_completion_status.json")["verdict"]
    allowed = set(LETTERS) | {fiber_letter}
    for cfg in ("toy_othello_08r2_fiber_spec.json", "toy_othello_08r2_null_state_transfer.json"):
        c = json.loads((PKG / "data" / "protocols" / cfg).read_text(encoding="utf-8"))
        allowed |= {c["site"], f"{c['expected_pair_manifest_sha256'][:8]}…{c['expected_pair_manifest_sha256'][-6:]}"}
    spans = set(re.findall(r"`([^`]+)`", md))
    check(spans <= allowed, "every code span is an official letter, the configured site, or a configured pair-manifest hash",
          str(sorted(spans - allowed)))
    pkg_text = {p: p.read_text(encoding="utf-8", errors="replace") for p in PKG.rglob("*")
                if p.is_file() and p.suffix in {".md", ".py", ".tex", ".json", ".txt", ".csv"}
                and "data" not in p.relative_to(PKG).parts[:1] and p.name not in {"verify_paper_claims.py", "test_paper_reproducibility.py", "integration_check_against_repo.py"}
                and "reviews" not in p.relative_to(PKG).parts[:1] and "logs" not in p.relative_to(PKG).parts[:1]
                and "audits" not in p.relative_to(PKG).parts[:1]}
    for phrase, why in FORBIDDEN.items():
        hits = [str(p.relative_to(PKG)) for p, t in pkg_text.items() if phrase in t]
        check(not hits, f"forbidden phrase absent: {phrase!r}", f"{why}; found in {hits}")
    check(not re.search(r"\bSection \d", md), "manuscript uses no hard-coded section numbers")
    heads = [l for l in md.splitlines() if l.startswith("#")]
    check(not [h for h in heads if re.match(r"^#+\s+\d", h)], "manuscript headings are unnumbered (LaTeX numbers them)")
    check(len(heads) == len(set(heads)), "manuscript headings are unique")
    figs = re.findall(r"^!\[Figure (\d+)\.[^\]]*\]\(\.\./figures/([^)]+)\)", md, flags=re.M)
    check([int(n) for n, _ in figs] == list(range(1, len(figs) + 1)), "figures numbered 1..N in order")
    for _, f in figs:
        check((PKG / "figures" / f).is_file() and (PKG / "figures" / f).with_suffix(".pdf").is_file(), f"figure files exist: {f}")
    on_disk = {p.stem for p in (PKG / "figures").glob("*.png")}
    check(on_disk == {Path(f).stem for _, f in figs}, "figures/ contains exactly the manuscript figures", str(sorted(on_disk)))
    caps = re.findall(r"^Table: Table (\d+)\.", md, flags=re.M)
    check([int(n) for n in caps] == list(range(1, len(caps) + 1)), "tables numbered 1..N in order")
    for kind, n in ((("Figure", len(figs))), ("Table", len(caps))):
        refs = {int(x) for x in re.findall(rf"\b{kind} (\d+)\b", md)}
        check(refs <= set(range(1, n + 1)), f"every {kind} reference resolves", str(sorted(refs)))
    apps = set(re.findall(r"^## Appendix ([A-Z])\.", md, flags=re.M))
    check(set(re.findall(r"\bAppendix ([A-Z])\b", md)) <= apps, "every Appendix reference resolves")
    check(md.count("Editor A reference activation") >= 10, "manuscript names the Editor A reference activation explicitly")
    for phrase in (
        "The absolute score stayed negative.",
        "below a preregistered 60% gate",
        "That interval includes zero.",
        "not by itself evidence of a stage-specific correction",
        "It does not replace the inconclusive letter",
        "Kernel steps of similar length barely move the output at a natural donor activation.",
    ):
        check(phrase in md, f"directional claim is present, not reversed: {phrase[:48]}")
    refs = [p.strip() for p in md.split("\n## References\n")[1].split("\n## Appendix A.")[0].split("\n\n") if p.strip()]
    bib = (PKG / "manuscript" / "references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{(\w+),.*?author = \{([^,}]+)[,}].*?year = \{(\d{4})\}", bib, flags=re.S)
    unmatched = [k for k, sur, yr in entries if sum(r.startswith(f"{sur}, ") and f"({yr})" in r for r in refs) != 1]
    check(len(entries) == len(refs) and not unmatched, "references.bib has exactly one entry per manuscript reference",
          f"{len(entries)} bib entries, {len(refs)} references, unmatched {unmatched}")


def protocol_and_archival() -> None:
    for k, (v, f, sub) in p1lib.PROTOCOL_CONSTANTS.items():
        p = PKG / "data" / "protocols" / f
        check(p.is_file() and sub in p.read_text(encoding="utf-8"), f"protocol constant {k}={v} appears in {f}")
    man = json.loads((PKG / "data" / "provenance_manifest.json").read_text(encoding="utf-8"))
    for e in man["console_excerpts"] + man["protocols"]:
        p = PKG / e["package_path"]
        check(p.is_file() and sha256(p) == e["sha256"], f"hash matches provenance manifest: {e['package_path']}")
    listed = {e["package_path"] for e in man["console_excerpts"] + man["protocols"]}
    on_disk = {str(p.relative_to(PKG)) for d in ("data/protocols", "data/archival/console_excerpts") for p in (PKG / d).iterdir() if p.is_file()}
    check(on_disk == listed, "every protocol and console excerpt in the package is in the provenance manifest", str(sorted(on_disk ^ listed)))
    arch = json.loads((PKG / "data" / "archival" / "archival_values.json").read_text(encoding="utf-8"))["values"]
    for v in arch:
        ex = (PKG / "data" / "archival" / "console_excerpts" / v["excerpt"]).read_text(encoding="utf-8")
        check(v["excerpt_contains"] in ex, f"archival {v['id']}: quoted text occurs verbatim in {v['excerpt']}")
        lit = repr(v["value"]) if isinstance(v["value"], float) else str(v["value"])
        check(lit in ex or f"{v['value']:.4f}" in ex or (isinstance(v["value"], float) and f"{v['value']:.6f}" in ex),
              f"archival {v['id']}: value {lit} occurs in its excerpt")


def build_checks() -> None:
    rec_p = PKG / "manuscript" / "build_record.json"
    tex, pdf = PKG / "manuscript" / "paper_master.tex", PKG / "manuscript" / "paper_master.pdf"
    check(rec_p.is_file() and tex.is_file() and pdf.is_file(), "manuscript .tex, .pdf and build record exist")
    if not rec_p.is_file():
        return
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    md_sha = sha256(MD)
    check(rec["md_sha256"] == md_sha, "PDF was built from the current paper_master.md")
    check(md_sha in tex.read_text(encoding="utf-8").splitlines()[0], "paper_master.tex header carries the current md hash")
    check(rec["tex_sha256"] == sha256(tex), "build record matches paper_master.tex")
    check(rec.get("pdf_sha256") == sha256(pdf), "build record matches paper_master.pdf")
    check(not rec.get("latex_warnings"), "LaTeX build had no overfull boxes or undefined references", str(rec.get("latex_warnings")))
    fp = json.loads((PKG / "figures" / "figure_provenance.json").read_text(encoding="utf-8"))
    for fig, entry in fp["figures"].items():
        for name, h in entry["inputs"].items():
            check(sha256(PKG / name) == h, f"{fig}: input unchanged since the figure was drawn: {name}")
        for name, h in entry["outputs_sha256"].items():
            check(sha256(PKG / name) == h, f"{fig}: file matches figure provenance: {name}")
    cur = {p.name: sha256(p) for p in sorted((PKG / "figures").glob("*.pdf"))}
    built_with = rec["figure_environment"]["matplotlib"]
    if rec["figures_sha256"] == cur:
        check(True, "PDF was built with the current figure files")
    elif built_with == fp["environment"]["matplotlib"]:
        check(False, "PDF was built with the current figure files", "figures changed after the PDF build; rerun scripts/build_pdf.py")
    else:
        print(f"NOTE figure files were regenerated with matplotlib {fp['environment']['matplotlib']}; the PDF embeds figures drawn "
              f"with matplotlib {built_with}, so bytes differ. Figure inputs are still checked against the recomputation.")


def package_hygiene() -> None:
    bad = [str(p.relative_to(PKG)) for p in PKG.rglob("*") if p.is_file() and
           (("confirmation" in p.name.lower() and "protocol" in p.name.lower()) or p.suffix.lower() in CHECKPOINT_SUFFIXES)]
    check(not bad, "no sealed-confirmation protocol or checkpoint files in package", str(bad))
    pyfiles = [p for p in PKG.rglob("*.py")]
    imp = [str(p.relative_to(PKG)) for p in pyfiles if p.name != "integration_check_against_repo.py"
           and re.search(r"^\s*(from|import)\s+src\b", p.read_text(encoding="utf-8"), flags=re.M)]
    check(not imp, "no package module imports the research repository", str(imp))


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manuscript", type=Path, default=MD, help="check a different copy of paper_master.md (used by tests)")
    md = ap.parse_args(argv).manuscript.read_text(encoding="utf-8")
    try:
        res = recompute()
        tables_regenerate()
        manuscript_tables(md)
        ledger(md, res)
        text_checks(md, res)
        protocol_and_archival()
        build_checks()
        package_hygiene()
    except p1lib.AnalysisError as exc:
        check(False, "analysis layer", str(exc))
    for p in passes:
        print("PASS", p)
    for f in failures:
        print("FAIL", f)
    print(f"\n{len(passes)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
