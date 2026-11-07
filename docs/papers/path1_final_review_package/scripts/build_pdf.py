"""Layer E: build manuscript/paper_master.tex and paper_master.pdf from manuscript/paper_master.md.

paper_master.md is the single authoritative source. Headings in it are unnumbered; LaTeX numbers
sections, figures and tables. Figure captions and table captions are taken from the Markdown.

Usage:
    python scripts/build_pdf.py            # write .tex and compile .pdf with tectonic
    python scripts/build_pdf.py --tex-only # write .tex only
Fails loudly if tectonic is missing (unless --tex-only) or if a referenced figure is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
MS = PKG / "manuscript"
MD = MS / "paper_master.md"
TEX = MS / "paper_master.tex"
PDF = MS / "paper_master.pdf"
RECORD = MS / "build_record.json"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\usepackage{amsmath,amssymb}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage[font=small,labelfont=bf,labelsep=period]{caption}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\graphicspath{{../figures/}}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
"""


class BuildError(RuntimeError):
    pass


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
UNICODE_MATH = {
    "−": "$-$", "×": "$\\times$", "Δ": "$\\Delta$", "≥": "$\\geq$", "≤": "$\\leq$", "→": "$\\rightarrow$",
    "α": "$\\alpha$", "δ": "$\\delta$", "·": "$\\cdot$", "±": "$\\pm$",
}


def escape_text(s: str) -> str:
    s = s.replace("\\", "\u0000")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace("\u0000", r"\textbackslash{}")
    s = re.sub(r'(^|[\s(\[])"', r"\1``", s).replace('"', "''")
    s = s.replace("’", "'").replace("‘", "`").replace("“", "``").replace("”", "''")
    s = s.replace("—", "---").replace("–", "--").replace("…", r"\ldots{}")
    for a, b in UNICODE_MATH.items():
        s = s.replace(a, b)
    return re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "$^{" + m.group(0).translate(SUPERSCRIPTS) + "}$", s)


def code(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}").replace("_", r"\_\allowbreak{}").replace("&", r"\&").replace("#", r"\#").replace("%", r"\%")
    s = s.replace("—", r"{\rmfamily\textemdash}").replace("…", r"\ldots{}")
    return r"\texttt{" + s + "}"


URL = re.compile(r"(https?://[^\s)]+[^\s).,;])")


def inline(s: str) -> str:
    out = []
    for part in re.split(r"(\\\(.*?\\\)|`[^`]+`|https?://[^\s)]+[^\s).,;])", s):
        if not part:
            continue
        if part.startswith(r"\("):
            out.append(part)
        elif part.startswith("`"):
            out.append(code(part[1:-1]))
        elif URL.fullmatch(part):
            out.append(r"\url{" + part + "}")
        else:
            t = escape_text(part)
            t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
            t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", t)
            out.append(t)
    return "".join(out)


FIG_RE = re.compile(r"^!\[(.+)\]\((.+)\)\s*$")


def figure(line: str, n: int) -> str:
    m = FIG_RE.match(line)
    caption, path = m.group(1), m.group(2)
    stem = Path(path).stem
    pdf = PKG / "figures" / f"{stem}.pdf"
    if not pdf.is_file():
        raise BuildError(f"figure file missing: {pdf}")
    m = re.match(r"^Figure (\d+)\.\s*(.*)$", caption)
    if not m or int(m.group(1)) != n:
        raise BuildError(f"figure caption must start with 'Figure {n}.': {caption[:40]}")
    cap = m.group(2)
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        r"\includegraphics[width=\textwidth]{" + stem + ".pdf}",
        r"\caption{" + inline(cap) + "}", r"\label{fig:" + str(n) + "}", r"\end{figure}",
    ])


def table(rows: list[str], caption: str, n: int) -> str:
    m = re.match(r"^Table (\d+)\.\s*(.*)$", caption)
    if not m or int(m.group(1)) != n:
        raise BuildError(f"table caption must start with 'Table {n}.': {caption[:40]}")
    caption = m.group(2)
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [c for c in cells[2:]]
    ncol = len(header)
    if any(len(r) != ncol for r in body):
        raise BuildError(f"ragged table: {rows[0]}")
    out = [r"\begin{table}[tbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\caption{" + inline(caption) + "}", r"\label{tab:" + str(n) + "}"]
    out += [r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule",
            " & ".join(r"\textbf{" + inline(h) + "}" for h in header) + r" \\", r"\midrule"]
    out += [" & ".join(inline(c) for c in r) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def convert(md: str) -> str:
    lines = md.splitlines()
    title, meta, body = None, [], []
    i, nfig, ntab, in_appendix, in_refs = 0, 0, 0, False, False
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            i += 1
            continue
        if title is not None and not body and s.startswith("**") and ":**" in s:
            meta.append(inline(s))
            i += 1
            continue
        if line.startswith("## "):
            h = line[3:].strip()
            if re.match(r"^\d", h):
                raise BuildError(f"numbered heading in Markdown (LaTeX numbers sections): {h}")
            if any(b == r"\begin{abstract}" for b in body) and not any(b == r"\end{abstract}" for b in body):
                body.append(r"\end{abstract}")
            if h == "Abstract":
                body.append(r"\begin{abstract}")
            elif h == "References":
                in_refs = True
                body.append(r"\section*{References}")
                body.append(r"\begingroup\small")
            else:
                m = re.match(r"^Appendix ([A-Z])\.\s+(.*)$", h)
                if in_refs:
                    body.append(r"\endgroup")
                    in_refs = False
                if m:
                    if not in_appendix:
                        body.append(r"\clearpage" + "\n" + r"\appendix")
                        in_appendix = True
                    body.append(r"\section{" + inline(m.group(2)) + "}")
                else:
                    if in_appendix:
                        raise BuildError(f"non-appendix section after appendices: {h}")
                    body.append(r"\section{" + inline(h) + "}")
            i += 1
            continue
        if line.startswith("### "):
            h = line[4:].strip()
            if re.match(r"^\d", h):
                raise BuildError(f"numbered subheading in Markdown: {h}")
            body.append(r"\subsection{" + inline(h) + "}")
            i += 1
            continue
        if FIG_RE.match(s):
            nfig += 1
            body.append(figure(s, nfig))
            i += 1
            continue
        if s.startswith("|"):
            raise BuildError(f"table without a 'Table: Table N.' caption line near line {i + 1}")
        if s.startswith("Table:"):
            caption = s[len("Table:"):].strip()
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            if len(rows) < 3:
                raise BuildError(f"table caption without a table near line {i}")
            ntab += 1
            body.append(table(rows, caption, ntab))
            continue
        if s.startswith(r"\["):
            chunk = [line]
            i += 1
            while i < len(lines) and r"\]" not in chunk[-1]:
                chunk.append(lines[i])
                i += 1
            body.append("\n".join(chunk))
            continue
        if not s or s == "---":
            i += 1
            continue
        para = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") and not lines[i].strip().startswith(("|", "![", r"\[", "Table:")):
            para.append(lines[i].strip())
            i += 1
        text = inline(" ".join(para))
        body.append(r"\noindent\hangindent=1.5em\hangafter=1 " + text if in_refs else text)
    if in_refs:
        body.append(r"\endgroup")
    if title is None:
        raise BuildError("manuscript has no '# ' title line")
    head = PREAMBLE + r"\title{" + inline(title) + "}\n" + r"\author{\parbox{0.9\textwidth}{\centering\normalsize " + r" \\[0.3em] ".join(meta) + "}}\n" + r"\date{}" + "\n"
    return head + r"\begin{document}" + "\n" + r"\maketitle" + "\n\n" + "\n\n".join(body) + "\n\n" + r"\end{document}" + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-only", action="store_true")
    a = ap.parse_args()
    if not MD.is_file():
        raise BuildError(f"missing {MD}")
    md_sha = sha256(MD)
    tex = f"% Generated by scripts/build_pdf.py from paper_master.md (sha256 {md_sha}). Do not edit.\n" + convert(MD.read_text(encoding="utf-8"))
    TEX.write_text(tex, encoding="utf-8")
    record = {"md_sha256": md_sha, "tex_sha256": sha256(TEX), "built_utc": datetime.now(timezone.utc).isoformat(),
              "figures_sha256": {p.name: sha256(p) for p in sorted((PKG / "figures").glob("*.pdf"))},
              "figure_environment": json.loads((PKG / "figures" / "figure_provenance.json").read_text(encoding="utf-8"))["environment"]}
    if a.tex_only:
        record["pdf_sha256"] = None
        RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("wrote", TEX)
        return 0
    exe = shutil.which("tectonic")
    if exe is None:
        raise BuildError("tectonic not found on PATH; install it or rerun with --tex-only")
    ver = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    proc = subprocess.run([exe, "-X", "compile", TEX.name], cwd=MS, capture_output=True, text=True)
    (PKG / "logs").mkdir(exist_ok=True)
    (PKG / "logs" / "tectonic_build.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not PDF.is_file():
        raise BuildError(f"tectonic failed (exit {proc.returncode}); see logs/tectonic_build.txt")
    warn = [l for l in (proc.stdout + proc.stderr).splitlines() if "Overfull" in l or "undefined" in l.lower()]
    if any("Overfull" in l for l in warn):
        raise BuildError("overfull boxes in PDF build: " + "; ".join(sorted(set(warn))))
    record.update({"pdf_sha256": sha256(PDF), "tectonic": ver, "latex_warnings": warn})
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TEX.name} and {PDF.name}; {len(warn)} overfull/undefined warnings")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
