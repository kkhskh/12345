#!/usr/bin/env python3
"""Convert paper_master.md to a full LaTeX file with figures. Not a general Markdown engine."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = (ROOT / "paper_master.md").read_text()
TEX = ROOT / "paper_master.tex"

HEADER = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{fontspec}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\hypersetup{colorlinks=false}
\title{Causal Information Outside a Frozen Board Probe in Othello-GPT}
\author{Authors withheld pending human confirmation}
\date{Private draft. Not submitted.}
\begin{document}
\maketitle
"""

FOOTER = r"""
\end{document}
"""

FIGURES = {
    "Figure 1 shows the geometry": r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig1_probe_geometry_schematic.pdf}
\caption{Schematic of the frozen-probe split. Both rowspace and nullspace enter the later layers of the frozen model. Only the rowspace is read by this probe. Probe-invisible means invariant to this probe, not unused by the network.}
\label{fig:geom}
\end{figure}
""",
    "Figure 2 shows that probe probabilities stay at roundoff": r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig2_probe_preservation.pdf}
\caption{Fixed-JS probe preservation versus output change. Each point is one of 336 interventions on 48 pairs. The inference unit is the pair. Source: \texttt{verified\_data/fixed\_js\_verified\_data.csv}.}
\label{fig:fig2}
\end{figure}
""",
    "Figure 3 shows the paired": r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{figures/fig3_fixed_js_contrasts.pdf}
\caption{Paired $\Delta Y$ at JS $=0.01$ for correct versus wrong and random controls. $n=48$ pairs.}
\label{fig:fig3}
\end{figure}
""",
    "not recovery of the oracle representation (Figure 4)": r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{figures/fig4_nst_contrasts.pdf}
\caption{Donor-free NST paired $\Delta Y$ contrasts. Mean $Y_{\mathrm{learned}}$ remains negative. $n=48$ pairs.}
\label{fig:fig4}
\end{figure}
""",
    "Figure 5 summarizes this gate": r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig5_limits_fresh_vnf.pdf}
\caption{Limits of the learned map: FRESH joint switching below the 60\% gate, and VNF interaction estimates below $+0.005$.}
\label{fig:fig5}
\end{figure}
""",
}


def escape_text(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", "``").replace("”", "''")
    s = s.replace("–", "--").replace("—", "---")
    s = s.replace("…", "...")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace(r"\_", "<<US>>")
    s = s.replace("_", r"\_")
    s = s.replace("<<US>>", r"\_")
    return s


def inline(s: str) -> str:
    parts = re.split(r"(\\\(.*?\\\))", s)
    out = []
    for part in parts:
        if part.startswith(r"\("):
            out.append(part)
            continue
        part = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1).replace("_", r"\_").replace("&", r"\&") + "}", part)
        part = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", part)
        part = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", part)
        out.append(escape_text(part))
    return "".join(out)


def convert() -> str:
    lines = MD.splitlines()
    body = []
    i = 0
    skipping_status = True
    while i < len(lines):
        line = lines[i]
        if skipping_status:
            if line.startswith("# "):
                skipping_status = False
                i += 1
                continue
            i += 1
            continue
        if line.strip() == "---":
            i += 1
            continue
        if line.startswith("## Abstract"):
            body.append(r"\begin{abstract}")
            i += 1
            abs_lines = []
            while i < len(lines) and not lines[i].startswith("## "):
                if lines[i].strip():
                    abs_lines.append(inline(lines[i].strip()))
                i += 1
            body.append(" ".join(abs_lines))
            body.append(r"\end{abstract}")
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            num = title.split(".", 1)
            body.append(r"\section{" + inline(title) + "}")
            i += 1
            continue
        if line.startswith("### "):
            body.append(r"\subsection{" + inline(line[4:].strip()) + "}")
            i += 1
            continue
        if line.startswith("**") and line.endswith("**") and "." in line[:40]:
            body.append(r"\paragraph{" + inline(line.strip("*")) + "}")
            i += 1
            continue
        if line.startswith(r"\[") or (line.strip() == r"\[" ):
            chunk = [line]
            i += 1
            while i < len(lines) and r"\]" not in lines[i]:
                chunk.append(lines[i])
                i += 1
            if i < len(lines):
                chunk.append(lines[i])
                i += 1
            body.append("\n".join(chunk))
            continue
        if line.startswith("- "):
            body.append(r"\begin{itemize}")
            while i < len(lines) and lines[i].startswith("- "):
                body.append(r"\item " + inline(lines[i][2:]))
                i += 1
            body.append(r"\end{itemize}")
            continue
        if not line.strip():
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") and not lines[i].startswith("- ") and not lines[i].startswith(r"\[") and lines[i].strip() != "---":
            para.append(lines[i])
            i += 1
        text = inline(" ".join(p.strip() for p in para))
        body.append(text)
        for key, fig in FIGURES.items():
            if key in " ".join(para):
                body.append(fig)
    return HEADER + "\n\n".join(body) + "\n" + FOOTER


if __name__ == "__main__":
    TEX.write_text(convert())
    print("wrote", TEX)
