"""Write audits/sentence_audit.csv: one row per manuscript sentence with its numbers, ledger rows and wording flags.

Usage (from anywhere):
    python scripts/sentence_audit.py

Each sentence of the main text, captions and appendices is listed with the numeric tokens it
contains and the claim-ledger rows that cover them, as found by the same matching rule as
scripts/verify_paper_claims.py. Wording flags mark sentences for manual review:
  causal    causal verbs and nouns (cause, effect, change, use, drive, influence)
  strength  strength adjectives (large, strong, clear, robust, substantial, significant)
  priority  priority wording (first, novel, new, unprecedented)
  hedge     hedges (may, might, suggest, appear, likely)
  scope     explicit scope (this probe, this model, this site, Othello-GPT)
  negative  negative or limiting statements (not, no, fail, below, negative)
audits/scientific_sentence_audit.md records the manual review of the flagged sentences.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_paper_claims as vpc  # noqa: E402

PKG = vpc.PKG
OUT = PKG / "audits" / "sentence_audit.csv"
FLAGS = {
    "causal": r"\b(caus\w*|effects?|chang\w*|uses?|used|drives?|influence\w*|moves?|moved|shift\w*)\b",
    "strength": r"\b(large|strong\w*|clear\w*|robust\w*|substantial\w*|significant\w*|dramatic\w*)\b",
    "priority": r"\b(first|novel\w*|new|unprecedented)\b",
    "hedge": r"\b(may|might|suggests?|appears?|likely|plausibl\w*)\b",
    "scope": r"\b(this probe|this model|this site|one probe|one model|Othello-GPT)\b",
    "negative": r"\b(not|no|fail\w*|below|negative|includes? zero|unsupported)\b",
}


def sections(body: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(1).strip()) for m in re.finditer(r"^#+\s+(.*)$", body, flags=re.M)]


def main() -> int:
    md = vpc.MD.read_text(encoding="utf-8")
    orig = md.split("\n## References\n")[0] + "\n" + md.split("\n## Appendix A.")[1]
    scan = vpc.strip_for_scan(md)
    orig = orig.rstrip("\n")
    if len(orig) != len(scan):
        raise SystemExit(f"scan text and manuscript text differ in length ({len(scan)} vs {len(orig)}); offsets would not align")
    rows = list(csv.DictReader(open(vpc.LEDGER, encoding="utf-8")))
    owners: dict[int, list[str]] = {}
    for row in rows:
        ctx, tok = row["context"], row["token"]
        for s in [m.start() for m in re.finditer(re.escape(ctx), scan)]:
            for m in vpc.TOKEN_RE.finditer(scan):
                a, b, t = m.start(), m.end(), m.group(0)
                if a >= s and b <= s + len(ctx) and (tok == "*" or t.replace("−", "-") == tok):
                    owners.setdefault(a, []).append(row["claim_id"])
    refs = md.split("\n## References\n")[1].split("\n## Appendix A.")[0]
    for m in vpc.TOKEN_RE.finditer(scan):
        if re.fullmatch(r"(19|20)\d\d", m.group(0)) and f"({m.group(0)})" in refs:
            owners.setdefault(m.start(), []).append("REFERENCE_YEAR")
    heads = sections(orig)
    out = []
    for para in re.finditer(r"[^\n]+", orig):
        text = para.group(0)
        if text.startswith("#") or text.strip().startswith("|") or text.startswith("**Authors") or text.startswith("**Status"):
            continue
        if text.startswith("\\[") or text.startswith("\\]") or text.startswith("G=") or not text.strip():
            continue
        for s in re.finditer(r".+?(?:(?<!\bal)(?<!\be\.g)(?<!\bi\.e)[.!?](?=\s+[A-Z\\(“\"`]|\s*$)|$)", text):
            sent = s.group(0).strip()
            if not sent:
                continue
            a0 = para.start() + s.start()
            a1 = a0 + len(s.group(0))
            sec = [h for p, h in heads if p <= a0]
            toks = [(m.start(), m.group(0)) for m in vpc.TOKEN_RE.finditer(scan, a0, a1)]
            led = sorted({c for p, _ in toks for c in owners.get(p, [])})
            flags = [k for k, rx in FLAGS.items() if re.search(rx, sent, flags=re.I)]
            out.append({
                "sid": f"s{len(out) + 1:03d}",
                "section": sec[-1] if sec else "Title",
                "sentence": sent,
                "numbers": " ".join(t for _, t in toks),
                "ledger_rows": " ".join(led),
                "all_numbers_covered": int(all(p in owners for p, _ in toks)),
                "flags": " ".join(flags),
            })
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    n_num = sum(1 for r in out if r["numbers"])
    n_unc = sum(1 for r in out if not r["all_numbers_covered"])
    print(f"wrote {OUT.relative_to(PKG)}: {len(out)} sentences, {n_num} with numbers, {n_unc} with an uncovered number")
    for k in FLAGS:
        print(f"  {k}: {sum(k in r['flags'].split() for r in out)}")
    return 1 if n_unc else 0


if __name__ == "__main__":
    sys.exit(main())
