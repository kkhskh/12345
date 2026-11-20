"""Write the delivery copy of the frozen protocol (INVESTIGATION_XX_FROZEN_SCIENTIFIC_PROTOCOL.md) from the frozen sources: DESIGN_XX.md,
protocol.py constants and FREEZE.json.  python scripts/make_delivery_docs.py OUT_DIR"""
import os, sys, json
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG); os.environ.setdefault("COR_DEVICE", "cpu")
from cor_research import protocol as PR, env as E
out = sys.argv[1]; P = PR.full(); fz = E.read_json(os.path.join(PKG, "protocols", "FREEZE.json")) or {}
L = ["# COR Investigation XX — frozen scientific protocol (delivery copy)\n",
     f"Freeze: `protocols/FREEZE.json`, freeze_sha256 **{fz.get('freeze_sha256', 'NOT FROZEN')}**, frozen {fz.get('frozen_at_utc')}, {len(fz.get('files', {}))} files.",
     "This document is generated from the frozen sources; the machine-readable protocol (protocols/*/*.json, generated from cor_research/protocol.py) is authoritative.",
     "Phase 1.5, Step 1 (≈ 25% provisional). No result exists until the package is run. Reviews are AI-agent reviews (reviews/), not human peer review.\n",
     "## Frozen Track A hypotheses\n", "| id | base | arm | expected | step | role | statement |", "|---|---|---|---|---|---|---|"]
L += [f"| {h['id']} | {h['bases']} | {h['arm']} | {h['expect']} | {h['step']}{' (+' + str(h['co_primary_step']) + ')' if h.get('co_primary_step') else ''} | {h.get('role')} | {h['text']} |" for h in P["HYPOTHESES_A"]]
L += ["\nExploratory (no verdict):"] + [f"- {h['id']} ({h['bases']}, {h['arm']}): {h['text']}" for h in P["EXPLORATORY_A"]]
L += ["\n## Arms\n", "| arm | kind | bases | role |", "|---|---|---|---|"] + [f"| {a} | {d['kind']} | {d['bases']} | {d['role']} |" for a, d in {**P["ARMS"], **P["HIGH_EXTRA_ARMS"]}.items()]
L += ["\n## Track B (frozen)\n", "```json", json.dumps(P["TRACK_B"], indent=1), "```", "\n## Track C (frozen)\n", "```json", json.dumps(P["TRACK_C"], indent=1), "```",
      "\n## Rules, thresholds, streams, presets\n", "```json", json.dumps({k: P[k] for k in ("SCREEN", "COMP", "CLASS", "CONF_D", "PILOT_D", "STREAMS", "STEPS", "BAND", "DELTA", "EPS", "EPS_SENS", "MIN_PAIRS_PER_PARENT", "MIN_PARENTS", "PARENT_RULE", "CROSS_RULE", "VALIDITY", "PARENT_VALIDITY")}, indent=1), "```",
      "\n```json", json.dumps({k: {x: v[x] for x in ("label", "parents", "n_D", "stability", "claims_can", "claims_cannot") if x in v} for k, v in P["PRESETS"].items()}, indent=1), "```",
      "\n---\n\n## Design and design-gate answers (protocols/DESIGN_XX.md, verbatim)\n", open(os.path.join(PKG, "protocols", "DESIGN_XX.md")).read()]
open(os.path.join(out, "INVESTIGATION_XX_FROZEN_SCIENTIFIC_PROTOCOL.md"), "w").write("\n".join(L) + "\n"); print("written")
