# Scientific protocol (index)

- Design and design-gate answers: `protocols/DESIGN_XX.md`
- Single source of every constant: `cor_research/protocol.py`; machine-readable copies: `protocols/frozen_hypotheses/track_a_hypotheses.json`,
  `protocols/causal_design/track_a_design.json`, `protocols/functional_design/track_b_design.json`, `protocols/od_design/track_c_protocol.json`,
  `configurations/presets.json`
- Freeze: `protocols/FREEZE.json` (sha256 of every protocol, configuration, script and code file; verified by the runner before any job;
  re-freezes are appended to `protocols/FREEZE_HISTORY.json`, never edited)
- Pre-run reviews, adjudication and re-reviews: `reviews/`
- Development observations (not evidence): `dev/DEV_NOTES_XX.md`
- Delivery copy of the frozen protocol in prose: `INVESTIGATION_XX_FROZEN_SCIENTIFIC_PROTOCOL.md` (outside the ZIP)
