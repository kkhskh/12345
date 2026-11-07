# Leakage and sample-reuse audit

Pair-manifest facts below were computed from `data/raw/*/..._pair_manifest.json` and `data/raw/nst/null_state_split_leakage.json`. FRESH and VNF facts come from the console excerpts in `data/archival/console_excerpts/`. To recompute the manifest counts:

```
python - <<'EOF'
import json
S = json.load(open("data/raw/spec/fiber_spec_pair_manifest.json"))
F = json.load(open("data/raw/fixed_js/fixed_js_pair_manifest.json"))
N = json.load(open("data/raw/nst/null_state_pair_manifest.json"))
g = lambda ps: {p["recip_game"] for p in ps} | {p["target_game"] for p in ps}
print(S["pairs"] == F["pairs"], len({p["recip_game"] for p in S["pairs"]}), len({p["recip_game"] for p in N["pairs"]}), len(g(S["pairs"]) & g(N["pairs"])))
wd = [d["game_id"] for w in S["wrong_donors"] for d in w["donors"]]
print(len(wd), len(set(wd)), len(set(wd) & {p["recip_game"] for p in S["pairs"]}), len(set(wd) & {p["target_game"] for p in S["pairs"]}))
EOF
```

## Which samples are shared

| Study | Pairs or triplets | Shared with | Consequence |
|---|---|---|---|
| FIBER | 32 validation pairs | excluded from SPEC (`exclude_fiber_games` = 1) | diagnostic only |
| SPEC | 48 validation pairs, manifest `2789accf…1143d3` | matched-JS study (identical pair list and hash) | the matched-JS study is a follow-up on the same sample, not a replication. Its first execution deviated from protocol and keeps its inconclusive letter. |
| Matched JS | the SPEC 48 | SPEC | as above |
| NST | 48 validation pairs, manifest `39abdc79…5092e6` | SEM reuses them and NST's frozen basis | SEM is exploratory; its FULL arm is the NST map |
| SEM | the NST 48 | NST | as above |
| FRESH | 32 triplets from a new 400-game cohort, seed 9182 | none at game level | 24 board-and-player states of the cohort also occur in earlier splits; whether any reach the selected triplets cannot be checked without the triplet arrays |
| VNF | 32 triplets from a new 400-game cohort, seed 10182 | none at game level | the console excerpt reports 400 kept games and 0 dropped; it records no state-level overlap count, so state overlap is unknown |

SPEC and NST share no recipient or target games. In both studies each recipient game and each target game appears in exactly one pair, and no pair's recipient game equals the target game. NST's split-leakage record lists no overlap between training, model-selection and evaluation games, between recipients and targets, between evaluation and prior-study games, or between wrong targets and the hold-out or prior games.

## Dependence not captured by the bootstrap

Because each recipient game appears in one pair, clustering on recipient game is the same as resampling pairs. Pairs are still not fully independent in SPEC and the matched-JS study. The 144 wrong-target donors come from 122 distinct games, and 18 of those games are recipient games and 13 are target games of other pairs. A game that is a wrong-target donor in one pair and a recipient in another links the two pairs' responses. The bootstrap does not model this. In NST the 48 wrong targets come from 35 distinct games, none of which is a recipient or target game of any pair. In FIBER, the 24 category-A pairs share only two target games, which the manuscript states.

## Selection and outcome leakage

- Pairs were selected from simulator features: player, occupancy, legal-set sizes and prefix length. No model response was used, and no pair was excluded after seeing \(Y\).
- The JS calibration searched over dose without reading \(Y\).
- The NST map's \(k\) and \(\alpha\) were chosen by grouped cross-validation on training games before any held-out response was computed. The oracle direction was computed only after the learned directions were frozen.
- The 0.05 floor for SPEC, the 44 of 48 coverage gates for the matched-JS study and NST, the 0.01 floor for NST and SEM, the 0.005 floors and 60% gate for FRESH and VNF, and the bootstrap settings are all in preregistrations written before the corresponding outcomes. `scripts/verify_paper_claims.py` checks each quoted value against the protocol text.
- The sealed confirmation split was never opened. No file whose name refers to a confirmation protocol is in the package; the verifier checks this.

## What the reuse means for the claims

The evidence comes from three independent samples. The SPEC pairs support SPEC and the matched-JS study. The NST pairs support NST and SEM. The FRESH and VNF cohorts support the two completeness tests. The positive kernel results therefore rest on two samples of 48 pairs, each reused. The two later independent cohorts are the ones whose completeness tests failed. The manuscript's Limitations section says this.
