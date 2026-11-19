# What XVIII established, what it did not, and why

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Sources.**
- XVIII package `step1_pathway_selection_causal_identification/`, HEAD `7fb7dad`, freeze `1547ec9`.
- Its primary outputs: `logs/score_XVIII.json`, `logs/posthoc_nobehav18.json`, `logs/score_cont18.json`, `logs/posthoc_parents18.json`.
- Its four reviews.

I re-checked the claims below against those outputs, not against the synthesis.

## 1. The failed primary evaluator

**Rule.** XVIII's frozen evaluability rule required the fresh baseline's organisation class to be unanimous across three diagnostics.

**Weak link.** One of the three was a behavioural gate (`cls_behav2`):
- EARLY needed lapse ≥ 0.017 and ρ_L ≤ 0.75;
- MIXTURE needed lapse ≤ 0.017 and ρ_L ≥ 0.68;
- the thresholds were calibrated on 24 XIV checkpoints with the development stream.

**On the confirmation stream (95_400_000), 4 of the 6 baselines failed the gate:**
- E|811: lapse 0.010;
- E|813: lapse just under 0.017;
- M|811: lapse 0.018;
- M|813: ρ_L 0.64.

Their causal classes were unambiguous: E baselines nec_E 1.36–1.58, nec_M ≤ 0.006; M baselines nec_M ≈ 0.65, nec_E ≤ 0.04.

**Consequences:**
- Only 2 pairs were evaluable, so every arm band was INCONCLUSIVE.
- The frozen sensitivity analysis removed the gate from the bands but **not from evaluability**: an implementation error.
- The same gate made the one bistable independent parent (uP163) non-evaluable.

**Lesson carried into XIX (mandate §9).** No fitted behavioural parameter may decide whether a competent baseline can be scored. Thresholds must be validated on streams that were not used to set them, before the freeze.

## 2. What survived, and at what level

| statement | level | n |
|---|---|---|
| Six of the eight frozen hypotheses are contradicted (H1d, Hm_window, H2g_R, H2g_RV, H4, H1d_plus_Hm); Hm_release and H1d_plus_H2g are inconclusive | frozen | 2 pairs |
| Under a literal reading of an ambiguous clamp-validity rule, Hm_window and H1d_plus_Hm are inconclusive instead | frozen, alternative reading | 2 pairs |
| vrg (content destroyed, value gradient stopped) switches the organisation | sensitivity | 5/5 evaluable pairs; uP163 1 F, 1 P |
| sha-hi-rolled switches; sha-lo-real never switches | sensitivity | 5/5 (sha-lo-real also N in the non-evaluable M\|813); uP163 2/2 each |
| gsv (all gradient through T's edges removed) switches fully on M; on E it switches route necessity but not behaviour | sensitivity | E 3/3, M 2/2; uP163 M F, E P |
| gsg is a stable hybrid on M (φ ≈ 0.36 at step 2546) and strong on E | sensitivity + stability readout | 3 + 3 |
| Odds × 0.1 suppression switches the organisation, so a complete block is unnecessary | sensitivity | 5/5 |
| Only 1 of 5 independent candidate parents is bistable | frozen screening | 5 parents |

## 3. The causal confounds XVIII left open (mandate §1.2)

1. **vrg confounds forward content with value learning.** It rolls the values *and* stops their gradient. The value-gradient-only arm (vsg) was never run.
2. **gsg and gsv are not complete removals of route learning.**
   - XIX's graph audit (`logs/graph_audit19.json`) measures how much leaks. Under gsg the routing signal moves wholesale onto the complement logits: d loss/d(complement bias) ≈ +0.008 on E at step 0, matching plain training's −0.008 on T.
   - Under gsv, upstream parameters still learn through other edges.
3. **sha-lo-real lets the latent route keep training**, so the hidden share rebounds at release.
4. **Graded suppression changes forward contribution and both gradients together** (lemma L2).
5. **No matched non-targeted disturbance arm was run.** Trajectory-loss H3 on E is therefore not excluded.
6. **Necessity is not computation.** XVIII never established that the switched models compute a different *function*. Its behavioural gate was weak, and on E, gsg and gsv switched necessity without switching behaviour.
7. **The state left by the window was never examined.** The intervention ends at step 100, yet XVIII only examined outcomes at step 1211. Which part of the step-100 training state carries the selection was untested.

## 4. Withdrawn statements (XVIII review C)

These are not to be resurrected under new wording:
- "The organisation switches when route T receives no useful learning signal."
- "Restricting a route's attention share does not change the organisation." Graded suppression contradicts it.
- "gsg is partial on M because M relies on recruiting T."
- "Complete double dissociation on every run."
