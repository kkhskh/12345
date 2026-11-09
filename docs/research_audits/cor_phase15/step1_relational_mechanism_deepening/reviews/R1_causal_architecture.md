# R1 — Adversarial review: causal inference, Transformer architecture, representation/invariance

Scope: GATE_FULL_CAUSAL_CIRCUIT.md, PCOUNT_CAUSAL_TRANSFER.md §A3–A5, CAUSAL_IDENTIFICATION_AND_INVARIANCE.md;
src/fwd3.py (against ../step1_latent_reliability/src/nets.py); scripts/gate_circuit.py, gate_redundancy.py,
gate_headpairs.py, pcount_causal.py; logs/gate/*.json, logs/pcount/*.json.
Reviewer checks were small re-runs on the stored checkpoints (no training). Each used ≤ 3000 episodes, one process at a time,
and finished in under 2.5 min. The scripts are in /tmp/r1chk/ (src_enc.py, rd_enc.py, gate_chk.py, abl_pos.py) and are
not part of the package. Their numbers are reviewer evidence and were not frozen.

Severity: **CRITICAL** = a headline conclusion is overturned; **MAJOR** = a conclusion is mis-stated or needs a
qualifier it lacks; **MINOR** = inconsistency or omission.

---

## 1. [CRITICAL] The A5 / §4 negative result for T1 is an encoder artefact: T1's source tokens do carry an additive one-hot count code

`pcount_causal.py` fits **one** source-site encoder that is shared by all 36 source tokens:
`Fs = np.repeat(onehot(nn), 6, axis=1).reshape(-1, 18); Ws = lstsq(..., an[:, :36].reshape(-1, d))`.
This forces the count code to point in the same direction at every source slot, so every slot gets the same delta,
whether it belongs to group 0, 1 or 2. A per-source reliability tag is slot- or group-specific by construction: a
source in group g should change with group g's count, possibly with opposite sign for the other groups. Pooling over
slots averages these directions and they cancel. Keeping the authors' alignment class (linear, additive, one-hot
leave-k counts, generator-defined, fitted on natural data with no outcome information) and only giving each slot its
own encoder gives:

| T1 transfer model, source site, cond pairs, IIA | ref_T1 (s0) | ref_T1p1 |
|---|---|---|
| original shared-slot encoder (reproduces the log) | 0.040 | 0.063 |
| **per-slot linear encoder, leave-k counts** | **0.821** | **0.801** |
| per-slot, full-episode counts | 0.806 | 0.818 |
| per-slot, only the slot's *own group's* count (6-dim) | 0.765 | 0.805 |
| per-slot, delta written only at question q's 6 tokens | 0.813 | 0.802 |
| per-slot joint-count lookup (non-additive control) | 0.807 | 0.788 |
| full-site source patch (ceiling) | 0.878 | 0.839 |

(1000 evaluation episodes, about 2300 informative pairs; encoder fitted on stream 9_910_000 as in the script.)
The per-slot additive count code recovers 93–96 % of the full-site interchange. A non-additive lookup adds nothing. The
own-group count alone recovers most of it. The same per-slot encoder in P-count gives 0.22 / 0.24 at the source site
(s0 / s1), which is close to that site's own full-site ceiling (0.15–0.29). At the readout, a per-question encoder
changes nothing (P-count 0.69 → 0.70, 0.66 → 0.67; T1 0.075 → 0.073).

Consequences:
- These statements are **refuted** in the linear alignment class: PCOUNT §A3(2) ("not in that code … reconstructed into
  a different variable"), §A5 ("T1's is not [causally effective] at either site … do **not** implement the same
  intermediate variable in this alignment class"), CAUSAL_IDENTIFICATION §4 bullets 2–3, and RELATIONAL_COMPUTATIONAL_CLASSES
  row "same relation, different codes".
- The supported reading is: **the same abstract variable (leave-q per-group agreement counts, additive one-hot),
  causally used, at different sites** — the readout token in P-count, slot-specific at the source tokens in T1.
- CAUSAL_IDENTIFICATION §6 says the interchange is "invariant under any invertible linear reparameterisation of the
  site". That holds per token, but the conclusion depended on a hidden cross-slot weight-sharing assumption that is not
  part of the stated abstraction.

## 2. [MAJOR] The alignment class favours P-count by construction

P-count pretraining (`train_tasks.loss`) reads exactly these targets: a **linear** head `WX` on the final-LN readout
features, predicting the leave-q counts of the 3 groups as 3×6-way softmaxes. So a linear one-hot count code at the
readout is what the P-count objective trains for. The "common coordinate system" (CAUSAL_IDENTIFICATION §1) is
therefore P-count's own output parameterisation. That P-count passes and another curriculum might fail is a weak
comparison unless the encoder is equally flexible across sites and slots (see 1). The documents should say this.

## 3. [MAJOR] The necessity controls in §A4 / CAUSAL §2.3 are built with pooled cross-position statistics; the "non-specific" verdict is partly an artefact

`pcount_causal.py` takes the replacement mean and the variance-matched PCs from activations **pooled over positions**:
`a0[:, pos].reshape(-1, d).mean(0)`, and `Zc` is centred on the pooled mean. At the readout site the pooled data
contains the variance between the 6 readout positions (E_q). At the source site it also contains value and source
identity. So (a) the "variance-matched" subspace mostly spans *position-identity* variance, and (b) mean-replacement
over the source site erases slot identity inside the subspace, which is off-manifold. Re-running with per-position
centring and per-position means (1000 episodes):

| drop fraction of relational gain | count subspace | random | variance-matched (per-pos. centred) | log (pooled) var-matched |
|---|---|---|---|---|
| P-count s0, readout | 0.63 | 0.06 | 0.32 | 1.33 |
| P-count s0, source | 0.51 | 0.03 | 1.22 | 1.87 |
| T1 s0, source | 1.74 | 0.01 | 0.11 | 1.68 |

With correct controls, readout ablation in P-count and source ablation in T1 are specific: count > variance-matched >
random. The statement "we make no necessity claim from ablation" is cautious, which is fine. But the reason given (a
variance-matched subspace does more damage) comes from the pooling, not from the model. The doc also reports only
seed 0. In seeds 1–4 the pooled count-subspace drop is 1.08–1.37, which is above the full relational gain, so it is also
damaging own-question processing.

## 4. [MAJOR] GATE "jointly necessary pair" / backup claims depend on the ablation operator and the metric; the pair is not jointly necessary for the gate decision

The headline numbers (headpairs 0.883 s0, 0.936 s1) are unique-argmax Bayes agreement under **mean** ablation. Two
problems:

(a) The metric does not isolate the gate. The gate-blind reference (Bayes with the gate marginalised at ½) already
agrees with Bayes on **0.890** of unique-argmax questions. Seed 0's "jointly necessary" pair (0.883) and seed 1's
(0.936) are therefore at or above the gate-blind level. The direct measure is the share of scored gate flips the
ablated model still follows, with both runs ablated:

| | none | single heads | pair | all L2 heads |
|---|---|---|---|---|
| s0 L1h2 / L2h0 | 1.00 | 1.00 / 1.00 | **0.86** | — |
| s1 L2h1 / L2h3 | 1.00 | 1.00 / 1.00 | **0.92** | **0.93** |
| s2 L2h1 (or L1h3) | 1.00 | **0.51** | — | — |

So the s0 and s1 pairs remove only 8–14 % of the gate effect. A third route carries the rest. In s1 this is the readout
residual before layer 2: it decodes the gate at 0.996 (B2 log), yet patching it transfers only 0.065 while layer 2 is
intact, which is the pattern of a backup.
"Jointly necessary" (CAUSAL §3 identification statement; GATE §2 row; §3 bullet "Seed-specific") is not supported.
The supported claim is "jointly *contributing*, with further redundancy". The s1 description "a redundant pair within
layer 2" misplaces the redundancy: it sits between layer-2 attention and the earlier readout residual.

(b) Mean ablation is not a neutral operator here. Resample ablation (head output taken from a random other episode)
reverses the single-head conclusions: s0 L1h2 alone → **0.623** (mean: 1.000); s1 L2h1 → 0.988, L2h3 → 0.980; s0 pair
→ 0.374; s1 pair → 0.391. These heads carry the evidence *values* (attention from R to E: 0.97–0.99), not only the gate
selection. Resample therefore injects another episode's answers, and mean ablation lets the residual stream route
around the gap. Neither result is "the head is (un)necessary for the gate". The redundancy statement in CAUSAL §6
("split-invariant") is invariant to head splits but **not to the ablation operator**, which matters more here. Recommended
instrument: gate-flip following under ablation (above), or ablation that resamples only from gate-flipped twins.

## 5. [MAJOR] The "conditional sufficiency … 1.000 in all seeds" test is informative only for seed 0

`gate_redundancy.py` always ablates `attn_out1` at R and then patches `attn_out2`. In seeds 1 and 2, `attn_out2@R`
already transfers 0.998 / 1.000 with layer 1 intact, and `attn_out1@R` carries almost nothing (0.07 / 0.00). For these
seeds the test is a re-run of the unconditional patch, not evidence of a backup. The GATE §2 sentence lists the three
seeds side by side as if they were the same result.

## 6. [MINOR→MAJOR] Mean-ablation scores near 0.66 are default-state collapse, not loss of the gate

Seed 2 single-head ablations (L1h3 0.664, L2h1 0.663) are far *below* the gate-blind 0.890. The flip-following of
0.505 is identical for both ablations to six digits. Together these show that the ablated model **commits to one gate
state** rather than marginalising. This fits a single route, but "agreement 0.663" should not be read as "the gate's
contribution is 34 % of decisions".

## 7. [Supported, with checks] Hooks, positions, recomputation, and flip scoring are implemented correctly

- `fwd3.forward` decomposes `einsum("nhte,hed->ntd")` into `head_out` (`nhtd`) followed by `.sum(1)`, so it matches the
  original exactly. A `head_out` patch propagates into `attn_out`, and an `attn_out` patch overrides the heads at the
  positions patched. Patches act on residual-stream contributions **before** LN, and every LN and every later layer is
  recomputed inside the same forward (full downstream recomputation). The logs record `equivalence_max_abs_diff`
  0.0; the doc says 1.7·10⁻⁶ (see 11).
- Positions are correct: G = (6q, 6q+1), E = 6q+2..6q+5, R = 36+q, T = 43.
- The gate flip is on-support: open → x₁ redrawn among the two closing values; closed → x₁ = −x₀. Evidence is kept.
- Scoring: patched argmax compared with the flipped Bayes argmax on cases where the Bayes argmax changes and both are
  unique. The ceiling (the unpatched model following the flip) is 1.00 in all seeds (reviewer check), so the
  transfer numbers need no rescaling. The negative controls are appropriate.
- The scored set comes from the same 3000 episodes in all seeds (n_scored 3211), so cross-seed differences are not
  sampling differences.

## 8. [Supported, strengthened] Representation claim (B2): the gate is stored at E as an abstract variable, not as the pair identity

The concatenated 4×64 E-residual could in principle encode the 9 (x₀, x₁) pair identities, from which a linear probe
also reads the gate. Reviewer control: a probe trained on two x₀ values and tested on the held-out x₀ (1 open + 2 closed
pairs never seen in training) gives 0.976–1.000 (s0), 0.994–1.000 (s1), 1.000 (s2) at the layer-2 input. This supports
"gate state as a variable" over "pair identity". The doc should include this control; without it, "stored variable"
was not identified.

## 9. [Supported] "Decision by readout attention to the evidence tokens" — the pattern evidence was missing from the doc

GATE §3(iii) is supported only by output patches. Those cannot tell R→E from R→G reads. Reviewer check of
readout-row attention mass for question q (G / E): decisive heads s0 L1h2 0.001 / 0.988, L2h0 0.004 / 0.966;
s1 L2h1 0.005 / 0.969, L2h3 0.006 / 0.933; s2 L2h1 0.001 / 0.987. Together with resid_pre@E transfer (0.83 / 0.93 /
1.00) this supports R→E routing. Caveat for seed 0: `attn_out1@R` transfers 0.97 but `resid_pre1@E` only 0.83, so about
14 % of the flip reaches R by another path. It is not via G (attention ≈ 0), so it is likely the R residual's own
layer-0 content. The doc does not report this gap.

## 10. [MAJOR] Cross-family "shared structure" needs restating after finding 1

The synthesis claims, for T4 and GATE, "relation variable at source tokens → readout attention" (structural, not
causal). For GATE this is supported (findings 7–9). Finding 1 changes the transfer picture. It no longer shows
"same relation, different codes". It shows the **same count code at different sites, depending on the curriculum**:
P-count stores it at the readout, so its transfer model is *not* an instance of "variable at source tokens". "Relation
variable at source tokens" is therefore a property of curricula that supervise the answer (T1/T4/GATE). It is not a
property of the relation, so it should not be stated as family-invariant. The claim that it is "structural, not causal"
across families stays correct: no interchange across tasks is possible.

## 11. [MINOR] Numerical and reporting inconsistencies

- GATE doc: equivalence 1.7·10⁻⁶ vs logs 0.0.
- CAUSAL §4: "2-D contrast subspace 0.72" vs logs `kdim: 4` (PCOUNT §A3 correctly says k = 4).
- Relabel change rate: PCOUNT "7–15 %", CAUSAL "9–14 %". Logs: T1 0.069–0.094, P-count 0.088–0.148, P-group 0.152.
- "Selectivity 1.7 %" is seed 0 only (`relabel|readout|count_subspace` change rate). Seeds 1–4: 2.8–4.6 %.
- The CAUSAL §1 table names "mean, resample" as necessity instruments, but GATE uses only mean ablation (see 4b).
- P-local's ablation drop fractions in the log are ~10⁷ because its denominator acc_clean − acc_own ≈ 0. Those entries
  should be marked undefined.
- `pcount_causal.py` §(5): the source delta is also written at the tokens of the resampled questions k ≠ q, using their
  leave-k counts while their contents stay at the base values. This is harmless: writing at q's tokens alone gives the
  same IIA (finding 1, 0.813 vs 0.821). It is still worth stating.

## Summary of verdicts

| claim | verdict |
|---|---|
| hooks/positions/LN/recomputation; flip-transfer scoring | correct |
| GATE: gate computed at E by attention, stored as abstract variable; decided by R→E attention | supported (strengthened by reviewer controls 8, 9) |
| GATE: "jointly necessary" backup pairs (s0, s1); s1 redundancy "within layer 2" | **not supported** — pairs remove 8–14 % of gate effect; operator-dependent |
| GATE: conditional sufficiency in all seeds | informative only for s0 |
| P-count: readout count variable causally used (0.60–0.72) | supported; alignment class favours it (2) |
| T1: cross-question info at source not in additive count code; curricula not interchangeable | **refuted** — per-slot linear count encoder gives 0.80–0.82 |
| necessity by ablation non-specific | artefact of pooled means/centring; with per-position controls, specific |
| cross-family shared structure | GATE side ok; transfer side must be restated (10) |
