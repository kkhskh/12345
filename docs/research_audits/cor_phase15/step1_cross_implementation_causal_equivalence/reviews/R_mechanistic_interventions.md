# Review R-mech: mechanistic interventions, encoder capacity, collateral damage

I am an independent reviewer and did not produce this package. I read `src/eqv.py`, `../step1_relational_mechanism_deepening/src/fwd3.py`, `scripts/{equiv_run,
necessity,info_structure,site_eval,transport_posthoc,site_train}.py`, the five result documents and `logs/{conf,cal,site}`.
New checks are in `reviews/checks/r_mech_*.py` and their outputs in `logs/review_checks/r_mech_*.out`. Every check uses the committed
checkpoints and the exact streams and RNG order of the script it audits. Each check reproduces the package's own number
before it changes anything: `marg_orig` equals `necessity.json`, and `enc_orig` equals `results.json` to 4 decimals.

Severity: **H** = changes a headline claim, **M** = changes a stated number or how it is read, **L** = wording or hygiene.

## Code audit (no defect found)

- Site and token indexing is correct. Source token t = 6k+s, its group is (t%6)//2, and GROUPS = [(0,1),(2,3),(4,5)]. The readout is 36+q.
  `site_positions`, `site_features`, `encoder_delta`, `soft_delta`, the local-token test and `decode` all agree with each other.
  The model has 3 layers, so layer 2 is the last one (`fwd3.forward` readout = `zf[:, 36:42]`).
- The leave-k counts per token are correct: token (k,s) gets n^{(-k)}_{g(s)}. `single_group_edit` changes exactly one A[k,g] with k ≠ q.
  `pi_preserving_edit` keeps π fixed. Relabel keeps every count.
- `run_frozen` (path patching) replaces attn_out2 / mlp_out2 only at the readout mask, as documented.
- `lstsq` with an intercept plus a full one-hot block is rank-deficient. The minimum-norm solution is still correct for the
  differences that are used. Effective ranks are ≤ 15 (P_full, one token) and ≤ 5 per token (T_own, six tokens).
- Late-binding: the `transport_posthoc.dec` fix is correct. I found no other closure that captures a loop variable it uses
  later: `necessity.summarize` is called inside its own iteration, and `Model` closures capture per-instance `pp`.
- Minor inconsistency: the within-model floor is (n0, n1[shuffled]), while the transport floor is (P0[sh], P1[sh]), i.e. another
  episode's whole delta. Both are legitimate, but they are different nulls, and both are labelled "mismatched" (L).
- `site_eval.py`'s docstring promises pathway fractions, but the script does not compute them (L).

## Findings

**F1 (H): the T2 "ceiling anomaly" is an encoder artefact, not a ceiling artefact.** The T_own encoder is a regression of
a[t] on n alone, so it also absorbs own-question content that correlates with n. In the cond family X_q is fixed, so an
intervention n0→n1 through this encoder also shifts the implied own evidence. I refitted the encoder jointly with
own-question covariates (one-hot of question k's 6 reports plus its 3 agreements) and used only the count block
(`r_mech_interchange.out`). Result for T2: cond IIA drops from 0.648 to **0.516**, r from 0.853 to 0.759 (the ceiling's r is 0.752), and adj
goes from 1.82 to **0.73**. Single-group adj drops from 0.95 to 0.66. T10 and T11 barely change (cond 0.95→0.92 and 0.94→0.91), nor do P2–P4
(IIA within 0.005). The document's handling ("raw IIA 0.648 and r 0.85 are the meaningful numbers") therefore overstates T2.
*Correction:* report T2 cond as IIA 0.52, adj 0.73. Drop "excluding T2" from the range in CAUSAL_EQUIVALENCE_RESULT, giving cond adj
0.73–0.95. Use the covariate-partialled encoder, or report it alongside, for all T models.

**F2 (H): "drop fraction > 1 = collateral damage" uses the wrong reference.** Input-level marginalisation needs no activation
intervention: average the model's own softmax over the same 8 natural draws of X_{-q} | X_q. This already ends *below* own-Bayes
accuracy in every model, with drop 1.02–1.14 (P2 1.14, P3 1.02, P4 1.10, T2 1.07, T10 1.07, T11 1.05). The models' own-question
processing is simply sub-Bayes (`r_mech_marg_partial.out`). Against the natural site-only reference (8 natural draws patched at the site only):
- T10: operator 1.19 vs natural 1.12. The operator is essentially clean.
- T11: 1.37 vs 1.12. A genuine excess of about 0.25 remains, and the partialled encoder does not remove it (1.37).
- T2: 1.14 vs **0.55**. The operator removes twice what the site naturally carries. The partialled encoder gives 0.54, with KL
  reduction 0.74 vs 0.75 natural, which matches. This is the F1 leakage again.

*Correction:* replace "T shows bounded overshoot 14–37 %" with this: the reference drop is ≈1.05–1.14; T10 has no excess;
T11 has about 0.25 unexplained excess; T2's overshoot is encoder leakage, which the partialled encoder removes.

**F3 (M): the P "no collateral damage" claim is not demonstrated.** P's drop of 0.86 sits below 1 because the readout site
does not carry all of the relational information (natural site-only drop 0.82–0.91), not because the operator is gentle. The
norm-matched rotated control, which the document reports only as a KL reduction, destroys a large share of the accuracy gain:
drop fraction 0.27 (P2), **0.59 (P3), 1.14 (P4)** (`logs/conf/necessity.json`). At this delta norm, perturbing P4's readout in
random directions costs more accuracy than marginalising V does. Specificity is carried by the *direction* of the change
(KL toward the own posterior), not by its size. *Correction:* report rotated-control accuracy and drop next to KL. Say that
collateral damage is controlled only in the KL sense.

**F4 (M): Prediction IV(a)'s 0.70 threshold mixes up site completeness with operator specificity.** The operator reaches
95–99 % of the KL reduction that *natural* site-only marginalisation reaches: P2 0.701/0.706, P3 0.830/0.850, P4 **0.677/0.711**,
T10 0.878/0.886, T11 0.860/0.878. P4 "fails" mainly because its readout site can deliver only 0.711. The frozen verdict stands
as frozen. *Correction:* the reading should say the operator is near its site ceiling everywhere, and that P4's shortfall is a
site-completeness property.

**F5 (H): neither curriculum's site is complete, and the T site misses a readout-resident component.** Full-patch ceilings
(cond family, `r_mech_interchange.out` and `r_mech_site_union.out`):

| | own site | own site + other site | all sources + readout |
|---|---|---|---|
| P2 / P3 / P4 (readout; + q's sources) | 0.68 / 0.80 / 0.68 | 0.85 / 0.86 / 0.77 | – / 0.97 / – |
| T2 / T10 / T11 (q's sources; + readout) | 0.55 / 0.84 / 0.79 | 0.94 / 0.96 / 0.95 | – / 0.99 / – |

In T models, 12–39 points of informative natural changes are carried by the readout's own layer-2 input (count information
already moved there by layers 0–1; T2's readout decodes n at 0.93). Floor-adjusted against the two-site ceiling, the encoder
IIAs are P 0.53/0.72/0.57 and T 0.43 (partialled 0.17)/0.73/0.63. "Centralised at the readout" vs "distributed on source
tokens" therefore describes where each model *mostly* holds V, not an exclusive site. *Correction:* state both ceilings and
qualify the localisation claims in CAUSAL_EQUIVALENCE_RESULT §1 and in the E-Int row.

**F6 (H): the T pathway result (attention frozen = 0) is architectural, and it carries no evidence for E-Comp non-equivalence.**
Layer 2 is the last layer and attention is the only operation that mixes positions. With the readout's attn_out2 frozen, a delta
of *random direction* at q's source tokens leaves a maximum |Δlogit| of 1.4e-6 (float noise) in T2, T10 and T11
(`r_mech_interchange.out`, `pathway_random_dir`). The document does note "necessarily so", but CAUSAL_EQUIVALENCE_RESULT 3(b) and the E-Comp
row still use it as a finding. The non-trivial mediator (the readout MLP) does *not* separate the curricula: the MLP-frozen
remainder is P 0.48–0.73 and T 0.44–0.73. *Correction:* drop 3(b) as evidence. E-Comp non-equivalence must rest on the
π-preserving response separation (VI-sep) alone. Under the mediator decomposition actually used, the only difference is
architecturally forced by where the site is, plus the variation among P seeds.

**F7 (M): the P → X transport shortfall comes from restricting the decoder to the readout.** The P readout is genuinely lossy
(F8). But a per-position LDA decoder that reads readout + q's six source tokens decodes P2's n at **0.973**, and with it
transport from P2 becomes exact: P2→P3 0.64→**0.98**, P2→P4 0.64→**0.97**, P2→T10 0.76→**0.98** (`r_mech_transport_dec.out`; same
soft-delta, floor and oracle as transport_posthoc). So the failure of Prediction II(a), and the P→P < P→T asymmetry, reflect
which donor positions the decoder may read. They do not show that P's layer-2 state lacks V. The caveat is that P's source
tokens carry little causal weight (q_src ceiling P3 0.08), so the decoder is reading information the donor barely uses. Only
P2 was tested as a donor. *Correction:* in §2 and in CAUSAL_EQUIVALENCE_RESULT point 2, say "P's *readout* holds a lossy image
of V; P's layer-2 input as a whole determines V (P2: 0.97), and transport from it is exact".

**F8 (confirmation): the P readout's "lossy code" is not an artefact of the decoder specification.** Per-component held-out
accuracy at the readout under several decoders: logistic standardised C=1 0.73/0.76/0.79 (P2/P3/P4), C=100 0.74/0.76/0.80, LDA
0.74/0.76/0.79, MLP(256) 0.73/0.75/0.79, regularised QDA 0.69/0.74/0.75 (`r_mech_decoder.out`). Decoding errors are large (mean
|error| 1.2–1.6 counts) and almost never π-equivalent (1.7–2.7 %), so the readout is not a coarsening to π. This argues against
the D5 "log-odds only" reading for P, which the document leaves open.

**F9 (L–M): the relabel "count channel" IIA is tautological, and the number reported is decoder noise.** Relabelling keeps n,
so every true-count encoder delta is exactly zero and IIA = 0.000 in every model. The values in the table (0.04/0.10/0.04/0.05/0.02/0.01) are
`self_decoded_then_encoded`, i.e. decoder jitter. Prediction III's real content is the full-site patch (0.33–0.62).
*Correction:* label the column correctly and note that it is 0 by construction.

**F10 (M): the "one token of the pair does not suffice" result is partly a dose effect.** Writing the same delta at double
size on one token (`one_token_x2`) changes adj as follows: T2 −0.04 → **0.58**, T10 0.55 → 0.66, T11 −0.10 → −0.10. The consistency requirement (Theorem B1) is
clearly supported only in T11. In T2 and T10, half of the effect is lost simply because half the norm is written.
*Correction:* report the dose-matched control and restrict the B1 "learned instance" claim to T11.

**F11 (M): the T2 redundancy claim ("all_sources = q-only ⇒ no redundant copy") depends on the leaking encoder.** By F2 and F5,
T2 holds about half of its natural relational effect at the readout. Both the all-sources and q-only operators overshoot that
through own-content leakage, so their equality says nothing about redundancy. *Correction:* withdraw the claim for T2 and
re-test it with the partialled encoder.

**F12 (L): the site test (Prediction V) cannot detect a variable that moved but stayed distributed.** In the S models the single-site
ceilings are low (0.23–0.46), but the unions are not: readout + q_src 0.54–0.64, all sources + readout **0.82–0.99**
(`r_mech_site_union.out`). The relational effect is jointly spread over readout and source tokens, in a strongly non-additive way (S6: 0.36
and 0.23 alone, 0.99 jointly). The document's "not localised at either site" is correct. "Does not move a causally used count
variable to the source tokens" should be softened to "not to q's source tokens alone".

## Verdict table

| claim (document) | verdict |
|---|---|
| Implementation (indices, leave-k counts, edits, path patching, transport binding) | **correct**; minor floor-naming inconsistency |
| E-Int above floor in all six (CROSS §1) | holds; **T2 is IIA 0.52 / adj 0.73** after removing encoder leakage (F1) |
| T2 anomaly "full patch is not a ceiling" | **incorrect diagnosis**: encoder leakage (F1) |
| Localisation: P at the readout, T at q's sources | **overstated**: neither site is complete; T's readout carries 12–39 pts (F5) |
| P readout code is lossy (decoder robustness) | **confirmed** (F8), and not π-coarsening |
| P → X transport partial because P holds lossy V | **readout-specific**: exact (0.97–0.98) with a readout + source decoder for P2 (F7) |
| T pathway (attention-only) as evidence against E-Comp | **architectural tautology** (F6); MLP route does not separate curricula |
| Marginalisation specific vs null (KL 0.68–0.88 vs 0.01–0.02) | holds; operator reaches 95–99 % of the natural site ceiling (F4) |
| T overshoot = collateral damage | **mis-referenced**: natural reference 1.05–1.14; real excess only in T11 (≈0.25); T2 is leakage (F2) |
| P "without collateral damage" | **not shown**: rotated control drop 0.59 (P3) / 1.14 (P4) (F3) |
| Redundancy absent in T (all_sources) | **not supported for T2** (F11) |
| One token insufficient ⇒ consistent writing (B1) | **T11 only**; T2 and T10 are partly dose (F10) |
| Relabel count channel ≤ 10 % | tautological (0 by construction) (F9) |
| Site test failed (Prediction V) | holds as frozen; distributed relocation is not excluded (F12) |

Net effect on the main answer: "same variable, different downstream computation" weakens on both sides. The variable-level
case for P improves, since transport out of P2's whole layer-2 state is exact (F7). The computation-level separation loses
the pathway argument (F6) and now rests on the π-preserving response separation (VI-sep) alone.
