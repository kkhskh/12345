# Diagnosis of the original P-count R1 failure (Workstream A)

**Evidence levels.**
- **[D]** development: parents xP80/xP81, `eval_dev`.
- **[R]** retrospective: Investigation VIII lineages zP100–107 and their existing evaluations.
- **[C]** confirmed under the frozen protocol: parents yP130–135, `eval_conf9` (see `PROSPECTIVE_PREDICTIONS.md` Part II).

**Definitions.**
- "C1" is the original transfer: frozen pretrained layers 0–1, a fresh layer 2 trained on T4 labels.
- Native states are the residual streams of the frozen pretrained network. resid_pre2 is C1's input; "final" is the
  output of all three pretrained layers.
- absE is the per-episode mean absolute response error.

## 1. A covariate nobody had looked at: pretraining completeness [R, D; C in CP1/CP2]

P-count pretraining (3000 steps) is **bimodal**. The pretrained model's own count accuracy is either ≈ 1.0 or stuck at
0.52–0.79. At those values one or two groups' counts were never learned: on xP80, group 1's count decodes at 0.50 from
the final readout token, while groups 0 and 2 decode at 0.999–1.0.

| lineages | pretraining count accuracy |
|---|---|
| complete | xP81–84, zP101, zP103–105, yP130–132, yP134 |
| incomplete | xP80 0.77, xP85 0.94, xP86 0.79, xP87 0.52, zP100 0.73, zP102 0.76, zP106 0.75, zP107 0.52, yP133 0.52, yP135 0.56 |

**Retrospective table** (`scripts/completeness_retro.py`): Investigation VIII's original transfers on zP100–107, eval_conf8.

| group | n | KL | absE R2∪R3 | absE R1 | E R1 |
|---|---|---|---|---|---|
| complete | 4 | **0.048** (0.036–0.062) | 0.91 | **0.95** | −0.91 |
| incomplete | 4 | **0.140** (0.111–0.202) | 1.70 | 0.68 | −0.57 |

Completeness separates KL perfectly: every complete lineage is below every incomplete one. It also roughly halves the
R2/R3 error. **It does not explain R1**: complete lineages have the *larger* R1 error. Two failures were therefore
superimposed in earlier investigations:

- **F-I (information).** Incomplete lineages lack a group's count, so KL and R2/R3 errors are large. This is H1, and it
  belongs to the parent, not to the relational computation.
- **F-R (relational).** Even with every count present, the R1 sign fails. It must be explained by H2–H7.

## 2. Is the information present and accessible? (H1, H2) [D; C in CP1, CP4]

**Probes on the complete parent xP81, final pretrained states.**
- Per-group, per-position linear probes recover every group's count from the readout tokens with accuracy **1.000**,
  with zero 4-vs-5 confusions.
- Linear probes recover each source's own report from its source token with accuracy 1.000.
- Plugging the decoded values into the exact formula (`scripts/decode_compose.py`) gives **KL 0.000 and absE 0.000 on
  every edit family**. No learned combiner is involved.

The ingredients are *linearly* present and complete. H1 is rejected for complete parents.

**At C1's input (resid_pre2), one layer earlier**, counts decode at 0.85 (4-vs-5 confusion 5 %), and the composed
posterior reaches KL 0.071. The original protocol discards the pretrained layer that finishes counting. Its input is
therefore partly information-limited. Keeping that layer does not help R1, though (§4).

**Native-state-only learned readers on xP81** use no side channels: the readout token plus the 6 source tokens of
question q.

| reader (native states) | KL | absE R1 | E R1 | absE R2∪R3 |
|---|---|---|---|---|
| mlp_q, final states, labels 8k | 0.014 | 0.40 | **+0.01** | 0.41 |
| mlp_q, final states, Bayes targets 32k | **0.000** | **0.074** | +0.006 | **0.064** |
| mlp_q, resid_pre2, labels 8k | 0.026 | 0.31 | +0.00 | 0.82 |
| mlp_q, resid_pre2, Bayes 32k | 0.005 | 0.18 | +0.05 | 0.27 |
| ds_src (per-source φ, sum, ρ), final, labels 8k | 0.014 | 0.39 | −0.02 | 0.47 |
| pair_src, final, labels 8k | 0.016 | 0.41 | +0.03 | 0.54 |
| **C1, the original one-block transfer (pool trainer), labels 4k** | 0.042 | **0.58** | **−0.55** | 1.23 |
| keep3: all pretrained layers frozen + fresh block, 4k | 0.061 | 1.49 | −1.49 | 0.86 |

A native-state reader of the complete parent reaches Bayes. This is **Type II sufficiency** in the mandate's
classification, in its strongest form: a linear decode plus the exact rule, and also learned. The native information is
accessible to a small reader. H2 is rejected for the final states and holds only partly for resid_pre2.

On the incomplete parent xP80, every native reader on q-tokens stalls at KL ≈ 0.10. The cap-at-4 information floor
(`MATHEMATICAL_STRUCTURE_AND_LIMITS.md` §5) is 0.081.

## 3. What the original downstream block computes instead (H3, H4, H7) [D; C in CP5]

On C1 trained on xP81, `scripts/mech_native.py`:

- **Attention routing (from readout token q to each group's 12 source tokens), by the group's own count:**

  | n_g | 0 | 1 | 2 | 3 | 4 | **5** |
  |---|---|---|---|---|---|---|
  | attention mass | 0.22 | 0.28 | 0.38 | 0.51 | 0.54 | **0.18** |

  The block implements reliability routing that rises with count, and then a **collapse for perfect groups**, applied
  to each group independently. On xP80 it is the same: flat at 0.36–0.40, then 0.20 at n = 5.
- **Confirmation (CP5, 6 independent parents).** Routing carries ≥ 50 % of the R2/R3 response in 6/6, and the freeze makes the R1 response less negative in 5/6. The frozen ratio criterion (perfect-group mass < 0.6 × n=4 mass) held in only 3/6, so CP5 as frozen is falsified.
- **Attention freeze.** Run the post-edit episode with the block's attention pattern frozen to its pre-edit value. The R2
  and R3 responses shrink by 83–91 % (−2.87 → −0.50, −2.62 → −0.24), so they are carried by routing. The wrong-signed R1
  response only halves (−0.21 → −0.12; on xP80 −0.59 → −0.30), and the rest flows through the value/MLP pathways.
- **Token patching.** Patch the post-edit value of the readout token q alone: it reproduces ≈ 100 % of the R1 response
  and about 2/3 of R2/R3. Patching the edited group's source tokens at q alone overshoots the R1 response (share 2.2).
  The pathways are therefore non-additive.

**Correction (`scripts/mech_routing2.py`).** The routing is not purely per-group.
- A perfect group receives 0.135 when it is the only perfect group and 0.326 when another is.
- But on the F45 edits, the edited group's mass falls 0.54 → 0.32 *identically in R1, R2 and R3*.

**Reading.** The original block learned a rule that is **regime-blind**: "a newly perfect group loses attention", by
the same amount whatever the regime. The R1 regime requires the opposite sign. That rule is correct in the 91.6 % of
decisions with a single perfect group, and it gives the right sign on R2/R3. It gives the wrong sign on R1. By Theorem 1
of Investigation VII and Proposition A, any such per-group rule must give R1 and R3 the same sign. This is **H7 (a
competing shortcut)**, implemented in native attention: a mechanism-level localization, not only an output-level one.

**H4 (incorrect downstream integration)** describes the same thing from the other side. The downstream block does
encode relational information after its computation: probes at its output decode e_g at 0.99 and m_g at 0.92 (xP80).
It does not route it into the candidate evidence. The decodable-but-unused pattern of Investigation VIII appears here
inside the native model.

## 4. Architecture versus optimization (H5, H6)

- **Keeping the third pretrained layer (keep3) makes R1 worse** (−1.49 vs −0.55), even though its input contains exact
  counts. Cleaner counts make the shortcut *more* available, which matches the retrospective §1 (complete lineages have
  larger R1 errors).
- A q-restricted MLP over the same native states does not take the shortcut (E ≈ 0). The difference between it and C1 is
  architectural: C1 must *route* through softmax attention over 43 tokens, and per-group routing weights are the natural
  first solution. This favours **H6 (an architectural bias of attention routing toward separable weighting)**, together
  with **H5**: the routing solution is a stable optimum within 4k steps.
- The remaining tests of H5 are C1 at 16k steps, C1 with Bayes targets, full fine-tuning and LoRA. They are in
  `NATIVE_RELATIONAL_LEARNING.md`. Any that did not run are listed there.

## 5. Hypothesis table

| hypothesis | verdict | decisive evidence |
|---|---|---|
| H1 information insufficiency | **true for incomplete parents (F-I); false for complete parents** | completeness covariate; decode-compose KL 0.000 on xP81 final states |
| H2 representation inaccessible | false at the final states (linear access); partial at resid_pre2 (counts 0.85) | decode-compose; native readers |
| H3 missing relational operation | **true for the C1 block**: it computes a per-group gate, not the cross-group conjunction | routing curve; attention freeze |
| H4 incorrect downstream integration | true: relational variables decodable after the block (e 0.99, m 0.92) but not used | probes at resid_post2 |
| H5 optimization | contributes: shortcut stable at 4k; longer training see `NATIVE_RELATIONAL_LEARNING.md` | — |
| H6 architectural constraint | supported as a *bias*, not an impossibility: an MLP over the same states succeeds | native readers |
| H7 competing shortcut | **true and localized**: per-group de-attention of perfect groups | routing; freeze |
