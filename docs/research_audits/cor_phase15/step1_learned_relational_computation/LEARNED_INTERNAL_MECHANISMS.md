# Learned internal mechanisms (Problem D)

Scope: the mechanism of the *successful* learned composition readers (attention, Condition R), and the contrast with failed
learners. Interventions are on the reader. Each is defined using the generator-side eligibility (perfect, agreeing at q)
as an analysis tool, which the model itself never receives. Code: `scripts/mech_reader.py`; output
`logs/mech_reader.json` (development) and `logs/mech_reader_conf.json` (confirmation, P3).

## 1. Attention-reader interventions (development, attn R label 16k seed 0; attn2 in brackets)

Values are the model's centred-logit response dm_M to the edit. Bayes gives R1 +0.23, R2 −2.38, R3 −2.03, F34 R1 +0.41,
and F54o (removing a perfect partner) ≈ −0.24.

| mode | F45 R1 | F45 R2 | F45 R3 | F34 R1 | F54o |
|---|---|---|---|---|---|
| base | +0.15 [+0.23] | −2.41 [−2.39] | −2.16 [−2.10] | +0.42 [+0.45] | −0.17 [−0.25] |
| self-only attention | +0.17 [−0.13] | +0.16 [−0.07] | +0.17 [−0.11] | −0.17 [+0.00] | −0.16 [+0.11] |
| no attention to eligible partners ("noperf") | **−3.56** [−2.44] | −1.03 [−0.63] | −4.11 [−3.77] | **+0.45** [+0.44] | **+3.46** [+2.35] |
| attention only to eligible partners ("noother") | +0.63 [+0.02] | +1.32 [+0.97] | +1.97 [+1.28] | +0.24 [+0.26] | −0.62 [−0.02] |

**Readings.**

1. **Self-only attention makes the response regime-independent.** R1, R2 and R3 move by the same amount (spread ≤ 0.06).
   This is exactly what Proposition C requires of any rule that is additive over groups. The regime-dependence of the
   trained reader is therefore carried entirely by cross-group attention.

2. **Blocking attention to eligible partners selectively recreates the P-count failure.**
   - The R1 response turns strongly negative (−3.56), the same sign as the separable discount of P-count models and of the
     early training phase.
   - Removing a perfect partner (F54o) now *raises* the target's margin (+3.46 where Bayes gives −0.24).
   - The 3→4 response (F34), which does not involve perfectness, is unchanged (+0.45 vs +0.42).

   The cross-group credit therefore flows through attention to eligible partners. Without it, the reader falls back to the
   count-based competitive rule.

3. **Attending only to eligible partners is not enough**: R2/R3 flip sign. Attention to non-eligible groups also matters,
   which is consistent with Bayes using every group's count as a reliability weight (§3 of the math document).

## 2. Representational evidence

| probe (held-out logistic, per group) | attn R | attn2 R | majority |
|---|---|---|---|
| m_g after the first block's attention | 0.967 | 0.957 | 0.667 |
| m_g after the last block | 0.978 | 0.988 | 0.667 |
| perfectness after the last block | 0.991 | 0.999 | 0.640 |

**Interchange on R1 pairs.** In the group states after block-1 attention, the m-probe subspace is replaced by its value
from the edited counterpart. The "share" is the response this produces, divided by the full response.

| subspace | attn R share | attn2 R share |
|---|---|---|
| m-probe subspace | **1.84** | **0.51** |
| random subspace of equal dimension | −0.23 | −0.40 |

The m-subspace carries the effect and a random one does not. Share > 1 in the one-block reader means the subspace
overshoots: the rest of the state partly cancels it. This is reported as is; it does not mean "the subspace is the whole
mechanism".

## 3. Decodable is not the same as used (development; prediction P7 tests it on new readers)

A linear probe for m on the pre-readout representation (`scripts/predictor_probe.py`, `FROZEN_PROTOCOL_ADDENDUM_A1.md`):

- **Failed** concatenation MLPs at 16k decode m at **0.986–0.988**.
- **Successful** attention readers decode it at 0.77–0.88.

Failure is therefore not a failure to represent m linearly. It is a failure to use it in the multiplicative form
RHO^{n_g}·m_g that the posterior requires. This agrees with the separable phase seen in every R learner
(`LEARNABLE_COMPOSITION_EXPERIMENTS.md`, finding 2). It repeats in a learned setting Investigation VII's lesson about
frozen P-count states, which were also decodable but not usable by a one-block reader.

## 4. Mechanism of the learned repair (Condition Z, radapt)

The repair adds the output of a group-attention reader over raw ingredients at the readout token. Its parameters have the
same form as the composition attn reader. Its internal mechanism was not dissected separately: it is a copy of the reader
architecture of §1, trained jointly with a fresh downstream block. What it contributes to the frozen P-count model is
measured behaviourally in `SELECTIVE_LEARNED_REPAIR.md`. **Open:** whether the downstream block uses the adapter as a
correction (an "m-like" term) or as a parallel posterior estimate. Ablating the adapter after training would distinguish
the two. This was not run.

## 5. Confirmation (P3)

See `FROZEN_PROTOCOL_AND_CONFIRMATION.md` Part II.
