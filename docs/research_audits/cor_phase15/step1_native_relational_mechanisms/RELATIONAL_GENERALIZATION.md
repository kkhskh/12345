# Relational generalization (Workstream D)

**Setting.** These experiments use the role-model generator (`src/genlib.py`) with exact brute-force Bayes for every
configuration (`MATHEMATICAL_STRUCTURE_AND_LIMITS.md` §0). The readers receive *primitive per-group evidence*: one-hot
leave-question count, one-hot target-question report pair, and optionally a group-id one-hot. This is
Investigation VIII's Condition R, transposed to the new generator, and it is **not** native-state evidence. Readers are
trained on generator labels for 12,000 steps (`scripts/gen_readers.py`).

**What each experiment tests.**
- **Held-out eligible pair (HOLD = 1).** Training rejects every episode in which groups 0 and 1 are *both* all-agreeing,
  so the eligible pair {0,1} is never seen. The episodes are on support, but the distribution is shifted.
- **Group permutation.** Swapping slots 0 ↔ 1 is a symmetry of the generator.
- **Size shift.** G = 4, 5, 6 at test, for readers that accept any G.
- **Semantic shift.** β = 0.7 (partial copying). The inputs are on support of the β = 1 generator, but the Bayes rule
  changes from a hard gate to a soft gate (Theorem 3).

Metric: KL to exact Bayes, overall and on the relational regime (S ≥ 2 eligible groups), split into the held-out pair
(KL_pair01) and other pairs (KL_S2_not01). 30,000 decisions in distribution; 6,000 per shifted setting.

## Frozen confirmation (seeds 10–12, evaluation streams never used in development)

Source: `logs/confirmation_results.json`.

| prediction | result | values (per seed) |
|---|---|---|
| **GP1** identity-free readers generalize to the unseen eligible pair (KL_pair01 ≤ 0.005) | **confirmed, 6/6** | ds_sum 0.0027 / 0.0025 / 0.0024; attn 0.0009 / 0.0010 / 0.0010 |
| **GP2** the concatenation MLP does not (KL_pair01 ≥ 5 × KL_S2_not01) | **confirmed, 3/3** | 0.401 vs 0.0052; 0.368 vs 0.0051; 0.388 vs 0.0058 (≈ 70×) |
| **GP3** no size extrapolation (KL at G = 4 ≥ 0.01) despite exact representability | **confirmed, 6/6** | ds_sum 0.053 / 0.050 / 0.050; attn 0.033 / 0.035 / 0.025 |
| **GP4** semantic shift to partial copying breaks every reader (KL at β = 0.7 ≥ 0.10) | **confirmed, 9/9** | ds_sum 0.31–0.37; attn 0.29–0.31; mlp 0.20–0.25 |

## Development grid (seed 0, 20 runs)

KL values; "perm" is the mean |Δ centred logit| under the 0 ↔ 1 slot swap.

| reader | id | hold | in-dist | S ≥ 2 (not 01) | pair01 | perm | G = 4 | G = 6 | β = 0.7 |
|---|---|---|---|---|---|---|---|---|---|
| ds_sum | 0 | 0/1 | 0.0004 | 0.0007 / 0.0010 | 0.0011 / 0.0014 | 0 | 0.044 / 0.048 | 0.21 | 0.26–0.27 |
| ds_sum | 1 | 0/1 | 0.0005–0.0006 | 0.0008 / 0.0014 | 0.0011 / **0.0038** | 0.027 | 0.060 / 0.066 | 0.22–0.24 | 0.28–0.30 |
| ds_mean | 0 | 0/1 | 0.0008 | 0.0009 / 0.0016 | 0.0012 / 0.0019 | 0 | 0.042 / 0.043 | 0.13 | 0.48 |
| attn | 0 | 0/1 | 0.0003 | 0.0008 / 0.0010 | 0.0009 / 0.0011 | 0 | **0.025** / 0.026 | **0.083** | 0.29 |
| attn | 1 | 0/1 | 0.0005 | 0.0019 / 0.0013 | 0.0011 / **0.0030** | 0.016 | 0.10 / 0.15 | 0.13–0.17 | 0.30–0.32 |
| relnet | 0 | 0/1 | 0.0006 | 0.0009 / 0.0015 | 0.0013 / 0.0017 | 0 | 0.11 / 0.12 | 0.88–0.90 | 0.33 |
| relnet | 1 | 0/1 | 0.0007–0.0008 | 0.0012 / 0.0015 | 0.0015 / 0.0040 | 0.025 | 0.13 / — | 1.06 | 0.35–0.38 |
| mlp | 0 | 0 | 0.0023 | 0.0065 | 0.0074 | 0.073 | n/a | n/a | 0.135 |
| mlp | 0 | 1 | 0.0111 | 0.0058 | **0.310** | 0.073 | n/a | n/a | 0.21 |

## Findings

1. **Identity-free architectures give exact permutation invariance and held-out-pair generalization; this is an
   inductive bias, not a learned principle.**
   - With group ids in the input, the held-out pair costs 2–4× (0.0011 → 0.0030–0.0040) and permutation invariance is
     only approximate (0.016–0.027).
   - Without ids it is exact by construction.
   - The concatenation MLP memorises configurations: its held-out-pair KL is about 70× its seen-pair KL.

   This is the D2 finding: architectural structure, not training, gives the invariance. The mandate warns against
   mistaking such an inductive bias for a learned universal principle.
2. **Representability does not imply extrapolation.** By Theorem 1, one fixed (φ, ρ) with sum pooling is exact for every
   G. No learned reader found it:
   - every identity-free reader degrades at G = 4 and badly at G = 6;
   - attention with mean pooling degrades least (0.025 at G = 4);
   - the pairwise relnet, whose sum over pairs grows quadratically, degrades most.

   The learned ρ is accurate only on the range of pooled values seen in training (S ≤ 2 with three groups). This is a
   concrete counterexample to inferring size generalization from an exact sum-decomposition theorem.
3. **Relational semantics shift (partial copying) breaks every reader, as Theorem 3 predicts.** At β = 0.7, 68 % of
   episodes contain *no* all-agreeing group. The hard gate the readers learned then routes no credit to any colluder
   hypothesis, while the Bayes rule uses soft per-group likelihood ratios. On the decisions where S ≥ 2 (hard and soft
   gates nearly agree), the readers stay accurate: ds_sum KL_S2 = 0.0006 at β = 0.7 in development. This localizes the
   failure exactly to the change in semantics.
4. **Within support vs off support.**
   - The held-out pair is within support and shifted in distribution: identity-free readers pass.
   - β = 0.7 is within the *input* support but has changed semantics: everything fails.
   - G = 4–6 is off support for the trained input arity: everything degrades.

   No reader was evaluated on impossible inputs.

## Not done

- Training on mixed G (for example {3, 4}) to test whether size extrapolation can be learned compositionally.
- Mixed-β training.
- Multiple colluders (C = 2), where Theorem 2 predicts interaction order 3. The generator supports it; it was not run
  for compute.
- Context-length or Q changes.
- Generalization of *native* P-count mechanisms. The native models are tied to G = 3 and Q = 6 by their architecture
  and pretraining.
