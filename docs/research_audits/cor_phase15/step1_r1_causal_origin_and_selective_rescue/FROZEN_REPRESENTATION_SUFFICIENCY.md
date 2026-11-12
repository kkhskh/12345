# Frozen-representation sufficiency and accessibility

**Status:** development, on the Investigation VI parents. The only part frozen and tested on new lineages is the probe
prediction P6 (INDEPENDENT_CONFIRMATION.md).

**Data and code.** Readers train on fresh generator samples. They are evaluated on `logs/eval_dev.npz` (stream 17_100_000).
Code: `scripts/downstream.py`, `scripts/oracle_reader.py`, `scripts/accessibility_probes.py`.

**Supervision.** "soft" readers use exact-Bayes soft targets during training only; the test input is the episode alone. These
are **sufficiency** tests. Success shows the frozen state can *support* Bayes-like predictions under that supervision and
capacity. It does not show the original model computes them.

## 1. Controlled readers on the frozen layer-2 input Z (43 × 64)

All readers train for 4000 steps with batch 256, AdamW and the same schedule. Parameter counts are approximate.

| reader | architecture | params | P80 E_R1 | P80 Δm_R1 | P80 KL | T70 E_R1 | T70 KL |
|---|---|---|---|---|---|---|---|
| original transfer (labels) | 1 block + inherited LN/W_A | 50 k | −0.895 | −0.662 | 0.104 | −0.098 | 0.008 |
| A soft | 1 fresh block + fresh LN/W_A | 50 k | −0.741 | −0.508 | 0.091 | (queued, not run) | |
| B soft | 2 fresh blocks | 100 k | −0.358 | −0.125 | 0.055 | — | |
| C soft | MLP on the full flattened state | 1.7 M | −0.288 | −0.055 | 0.056 | **+0.415** | 0.004 |
| E-src soft | MLP on the 36 source tokens only | 1.4 M | −0.297 | −0.064 | 0.055 | **+0.449** | 0.004 |
| E-read soft | MLP on the readout token only | 0.08 M | (not run) | | | | |
| _rtemb (labels) | 1 block, readout-token input replaced by its embedding-level state | 50 k | −0.598 | −0.365 | 0.108 | −0.147 (Δm +0.086) | 0.004 |
| P81: original / _rtemb | | | −0.716 / −0.748 | | 0.028 / 0.045 | | |
| **P81: C soft** | MLP on the full state | 1.7 M | **+0.453** (Δm +0.686) | | 0.010 | | |

**D (oracle-feature MLP, positive control; not an equally informed competitor)**, 45 or 36 inputs, Bayes targets:

| features | steps | E_R1 | KL |
|---|---|---|---|
| counts + m + q-reports | 4000 | +0.014 | 0.000 |
| counts + q-reports, **without m** | 4000 | **−0.277** | **0.002** |
| without m | 16,000 | +0.008 | 0.000 |
| without m | 48,000 | +0.000 | 0.000 |

## 2. What the readers establish

1. **No information-theoretic insufficiency is shown, and none is claimed.** High-capacity readers on P-count states approach
   or reach the Bayes sign:
   - for P80, C and E-src give Δm_R1 ≈ −0.06, against −0.66 for the original transfer;
   - for P81, the C reader over-credits (E +0.45), as T70's does.
2. **Architecture-limited accessibility.** With the original one-block downstream architecture, the P-count R1 error persists
   under every training signal tried:
   - labels (−0.90);
   - exact-Bayes targets (−0.74);
   - a fresh readout (C2/C5 ≈ −0.60);
   - removal of the readout-token route (−0.60 / −0.75).

   More capacity (B: −0.36; C: −0.29) reduces it.
3. **Under identical readers the effect is parent-dependent, not uniformly curriculum-dependent.** The same MLP reader, trained
   identically, gives:

   | parent | R1 E |
   |---|---|
   | T1 parent 70 | +0.42 |
   | P-count parent 81 | **+0.45** |
   | P-count parent 80 | **−0.29** |

   The pattern follows the accessibility probes (§3). P81's states expose H2 better (linear 0.952, MLP 0.984) than P80's (0.912,
   0.89; majority 0.912). The original one-block learner fails on both P-count parents. A high-capacity reader fails only where
   the relation is least accessible.

   The earlier draft's statement, "curriculum matters even under identical readers", was based on P80 alone. It is **corrected**:
   with 2 P-count parents and 1 T1 parent this is not a curriculum-level result.
4. **The rare relational component is the last thing learned.** The oracle readers show this even with perfect, explicit
   information. Without m, an MLP matches Bayes on average (KL 0.002) long before it gets R1 right (≈ 16,000 steps). The R1
   error is the default state of a learner that has not yet built the cross-group interaction.

**Distinctions** (mandate §5.2):

| possible explanation | verdict |
|---|---|
| information-theoretic insufficiency | not shown |
| architecture-limited accessibility | **shown** for the one-block downstream; for a high-capacity MLP only in the least-accessible parent (P80) |
| optimisation failure | partly: the one-block learner moves *away* from Bayes in R1 over training (LEARNING_DYNAMICS) |
| limited training signal | not the cause: Bayes targets do not fix it |
| data insufficiency | not the cause: fresh samples every step; the oracle reader with the same data learns it with m |

## 3. Accessibility probes (held-out, generator-derived targets)

Source: `logs/accessibility_probes_xP80.json`. Natural decisions; fit stream 17_300_000, test stream 17_400_000.

**H2 (two or more perfect groups agreeing at q), linear probe.** The majority-class rate is 0.912.

| site | T1 (70 / 71 / 73) | P-count (80 / 81 / 83) |
|---|---|---|
| q-source tokens | **0.997 / 0.993 / 0.981** | 0.912 / 0.952 / 0.928 |
| readout token | 0.919 / 0.917 / 0.912 | 0.911 / 0.926 / 0.917 |

**Other targets:**

| target | site | probe | T1 | P-count |
|---|---|---|---|---|
| agree2 (two perfect groups vote the same) | readout token | linear | 0.73–0.84 | 0.66–0.68 (majority 0.675) |
| agree2 | q-sources | MLP | 0.996–0.999 | 0.94–0.99 |
| W (normalised Bayes mixture weights), R² | q-sources | linear | 0.70–0.79 | 0.19–0.59 |
| W, R² | q-sources | MLP | 0.99 | 0.51–0.92 |
| exact posterior, R² | q-sources | MLP | 0.97–0.98 | 0.66–0.91 |

**Reading.**
- In T1 frozen states, "another group is perfect" is linearly available at the very tokens layer-2 keys and values are read
  from. This matches the T1 mechanism, which runs through the keys of other tokens (FULL_LAYER2_CAUSAL_RECONSTRUCTION.md §2).
- In P-count states it is at chance linearly (parent 80) or weakly above chance. It is partly recoverable by an MLP probe.
- The probes are consistent with the accessibility account. **They are not evidence of use** (Investigation VI, T2). The causal
  evidence is the injection rescue and its controls.
- **They are not evidence of absence either (review R3-6).**
  - A class-balanced re-probe on a confirmation parent gives AUC 0.73 for H2.
  - The per-group "perfect" indicator reaches AUC 0.93–0.997.
  - agree2 is MLP-decodable at 0.94.

  The ingredients are present; their *conjunction* is weakly accessible. The post-review injna result (SELECTIVE_RESCUE.md §5)
  shows that even supplying the ingredients explicitly does not help the learner.

## 4. Representation collisions (§5.3)

No search for exact collisions was run.
- High-capacity readers recover most of the R1 response from P-count states, so a collision-based lower bound would be at best
  small.
- The approximate-collision argument would need a metric that governs the downstream learner's inductive bias, and no such
  metric was established.
- Status: **not established**. The empirical evidence above stands in its place.

## 5. Count availability versus conditional use (§5.4)

| quantity | site in P-count states | accessibility |
|---|---|---|
| per-group counts | readout token (by construction of P-count pretraining: the count head reads the readout tokens) | linear |
| eligibility m (≈ H2 per group) | readout token and q-sources | not linear; partly nonlinear |
| agree2, the joint "which perfect groups agree" relation | readout token | at chance |
| agree2 | q-source tokens | MLP-recoverable |

P-count states therefore encode each count *separately* in an accessible form, but not their *joint* relation. Under
Theorem 1, the separate form is exactly the one that leads a learner to the wrong R1 sign. The n-injection result confirms this
causally on new lineages: making the separate counts *more* accessible makes R1 worse.
