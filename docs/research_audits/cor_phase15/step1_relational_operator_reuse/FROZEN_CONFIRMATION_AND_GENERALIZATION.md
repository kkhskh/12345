# Frozen confirmation and generalisation

> **Post-review corrections** (`reviews/ADJUDICATION.md`; agent reviews, not human peer review). All frozen calls stand as scored. **W5:** P1 is architectural and P3 equals P2 measured on content-swapped episodes, so the 6/6 calls are 4 independent mechanism tests plus P0. **W10:** every selected checkpoint lies 20–220 steps before natural acquisition of the unacquired slots. **W11:** U→U controls were recorded once per donor (duplicated). Random-code controls added by Reviewer B give 0.00–0.07, and transfer survives them. **W12:** P0 at arity 2 is unresolved at 20-step resolution. **W7, W8, W9:** see MAIN. Text withdrawn after review is kept in place and marked *[withdrawn, Wn]* so the record is preserved.


**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

## 1. Chronology (logs/COMPUTE_LOG.txt, git)

| time (UTC, 2026-11-18) | event |
|---|---|
| 06:21 | XVII start; branch `cor-phase15-step1-relational-operator-reuse` from `9c56c59` |
| 06:26 | **Stage 1 committed (`431b1b6`)**: O/D determination (Case D, silent), before any neural mechanism result |
| 06:26–07:24 | development on existing XVI models (replays, per-slot reconstruction, relabel, routing, interpolation, slot training); scorer preflight; analysis smoke test |
| **07:24:42** | **freeze `c419773`** pushed (27 hashed files) |
| 07:24:54 | blind confirmation queue launched (48 jobs, 2 lanes) |
| 07:24–12:03 | 12 training runs, all with exit code 0 |
| 12:01–12:39 | 12 frozen analyses, all with exit code 0 |
| 12:39–13:03 | learning arms (5 models × 2 arms), all with exit code 0 |
| about 13:04 | **first read of any confirmation output: `score_XVII.py`** |

**Monitoring.** Following the integrity rule adopted after XVI's review, progress was monitored only through the lane log and the existence of `done.json`, analysis and learning directories. No output file was opened or line-counted before scoring.

**No restarts** occurred during XVII's confirmation.

## 2. Cohorts and independence

| cohort | fresh independent models | evaluable (partial checkpoint at 20-step resolution) | learning subset |
|---|---|---|---|
| arity 3 (3:0.9:0.9:1/3, constant rate, 5000 steps) | 4 (seeds 801–804) | 3: s802, s803, s804 (s801 never acquired the gate: right-censored at 5000) | s802, s803, s804 |
| arity 2 (2:0.9:0.9:0.05, cosine, 4000 steps) | 8 (seeds 701–708) | 3: s704, s705, s707 | s704, s705 |

- Every model has an independent initialisation and data stream.
- Pairs (2–9 per model), slots and episodes are within-model repeated measures. **The replication unit is the model.**

## 3. Verdicts (logs/score_XVII.json; pooled is primary)

| | pooled | arity 3 | arity 2 | expectation |
|---|---|---|---|---|
| **P0** partial checkpoint resolvable | **MIXED** (6/11; 95% CI 0.23–0.83) | STAGGERED 3/3 | MIXED 3/8 | STAGGERED: **not met** |
| **P1** full code exchange moves gate use | **FOLLOWS_CODE** 6/6 | 3/3 | 3/3 | met |
| **P2** early-code [0,2) transfer beyond control | **TRANSFER** 6/6 (CI 0.54–1.00) | 3/3 | 3/3 | met |
| **P3** late layer less code-dependent | **LATE_LESS_CODE_DEPENDENT** 6/6 | 3/3 | 3/3 | met |
| **P4** routing alone insufficient | **ROUTING_INSUFFICIENT** 6/6 | 3/3 | 3/3 | met |
| **P5** code keying persists after acquisition | **CODE_KEYING_PERSISTS** 6/6 | 3/3 | 3/3 | met |
| **PL** targeted training: interference, no acceleration | **INTERFERENCE_NO_ACCELERATION** 5/5 | 3/3 | 2/2 | met (see §5: "no acceleration" passed only at the ceiling) |

## 4. Model-level numbers (`logs/xvii_battery_summary.txt`)

| model | A / U at selected checkpoint | [0,2) recipient (control) | [0,3) recipient / donor | donor under [2,3) against [0,2) | routing (layers 0–1) | late: [0,2) against mixed [0,1) |
|---|---|---|---|---|---|---|
| A3_s802 | {1} / {0, 2, 3, 4, 5} | 0.65 (0.05) | 0.95 / 0.05 | 0.60 against 0.06 | 0.07 | 0.95 against 0.43 |
| A3_s803 | {0, 2, 3, 4, 5} / {1} | 0.90 (baseline 0.01) | 0.91 / 0.02 | 0.90 against 0.06 | 0.05 | 1.00 against 0.54 |
| A3_s804 | {1, 4} / {0, 2, 3, 5} | 0.78 (0.06) | 0.87 / 0.04 | 0.79 against 0.04 | 0.07 | 0.95 against 0.62 |
| A2_s704 | {2, 3, 4} / {0} | 0.68 (baseline 0.12) | 0.90 / 0.12 | 0.70 against 0.17 | 0.06 | 0.81 against 0.27 |
| A2_s705 | {4, 5} / {0, 1, 2, 3} | 0.83 (0.04) | 0.95 / 0.03 | 0.82 against 0.02 | 0.10 | 0.97 against 0.52 |
| A2_s707 | {1, 3, 5} / {0, 2, 4} | 0.79 (0.04) | 0.99 / 0.06 | 0.79 against 0.08 | 0.17 | 0.79 against 0.26 |

## 5. Readings

**P0 (staggering).**
- At arity 3, all 3 acquiring models had partial checkpoints. Slot acquisition spread over 80–160 steps: s802 3700–3760; s803 1360–1440; s804 2580–2680.
- At arity 2, 5 of 8 models acquired the gate in every slot within one or two 20-step intervals (spreads of 20–40 steps). Staggering there is below this resolution, or absent.
- **The frozen expectation (STAGGERED pooled) is not met.** Slot staggering is reliable at arity 3 and inconsistent at arity 2 in this condition.
- **None of the fresh models shows a long partial window like C_s202.** The longest is A2_s705, with 10 partial checkpoints, about 200 steps. C_s202's 1600-step window remains a rare case (1 of 16 XVI acquisitions and 0 of 11 fresh acquiring models).

**P2 and P4 (transfer, and routing as insufficient).**
- In all 6 evaluable models, an unacquired slot ran the gate computation when given an acquired slot's code in layers 0–1 only: recipient F_G 0.65–0.90, against U → U controls of 0.01–0.12 (perturbation controls 0.00–0.07 after review, W11). Nothing but the slot code moved.
- Transplanting attention routing gave 0.05–0.17.
- This replicates both development findings in fresh models at both arities [frozen].

**P3 (layer locus).**
- Acquired slots keep 0.60–0.90 of their gate use when only layer 2 carries another slot's code, and keep 0.02–0.17 when layers 0–1 do. Layers 0–1 carry the *raw*-code dependence. *[layer-locus reading withdrawn, W4]*

**P5 (after acquisition).**
- Consistent exchange across layers 0–1 keeps 0.79–1.00. Mixing codes between layers 0 and 1–2 drops to 0.26–0.62.
- The slots never become layer-wise interchangeable. Each runs the shared computation keyed consistently by its own code [frozen].

**PL (learning).**
- **Interference** [frozen]: targeted training (loss on one unacquired slot) took the acquired slots from 0.86–0.96 down to 0.11–0.29 within 400 steps. Full training kept them at 1.00.
- **"No acceleration" passed only at the ceiling.** The targeted slot acquired the gate in both arms within 100–200 steps (1.00 against 1.00), so that half of PL is uninformative here. This differs from development: in C_s202 the targeted slot never acquired the gate in 1000 steps.
- **Exploratory, beyond the frozen rule:**
  - Under targeted training the **other unacquired slots did not acquire the gate** (0.02–0.19 at 400 steps). Under full training they all did (1.00).
  - So learning at one slot does not spread to the others, and each slot's competence is sustained by its own loss.

## 6. What generalised and what did not (Q19, Q20)

- **Generalised across fresh models and both arities** [frozen]:
  - code-keyed early computation (P1–P3, P5);
  - transfer of gate use by early-layer code exchange (P2);
  - routing not being sufficient (P4);
  - interference under slot-targeted training (PL).
- **Did not generalise** [frozen]:
  - the long partial window (rare);
  - reliable staggering at arity 2 (3/8).
- **Arity 2 against arity 3** [frozen]: the same mechanism pattern in every evaluable model. Arity 3 staggers more reliably.

## 7. Deviations and limits

1. **A3_s801 never acquired the gate within 5000 constant-rate steps.** It is excluded as non-acquiring (the frozen rule) and reported as censored.
2. **Only 6 of 11 acquiring models were evaluable.** The mechanism verdicts rest on 6 models (3 per arity). An exact 95% interval for 6/6 is 0.54–1.00.
3. **Partial windows in fresh models are short (20–200 steps).** Most selected checkpoints lie within 100 steps of the remaining slots acquiring anyway. The U → U control (0.01–0.06) and the per-slot baseline (0.01–0.12 where |U| = 1) show that the transfer is not mere imminence.
4. **PL's "no acceleration" half is ceiling-limited** (§5).
5. **Architecture control (Experiment E): not run** (the frozen budget decision).
