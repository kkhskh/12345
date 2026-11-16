# XIII-E freeze B: early-training causality (Workstream B)

Committed before any B confirmation run starts.

**Development basis (B1/B3; `scripts/b_calib.py` → `logs/b_calib.json`)**

Immediate loss disruption on the XIII-D base checkpoints at steps 0, 25, 50, 100 and 546.

| intervention | base | disruption |
|---|---|---|
| ro0 | early | 0.04–0.54 in the window; 1.29 at step 546 |
| l2q | mixture | 0.002–0.014 in the window; 0.46 at step 546 |
| noise | — | saturates at 0.6–0.8 |
| sham (masked code path, zero bias) | — | 0 |

- The sham was verified **bitwise identical** to plain training over 25 steps.
- Matching:
  - short-window noise σ is 2.323 on the early base (matched to ro0) and 0.25 on the mixture base (matched to l2q);
  - strong-noise σ is 2.323 on the mixture base;
  - late noise σ is 4.0 on the early base and 2.06 on the mixture base (matched to the late-l2q disruption, ≈ 0.46);
  - the late partial early-route block is ro0p with k = 2 (0.39).
- The full late ro0 disruption (1.29) cannot be matched by noise, so F1e is not re-tested.

**Design (`logs/B_design.json`, 41 runs)**

- Bases: uP160 21/21/21 (early) and 23/23/23 (mixture).
- Data seeds: D ∈ {anchor, 801, 802}.
- Short-window conditions, to step 1211 (34 runs: 32 conditions and baselines, plus the 2 D=801 sham checks):
  - real route mask over windows of 25, 50 and 100 steps;
  - matched noise for 100 steps;
  - strong noise for 100 steps (mixture base only);
  - a sham for D = 801;
  - no-mask baselines for D = 801 and 802. The anchor-D baselines are the XIII-B natural seeds.
- Severity-matched late conditions, 546–1146 and run to step 4000, branched at 546 (8 runs):
  - route-specific: ro0p on the early base, l2q on the mixture base;
  - matched noise;
  - each for 2 D values.
- Continuations to step 4000: the 100-step real mask and the baseline at D = 801, for both bases (4 runs).
- Queues: `queue_B.txt`, `queue_Blate.txt`, `queue_Bcont.txt` and `queue_Bmeasure.txt` (streams 95.2 M).
- Harness: `scripts/ft_factorial_b.py`, which is `ft_factorial.py` plus the sham, ro0p and noise interventions and branching from XIII-D bases.

**Predictions (`scripts/score_B.py`, which fails closed)**

- **R3 (early-mask specificity):** for both bases, the 100-step real mask flips the 1211 class in at least 2 of 3 D seeds; the matched noise flips it in at most 1 of 3; and the sham is bitwise identical to its baseline.
- **R4 (specifically early versus general disruption):** severity-matched late disturbances flip the class at 4000 in at most 2 of 8 runs; strong early noise on the mixture base flips it in at most 1 of 3; and the 25-step window flips no more than the 100-step window.
- Exact CONFIRMED / FALSIFIED / INCONCLUSIVE rules are in the scorer docstring.

**Preflight (`logs/preflight_B.json`):** P0 and N1–N7 all OK.
