# Readout, final-LN and layer-2 factorial (development)

## 1. Which conditions are distinct in the actual code

Source: `cohort6_train.py` and `scripts/downstream.py`.

**What is inherited and what is fresh.**
- **Transfer re-initialises** layer 2 and W_R. W_R is unused by the T4 loss.
- **It inherits** W_A (the answer readout), g_f and b_f (the final LN), and W_C (unused).
- **It trains** layer 2, g_f, b_f, W_A, W_C and W_R. Embeddings and layers 0–1 receive zero gradient and zero update.

**What the inherited readout actually is.**
- In P-count parents, the inherited W_A **equals its random initialisation** (relative difference 1.4e-4). P-count pretraining
  trains a separate head, WX, from the final-LN readout states, so W_A received only weight decay.
- In P-count parents, the inherited **final LN is trained** by the count objective (g_f mean 1.13–1.29, SD 0.08–0.15).
- In T1 parents, W_A moved about 15 % from initialisation during T1 pretraining, and g_f ≈ 0.98.
- "Fresh W_A" for a P-count parent is therefore only another random draw. The readout variable that is meaningfully learned in
  P-count parents is the final LN.

**The mandate's five conditions reduce to four distinct ones.**

| condition | W_A | final LN | identical to |
|---|---|---|---|
| C1 | inherited | inherited | original |
| C2 | fresh | inherited | mandate conditions 2 and 4: identical in this code, because a fresh layer 2 and fresh readout leave the LN inherited unless reset |
| C3 | inherited | fresh (g = 1, b = 0) | |
| C5 | fresh | fresh | fully fresh downstream stack after the frozen layers |

Every condition draws its fresh layer 2 and W_A from the same `nets.init(seed + 7777)` as Investigation VI. Every condition uses
the same data stream and the same 4000 steps.

## 2. Results (R1 excess E, raw Δm_R1, quality)

**P80:**

| condition | E_R1 | Δm_R1 | E_R2 | E_R3 | E_F34-R1 | KL | acc |
|---|---|---|---|---|---|---|---|
| C1 (Investigation VI checkpoint) | −0.895 | −0.662 | +0.932 | +0.201 | −0.116 | 0.104 | 0.727 |
| C2 fresh W_A | −0.612 | −0.379 | +0.976 | +0.579 | −0.232 | 0.126 | 0.712 |
| C3 fresh LN | −0.891 | −0.658 | +0.873 | +0.148 | −0.072 | 0.104 | 0.728 |
| C5 fresh W_A and LN | −0.600 | −0.367 | +0.984 | +0.579 | −0.228 | 0.125 | 0.711 |

**T70:**

| condition | E_R1 | Δm_R1 | E_R2 | E_R3 | E_F34-R1 | KL | acc |
|---|---|---|---|---|---|---|---|
| C1 (Investigation VI checkpoint) | −0.098 | +0.135 | +0.241 | +0.260 | −0.039 | 0.008 | 0.745 |
| C2 fresh W_A | −0.021 | +0.213 | +0.159 | +0.100 | −0.021 | 0.006 | 0.747 |
| C5 fresh W_A and LN | −0.019 | +0.214 | +0.161 | +0.099 | −0.021 | 0.006 | 0.747 |

| C3 fresh LN | −0.093 | +0.140 | +0.237 | +0.265 | −0.036 | 0.008 | 0.745 |

Weakness of C3 (review R2): a trainable W_A can absorb the LN gain, so "fresh LN" is a weak test.

## 3. Interpretation

1. **The final LN does not matter.** C3 ≈ C1 for P80 (−0.891 vs −0.895), so the P-count-trained LN does not cause the error.
2. **Re-drawing W_A modulates the error but does not remove it.** In P80 it reduces the R1 error by 0.29. It also makes the
   model *less* accurate overall (KL 0.104 → 0.126, accuracy −1.5 points) and worsens R3 and F34. A new readout therefore
   changes the downstream solution as a whole. It does not specifically repair R1.

   Because P-count's inherited W_A is just its random initialisation, this is an initial-state sensitivity of the downstream
   optimisation. It is not an effect of a learned readout.
3. **The curriculum difference survives a fully fresh downstream stack.** Under C5 (everything after the frozen layers fresh),
   P80 has E_R1 −0.60 and T70 has −0.02, a difference of 0.58. The original C1 difference is 0.80.

   **The inherited readout and final normalisation are not the cause of the curriculum difference.** At most they modulate it,
   by about a quarter (n = 1 parent per arm; development).
4. **The fresh layer-2 initialisation is the same draw in every condition.** Initial-state sensitivity of layer 2 itself was
   therefore not separated here. The confirmation cohort shows the R1 reversal in 8/8 new lineages, each with its own layer-2
   draw.

## 4. Additional development controls

**Unfreezing control** (mandate §6.3): **not run**, for compute reasons. The claim stays restricted to frozen-feature transfer.
See OPEN_SCIENTIFIC_PROBLEMS.md.

## 5. Extra training (3× budget, development, P80)

C1 run for 12,000 steps with the same schedule shape:

| step | E_R1 | Δm_R1 | KL |
|---|---|---|---|
| 4000 | −1.03 | | |
| 8000 | −0.70 | | |
| 12,000 | −0.68 | −0.45 | 0.067 |

The original 4000-step model has −0.90. Tripling the budget improves average quality but leaves the R1 sign wrong (n = 1;
development). Generic extra training does not account for the rescue in this parent. Mandate outcome F remains **undetermined
for larger budgets and other parents**.
