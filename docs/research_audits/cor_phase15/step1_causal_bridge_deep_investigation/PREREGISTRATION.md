# Preregistered experiments and predictions

This file was written and committed **before** any experiment code in `src/experiments.py` existed. The git history shows this file's commit before the experiment commit. Each prediction is derived from the analytic structure of the mechanism, not from any run. A prediction that fails is recorded as failed in `logs/experiments.txt` and in `THEOREMS_AND_COUNTEREXAMPLES.md`. It is not revised.

Notation: `m = L_c − L_{c'}` for a fixed candidate pair. `h(t) = φ(t) + φ(−t)`. Source coordinates `x, y` are the projections of the two sources' contributions onto the reader's input directions after normalization. Their signs equal the signs of the input evidence levels `a, b`.

## Mechanism A: agreement-sensitive MLP (positive control and family)

- **PA1.** Positive control (d = 4, 3 tokens, uniform attention, 4 GELU units). `sign(m) = +` iff `a = b`, for all four binary cells, with and without final LN.
- **PA2.** The margin of the residual *before* the MLP is 0 (up to rounding) in all four cells. All of `m` comes from the MLP write.
- **PA3.** Clamping the attention pattern to any fixed row-stochastic pattern that gives equal positive weight to both sources leaves `sign(m)` unchanged in every cell. Attention is not the source of the sign change.
- **PA4.** Flipping A with B fixed reverses `sign(m)`, and so does flipping B with A fixed. Flipping both leaves `m` unchanged.
- **PA5.** Analytic law: for every real `x, y` and any activation with `h` strictly increasing in `|t|` (GELU in both forms, ReLU, SiLU, softplus), the MLP's contribution to `m` has sign `sign(xy)`. For ReLU it equals `2λc·sgn(xy)·min(|x|,|y|)` exactly.
- **PA6.** Odd activations (tanh) give `h ≡ 0`, so the MLP contribution is identically 0. Sigmoid gives `h ≡ 1`, so it is also 0.
- **PA7.** Small-signal regime (GELU): `q ≈ 4√(2/π)·xy`. Large-signal regime: `q ≈ 2·sgn(xy)·min(|x|,|y|)`.
- **PA8 (matched-sum control).** At fixed `S = x + y`, `q` decreases strictly as `|x − y|` grows. For the triple `(1,1), (2,0), (3,−1)`, `q > 0`, `q = 0`, `q < 0`.
- **PA9 (sum-only negative control).** A reader using only `±S` neurons produces, after a constant offset, the same signs as the designed circuit on the four binary cells. On the matched-sum triple it is constant. **The binary positive control alone cannot distinguish relation computation from even nonlinearity of net evidence.**
- **PA10.** Each structural condition removed in turn produces a sign error somewhere on `[−1,1]²`: skewed read weights, unequal output weights, odd activation, a direct readout path, a final-LN bias with `Δu^T β ≠ 0`, an offset in the read directions.
- **PA11 (perturbations).** Under small i.i.d. weight perturbations of relative size `η`, sign errors appear first near the axes, where `|q|` is small. The binary cells stay correct for small `η`.
- **PA12 (continuous family with anchors).** `sign(m) = sign(ab)` holds over a grid of `a, b ∈ [−r, r] \ {0}` in the fully recomputed block.
- **PA13.** The circuit is not candidate-relative: `m(a,b) = m(−a,−b)`. Two sources agreeing on `c'` push toward `c`.

## Mechanism B: query–key agreement gate (two layers, dynamic attention)

- **PB1.** In the designed gate, B's vote enters `m` multiplied by an attention weight that is strictly increasing in `xy`. B's contribution counts more when A agrees with B.
- **PB2.** For agreeing cells, `sign(m)` equals the shared sign (unanimity).
- **PB3 (dissection).** With the layer-2 pattern clamped to its `a = 0` value (routing fixed), the dependence of B's contribution on `a` vanishes, leaving only A's direct vote. With the layer-2 value vectors clamped (values fixed) and routing free, the gate effect remains.
- **PB4.** Query-only and key-only substitutions each change B's weight. Their effects are not additive on the margin: the score is bilinear in the query and key features.
- **PB5 (preference reversal).** For some conflicting `(a, b)`, the fixed-routing baseline prefers B's candidate while the gated model prefers A's.
- **PB6 (negative control).** With `W_Q = 0` in layer 2, the margin is additive in the A and B contributions, and the RB check is relation-blind.
- **PB7 (masking).** Placing B after the read position removes every dependence on `b`.
- **PB8 (competition).** With two B-type sources of opposite sign, attention concentrates on the one that agrees with A, so the margin moves toward A's candidate more than the fixed-routing baseline predicts.
- **PB9 (full-margin failure).** Adding a downstream MLP that re-reads the gated output nonlinearly (or a final LN with bias) produces a cell in which the sign of the total margin change differs from the sign of the gated head's direct change.

## Mechanism C: two-block reconciliation (candidate-relative amplification)

- **PC1.** Block 1 computes the agreement signal `q`. Block 2 gates the net-evidence vote `S` by `q`. Across the grid, `sign(m − bias)` equals `sign(S)`, so agreement never flips the evidence-favored direction in this class.
- **PC2 (decision consequence).** With a prior bias toward `c'` and weak positive net evidence, the decision is `c'` under disagreement and `c` under agreement, at **fixed `S`**.
- **PC3 (mediation, interchange).** Patching the block-1 MLP output from a counterfactual input with the opposite agreement and the same `S` moves `m` to the counterfactual's value up to the change in the LN scale. Ablating the block-1 MLP removes the agreement dependence at fixed `S`.
- **PC4 (bottleneck negative control).** If block 1 writes both sources onto one direction, every matched-sum pair gives identical margins, as screening requires.
- **PC5 (irrelevant direction).** Adding a component along an unread residual direction changes `m` only through the block-2 LN scale. The sign is unchanged.
- **PC6 (alternative decomposition).** A single-block wide MLP fitted to the C margin function on a grid approximates it with small error and has no explicit `q` variable. The I/O function does not identify the two-block mechanism.

## Identifiability

- **PI1.** Head-output redistribution `W_O^1 + E`, `W_O^2 − E` with shared Q/K/V leaves every logit fixed and changes each per-head direct effect, and each per-head patching effect.
- **PI2.** The QK gauge `(W_Q M, W_K M^{-T})` and the VO gauge `(W_V N, N^{-1}W_O)` leave scores and head outputs fixed and change the Q, K, and V vectors.
- **PI3.** For ReLU, positive neuron rescaling changes per-neuron output weights and leaves the function and `q` fixed. GELU has no such symmetry.
- **PI4.** Orthogonal residual maps fixing `1` (with `γ = 1`) leave the function and the source coordinates `x, y` (defined through compositions) fixed.
- **PI5.** Splitting one neuron into two half-weight copies leaves the function fixed and changes per-neuron attributions.
