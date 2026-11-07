# 07 — Multi-path coordination

## B1. Separate phenomena

Two contributions enter one downstream map. The words below are not synonyms.

| term | definition used here |
|---|---|
| redundant support | each contribution, alone, puts the margin on the same side of zero |
| independent support | the downstream map is a sum of functions of each contribution, so the cross-interaction is zero |
| compensatory support | after one contribution is removed, a descendant equation changes another contribution so that the margin is restored |
| competition | increasing one contribution decreases the weight or the effect of another inside a shared normalization or a shared capacity constraint |
| suppression | one contribution lowers the margin, against the other |
| gating | one contribution selects which function is applied to the other |
| nonadditive interaction | the four-term interaction of a named pair is nonzero |
| cancellation | a linear form on the sum is near zero while the summands are not |

Redundant support does not imply independent support. A gate can make either path sufficient in its context and still have a nonzero interaction. Cancellation does not imply suppression: both paths can be large and opposite in a readout that neither path "suppresses" as an operation. Competition in softmax is the partition constraint, derived in `04`.

## B2. Explicit structural examples

The downstream margin in the linear case is `m=w·(p_1+p_2)`.

- Either path alone is sufficient for the sign: `w·p_1=w·p_2=1`. Ablating either leaves `1`.
- Both are necessary: `w·p_1=w·p_2=1` is not this case. `w·p_1=1`, `w·p_2=-0.5`, margin `0.5`. Ablating `p_1` leaves `-0.5` and the sign reverses. Both signs of the pieces are required for the positive margin.
- One path suppresses another: the second case, if "suppress" means the inner product is negative. The mechanism is the inner product.
- One path compensates after ablation: not the linear sum. In the linear sum, ablating `p_1` does not change `p_2`. Compensation requires a structural equation in which `p_2` is recomputed from a parent that the ablation changed. Attention with a clamped pattern does not do this for a value ablation: `A_j v_j` for `j\neq i` stays put when `v_i` changes. Attention with recomputed scores can move weight onto the remaining tokens. That movement is softmax, not a repair circuit.
- Both agree and the answer is wrong: `w·p_1=w·p_2=-1`. The margin is `-2`. Agreement is not correctness.
- The paths disagree as vectors and the margin is stable: `p_1=(1,100)`, `p_2=(1,-100)`, `w=(1,0)`. The vectors are far apart. Each inner product is `1`. Single ablation preserves the sign.
- The interaction changes with context: `relu(u+v)` has interaction `0` at `(1,1)` and `-1` at `(1,-2)`. The same formula, two operating points.

Checked by `test_vector_agreement_is_not_readout_redundancy`, `test_single_ablation_robustness_needs_the_remaining_inner_product`, and `test_mlp_relu_is_not_automatically_an_interaction`.

## B3. Derivatives and the four-term term

For a map with a Taylor expansion at the operating point,

\[
R(\delta)=\nabla m\cdot\delta+\frac12\delta^\top H_m\delta+\text{remainder}.
\]

The cross term in the Hessian is the infinitesimal interaction. Error 1: a nonzero cross term at the baseline does not force every finite pair to have nonzero four-term interaction, and a zero cross term at the baseline does not force the finite interaction to vanish. `(u+v)^3` has a vanishing mixed partial at `0` and interaction `6` at `(1,1)`. `uv-u^2v-uv^2` has mixed partial `1` at `0` and interaction `0` at `(2,-1)`.

On `C^2` rectangles the interaction function is identically zero exactly when the map is separable. That is the clean boundary. It is a statement about the function, not about a residual of the unintervened state.

A nonzero interaction is not evidence of a mechanism beyond the map that was evaluated. If the map is the MLP or the softmax, the interaction is the MLP or the softmax.

## B4. What a structural relation would have to add

Direct path patching, the Jacobian, the Hessian, and a factorial intervention already determine the four-term interaction by evaluating the map. A relation among pathways predicts that interaction without those evaluations only if the relation is a sufficient statistic for it.

The architectural restriction that does predict an interaction without a second evaluation is: the pattern is clamped, and the edits are value edits. Then the interaction is zero, because `Y=AVW_O` is linear in `V`. The information used is the attention equation and the intervention semantics. A compatibility residual of the pre-intervention residual is not an input.

If the edits are residual edits, the attention equation does not predict the numerical interaction without the weights, the baseline, and the displacements. Computing it is computing the block. A scalar agreement between two projected states does not determine that number. The fiber counterexample in `06` has two states with the same projection and different attention outputs. Their interactions with a further edit are not constrained by the shared projection.

So the examined relational scores add no prediction about interaction beyond the tuple they are computed from, and the tuple adds a prediction only when it is sufficient for the map. The zero-interaction law for clamped patterns is real and it is the linearity of attention in `V`. It is not a consistency mechanism.

## B5. Conclusion

Distributed contributions constrain a margin when the downstream equation says they do: by inner products, by a joint map that is nonseparable on the intervention, or by a softmax that renormalizes. Each of those is readable from the equation and from interventions on that equation. None of them is the statement that a coboundary of projected states is small. Vector agreement is neither necessary nor sufficient for redundant readout support.

An affine map has interaction zero, so a nonzero interaction at a named pair requires the map to be nonlinear. Nonlinearity is not sufficient: a separable nonlinearity has interaction zero at every pair. A nonzero mixed partial at the baseline is a different condition again. It is not necessary for a nonzero finite interaction, because `(u+v)^3` has mixed partial zero at the origin and interaction `6` at `(1,1)`. It is not sufficient for every finite pair, because `uv-u^2v-uv^2` has mixed partial `1` at the origin and interaction `0` at `(2,-1)`.
