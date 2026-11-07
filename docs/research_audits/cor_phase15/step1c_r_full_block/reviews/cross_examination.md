# Reviews

These are separate passes in this session. They are not external referees. The passes are [calculus](30aa17b0-4c69-43e1-85e6-32cd2abad62d), [architecture](4725b44d-602c-4fa5-843c-6f7c42727c62), [causal inference](f27b34df-5369-4b7c-83b6-9212c63cd80b), [counterexamples](e6db242e-e018-4535-88ea-c2021050cbda), [reduction](c0971297-2ee2-4997-bb70-6744095657eb), [methodology](1a1d8b94-8a99-4d09-94b6-5f5af4e2eb23), and [blind spots](c4c2f0c4-3a2d-4087-aa55-7d66b8edd191).

Agreement among them is not a proof. The sentences they showed to be false were rewritten, and the rewritten sentences were checked again.

## What was false, and what replaced it

The scale biconditional in R5 originally said that `α>0` leaves LayerNorm unchanged for all `h` if and only if `ε=0`. The identity scale `α=1` holds at every `ε`. Dimension `1` has nothing to scale. If every gain is zero the map is the bias at every `ε`. R5 now excludes those cases. The normalized vector, which does not see `γ`, still matches for every admissible `h` under `d≥2` and `α≠1` only at `ε=0`.

"A non-constant `dH_1` is not killed" was false at `ε=0` along `z`, and at `ε>0` when two gains vanish. The replacement states the nondegenerate case and those two larger kernels.

"The radial branch term is not zero" was false when `z=0` and when `γ` vanishes on the support of `z`. The replacement says so.

"The skip makes every radial direction visible to the block" was false. The block derivative is the skip plus the branch, and the branch term can cancel the skip. A constant is the case where the branch term is identically zero. That sentence now says this.

"The first-order block response is the sum of the two head increments" dropped the skip. The replacement is `dH + dA^0 + dA^1`. A constant edit has both head increments zero. The response is `a 1^T w` only when the edited token is the read token, and it is zero when `1^T w=0`.

"Attention is the sum of the head maps after `W_O`" counted `b_O` wrong. `concat @ W_O` is the sum of the head-row pieces. The bias is added once. Directional derivatives add because the bias has derivative zero.

"Signs of `Dm` on a spanning set determine the gradient" was false. The real values do. The signs fix an orthant. A sign rule that is unchanged under `δ ↦ -δ` still cannot match both directions when `Dm·δ≠0`.

"Every proper projection fails the fiber test" was false. Feature 3 fails one pair. Every projection succeeds on the constant-shift family, because `R` does not depend on `H`.

Feature 3 was described as chosen without the response. It was chosen by looking at this pair's responses. The writeup now says the reported pair and the selection pair are the same pair.

The ratio `R/step² ≈ -14` was described as the degree-2 Taylor coefficient. It is the one-step ratio, which still contains higher-order terms.

A clamped attention output was called linear in `V`. It is affine. `b_O` does not depend on `V`.

File 22 had restated R21 as a claim about `sign(Dm)`. It now stays with `sign(R)`.

The positive-scale sign remark was called a Step 1C discovery. A positive multiple preserves signs because it is positive. The sentence now says that.

## What the passes did not break

The Jacobian formula, the `ε>0` kernel `span{1}`, the image `1^⊥`, the one-zero and two-zero rank counts, the `ε=0` kernel `span{1,z}`, the masked softmax derivative, and the `tanh` chain rule. The causal mask blocks attention outputs at earlier tokens. Zeroing `dQ` and `dK` freezes the pattern. The shift identity survives output bias and MLP bias. No zero-baseline sum makes both deletions necessary. The constant-shift response is a real restricted identity and is pre-norm mean subtraction plus the skip. No pass produced a non-tautological constraint that these documents had discarded.

The counterexample pass did not falsify a proved row under the attacks it was given. The later quantifier repairs are restrictions of sentences, not withdrawals of R1–R4 or R6–R8.

## What this does not certify

A second pass can miss a quantifier the first pass also missed. The scale paragraph was repaired three times. The version in `02` and R5 is the one that survived the last check of those sentences. The package tests do not quantify over every gain pattern. They check the formula, the nondegenerate kernel, one zero gain, and two zero gains on one vector.
