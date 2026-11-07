# Review B. Transformer architecture

Same cloud environment. Not an external referee.

**Objection.** Uniform attention under `W_Q=W_K=0` is a degenerate block, so Theorem U only unbounded a toy.

**Resolution.** The theorem's claim is unboundedness over the weight class of this architecture, not a typical draw. The degenerate point is inside the class. A bound valid for every weight must survive it. The Gaussian ensemble is a different quantifier and is not where Theorem U is stated. On that ensemble the explicit `Λ` is finite and much larger than `||g||`.

**Objection.** Biases on `Q,K,V` were omitted, so the shift identity might use a missing cancellation.

**Resolution.** An additive bias after `XW` does not depend on `X`. A constant shift leaves `X` fixed, so it leaves `Q,K,V` fixed whether or not those biases are present. The MLP bias is constant. The identity does not need them to vanish. They were not added to the numeric Jacobian.

**Objection.** The raw readout is not an LM logit.

**Resolution.** Agreed. The final-normalization check shows the constant skip contribution becomes zero after another LayerNorm. No identity in this package is stated for a final-normalized logit except that negative transfer.

**Status.** Architecture objections restrict the quantifiers. They do not restore a centered bound in budget B0.
