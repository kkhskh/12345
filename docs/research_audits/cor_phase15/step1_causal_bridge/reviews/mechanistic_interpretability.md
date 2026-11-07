# Reviewer C — mechanistic interpretability

Independent pass. The reviewer read files 03, 05, and 11.

## Objections

1. **Theorem Z is substitution into one map `f` of one argument.** A pre-norm block `h + Attn(LN(H)) + MLP(LN(h+Attn(LN(H))))` is not a constant linear function of one token. Softmax, LayerNorm, and the MLP are nonlinear. For `P=I`, no constant `ρ` intertwines the block on an open set. A one-token edit recomputed through attention is not `f(h+δ)` for an `f` of that token alone. Domain rows A and B do not describe a Transformer trace.

2. **The residual-skip test is the doubling map `h+h=2h`.** For `h+tanh(h)`, neither `I` nor `2I` is an intertwiner. "Include the branch" returns the residual to zero only in the linear test. The addend identity `P(m+u)=Pm+Pu` is exact once `u` is already a node. The edge that builds `u` is the nonlinear one, and the ReLU test already leaves a residual there.

3. **The attention-mix test is a constant average of two scalars.** Softmax coefficients depend on the inputs. A shared linear pair that matches one pattern fails on another. Theorem Z's universal quantifier over inputs does not fire. A correctly parented softmax edge can still have a nonzero shared-linear residual. That is approximation error, not an omitted parent.

4. **Family E's "unique path" is the linear parent `φ_v = J φ_u`.** Equation (11) at one point leaves a linearization remainder on a finite step. The derivative of attention in one token runs through the softmax weights and therefore through other tokens.

5. **Family H.** `I=0` is not a consequence of residual addition once the sum is passed through a nonlinearity or a shared softmax.

6. **Geiger et al.** CE-13 does not sit strictly below interchange invariance. The claims are incomparable. A zero linear residual is neither necessary nor sufficient for exact transformation of the algorithm whose output is the logit. Mechanism replacement, as opposed to translating an activation and reusing `f`, is outside Theorem Z.

7. **Sutter et al.** A shared linear `ρ` is outside the unrestricted alignment class. That does not place it under Theorem Z. An input-dependent attention coefficient is the expressive object the vacuity result is about.

The finite-map calculations were not rejected. The Transformer-wide reading was.
