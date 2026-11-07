# 11 — Family E: stability, not correctness

Stability of a sign under `||δ||≤r` follows if `|L(h)|` exceeds the increment bound `S(r)`. That is the corrected reading of Update Proposition 2. A pointwise Jacobian is not `S(r)` (`L(t)=1-2t^2`).

**Can a relational constraint tighten `S`?** Only by restricting the perturbation family to a subset where `L` is less sensitive, with the subset identified without `∇L`. Kernel of a probe that `L` actually uses would do that — and that identification is sufficiency of the probe, which kernel edits in Direction 1 contradict for the board probe.

If the subset is identified using `∇L`, the tighter bound is a restriction of the gradient bound. Not distinctive.

**Worst-case versus average-case.** A Lipschitz constant is worst-case on a ball. Average-case stability is a distributional statement and is `OUTSIDE_SCOPE` without a distribution. It would still be a statement about `L`, not about `D`.

**Stability is not correctness.** A constant wrong logit is perfectly stable.

**Status.** Generic sensitivity mathematics. `REDUNDANT_WITH_EXISTING_THEORY`.
