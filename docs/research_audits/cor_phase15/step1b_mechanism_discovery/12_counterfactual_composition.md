# 12 — Family F: composition, with truncation defined

**Semantics.** Nodes are a DAG. `do(V_i ← v_i` for `i in I)` replaces those structural equations by constants and evaluates the rest in topological order. Ancestor and descendant may both lie in `I`. That is defined. Last-write order without truncation is a different, order-dependent operation and is not used here.

**Chain `c=2p`, `y=c`.** Truncation of the child screens the parent, so `do(p,c)` and `do(c)` yield the same `y`. The four-term contrast then equals minus the parent-only effect. It is defined, and it is degenerate as a two-cause interaction: one of the two clamps already screens the other. It is not a mixed partial of the truncated map `G(p,c)=c`. The code class `TruncationSCM` is this two-node calculator, not a general DAG evaluator. Parallel-parent tests (`I=0`, `I=2`) are a different graph. They are not nested truncation.

**Unintervened residual.** Two systems with additive residual `0` at the origin have `I=0` and `I=2` respectively (`test_same_unintervened_residual_two_interaction_values`). So `I` is not a function of that residual.

**Parallel addends, linear readout.** `I=0` identically (`test_randomized_linear_interaction_is_zero`).

**Parallel addends, square bottleneck.** `I=2` at unit clamps, residual of the sum 0 (`test_shared_square_bottleneck_has_interaction_two`).

**Hypothesis H-COMPOSE.** For two free parallel arguments, the four-term contrast is the mixed second difference of `F`. If `F` is affine, `I=0` identically. If `F(u,w)=(u+w)^2`, `I=2` at unit clamps. Both statements use `F`, which is the outcome map. They are not forecasts of an unknown `L`, and they are not Hessians of a truncated ancestor-descendant graph.

**Comparison.** Factorial effects, Hessians, mediation. A skeptical researcher would call H-COMPOSE those names. `REDUNDANT_WITH_EXISTING_THEORY` as COR. `PROVED_UNDER_RESTRICTED_ASSUMPTIONS` as the linear-versus-square calculation.

The visible/null factorial letter is not reused as evidence for or against H-COMPOSE.
