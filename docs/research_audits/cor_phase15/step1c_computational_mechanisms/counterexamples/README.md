# Counterexamples

Each row is a finite map used to refute a universal sentence or to mark a boundary. The executable check is in `tests/test_step1c_math.py`. The map is not a pretrained model. A passing check shows the arithmetic of that map.

| id | map | sentence it hits | test |
|---|---|---|---|
| C1 | `u^2+v^2` | nonlinearity implies nonzero `I` | `test_error1_separable_square_has_zero_interaction` |
| C2 | `(u+v)^3` at `(1,-1)` and `(1,1)` | one pair's interaction is the interaction of the formula | `test_error1_cubic_vanishes_at_one_pair_and_not_another` |
| C3 | `uv-u^2v-uv^2` at `(2,-1)` | nonzero mixed partial at `0` implies every finite `I` is nonzero | `test_error1_nonzero_baseline_hessian_does_not_force_every_finite_I` |
| C4 | `\tanh(u)+u^3+\cos(v)` | unmixed channels interact | `test_error1_channels_that_do_not_mix_stay_separable` |
| C5 | `O` with two values of `M` | changing `M` changes `O` | `test_error2_changing_only_M_changes_D_and_not_O` |
| C6 | LN on `(1,0)` with `w=(1,2)` | fixed readout sign survives LN | `test_error3_fixed_readout_need_not_keep_its_sign_under_layernorm` |
| C7 | LN with `\gamma=(-2,1)` | `\gamma` can be ignored | `test_error3_learned_affine_can_flip_a_sign_that_plain_layernorm_kept` |
| C8 | `P_u=(1 0)`, `P_v=(0 1)`, `\delta=(0,1)` | `ker P_u` preserves `r` | `test_error4_kernel_of_one_projection_need_not_preserve_the_residual` |
| C9 | `Q=(1 0)`, heads `\pm e_2` | a blind linear statistic determines both signs | `test_n1_narrow_counterexample_still_stands` |
| C10 | scalar causal QKV, weights `1`, baseline `(1,1)` | residual edits are additive | `test_residual_intervention_moves_qkv_together_and_can_interact` |
| C11 | `W_Q=0`, no bias, no LayerNorm | softmax implies every family of that map has nonzero `I` | `test_zero_query_map_makes_residual_edits_additive` |
| C11b | `Q=b_Q` with `W_Q=0` | the previous row survives a query bias | `test_query_bias_keeps_scores_dependent_on_keys_when_W_Q_is_zero` |
| C11c | per-token LN, constant pattern, two coordinates of one token | constant `A` makes every edit additive | `test_constant_pattern_through_layernorm_is_not_additive_inside_one_token` |
| C12 | `relu(u+v)` and a diagonal ReLU | shared or elementwise ReLU implies a nonzero `I` at the tested pair | `test_mlp_relu_is_not_automatically_an_interaction` |
| C13 | two states, same `P`, different keys | that `P` determines attention outputs | `test_same_projected_residual_different_attention` |
| C14 | `z_v-z_u` on `(0,0)` and `(1,1)` | the relation retains the tuple | `test_a_relation_collapses_pairs_that_the_tuple_separates` |
| C15 | recomputed `\tanh` | a structural residual tracks `R` | `test_recomputed_structural_residual_is_identically_zero` |
| C16 | rank-one `\rho` | a zero residual identifies the computation | `test_manufactured_compatibility_zeroes_one_pair` |
| C17 | `5h+3` and `h^2+3` | `J` applied to the absolute state, or a finite step with no remainder | `test_affine_linearization_keeps_the_offset` |
| C18 | total, skip-only, and clamped descendant | the three intervention rows agree | `test_three_intervention_semantics_differ` |
| C19 | `(1,100)` and `(1,-100)` | vector agreement is readout redundancy | `test_vector_agreement_is_not_readout_redundancy` |
| C20 | both contributions `-1` | stability is correctness | `test_single_ablation_robustness_needs_the_remaining_inner_product` |
| C21 | `2W_1` and `W_2/2` with ReLU | the internal score is a function of `F` | `test_function_preserving_rescale_changes_an_internal_score` |

`test_value_only_attention_interaction_is_exactly_zero` is a positive identity under a clamp, not a counterexample. `test_error3_plain_rmsnorm_preserves_linear_sign_and_gamma_need_not` is a lemma plus a `\gamma` counterexample. `test_error3_jacobian_of_layernorm_is_singular` checks the kernel of `D\mathrm{LN}` at one vector in `\mathbb{R}^3`. `test_error4_the_full_measurement_kernel_does_leave_the_residual_fixed` checks the repaired hypothesis on one vector.
