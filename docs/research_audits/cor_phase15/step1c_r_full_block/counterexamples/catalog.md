# Counterexamples

The statements, equations, and numerical values are in `17_counterexample_catalog.md`. The checks are in `tests/test_block.py`. This file records the map from each catalog row to the test that recomputes it, so the catalog is not a second unverified source.

| catalog row | test |
|---|---|
| 1, 2 | `test_shift_is_exact_and_scale_is_not_when_epsilon_is_positive` |
| 3 | `test_value_only_linearization_can_have_the_wrong_sign` |
| 4, 8 | `test_zero_first_order_response_need_not_be_a_zero_finite_response` |
| 5 | `test_same_projection_opposite_intervention_signs` |
| 6 | `test_exact_equations_hold_while_margins_differ_across_inputs` |
| 7 | `test_rank_one_transport_can_disagree_when_the_margin_does_not` |
| 11 | `test_biased_addition_makes_both_necessary_with_zero_interaction` |
| 11, zero baseline | `test_zero_bias_addition_cannot_make_both_deletions_necessary` |
| 12, 13 | `test_head_derivatives_add_and_token_edits_need_not` |
| singular-value cutoff | `test_a_loose_singular_value_cutoff_misreads_small_epsilon` |

The maps are `proofs/block_math.py`. Weights come from `make_params(seed)` and are not fit to a response.
