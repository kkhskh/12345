# Counterexamples

The ten attacks required by the assignment are recorded with their numbers in `17_counterexample_register.md`. The constructions are executed in `tests/test_step1e.py` and `proofs/examples.py`.

The attack that limits the theorem most sharply is the edit `(1.0, 1.2)`. It uses the same frozen special-gain block and the same zero base as Theorem S3, and the branch sum is positive. The invariant-line hypothesis is necessary for that bound. A second limiting attack is `s_gain = -1`: the joint set remains a strict subset of the product, and the support gap disappears. Dependence alone is not an improvement.
