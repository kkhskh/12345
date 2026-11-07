# 09 — Family C: error correction versus gating

Three systems, all possible as mathematics:

1. Conflict lowers `L`. (The original COR hope.)
2. Conflict triggers a correction that restores `L`.
3. Conflict is unread because `w` ignores it.

**System 3** is N1: the statistic of conflict can be large and `ΔL` can be zero.

**System 1** is not forced by residual addition. Cancellation of addends can leave `L` correct.

**System 2**, constructed: majority of three bits. One flip is corrected. Pairwise disagreement is 2 for a one-flip (still correct) and 0 for the all-wrong codeword. Disagreement does not rank correctness (`test_majority_vote_is_real_correction_and_disagreement_does_not_rank_it`). The correcting object is the majority function, a signed vote, not a nonnegative residual.

**Gating is not correction.** `y = tanh(a) + ε b` can ignore a large `b`. Flipping `a` still flips `y`. There is no discrete restoration from redundant copies (`test_nonlinear_gating_can_look_like_correction`). A residual skip preserves `h`; it does not compute a parity check on `h`.

**Falsifiable signature of correction, if anyone later claims a Transformer implements it:** a perturbation family of size 1 in a discrete code is absorbed, size 2 is not, and the absorbing map is not a saturating scalar nonlinearity of a single coordinate. `EMPIRICALLY_UNTESTED`. This step does not run that test. Existence of a toy is not evidence of the architecture.

**Status.** System 2 is a constructive restricted class. It is not identified in Transformers. It does not revive `O`.
