# Reviewer C — Transformer computation

Independent pass. Sequential pre-norm does not freeze addends; the next map is not a linear readout of `h+a+m`; the attention test supplies logits rather than QK; LN is a reader-side map, not a free gauge on the skip; majority vote is not a block primitive; causal decoder clamps do not update earlier tokens.

Accepted: toys are toys. The negative that a constant linear `ρ` is not the block still stands. Direct-effect `w·p_i` is not the total effect through MLP(LN(h+a)).
