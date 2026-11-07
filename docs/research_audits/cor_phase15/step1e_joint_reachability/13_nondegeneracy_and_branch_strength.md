# 13. Where the gap stays and where it shrinks

**Order-one paths.** Special gain, `ρ = 1`: attention readout `0.7070997102248`, MLP readout `-0.888384538745181`, raw response `0.818715171479619`. Strong gain, `ρ = 2`: attention `2.828420053704895`, MLP `-1.776770867462384`, joint branch about `1.051649186242511`.

**Not a near-zero weight.** The strict theorem uses any positive `μ`, `α`, and `s_gain`. Setting either structural gain to zero removes a path. With `μ = 0` the attention increment is identically zero and the joint set equals the product `{0} × S_M`. That is degeneracy, and it is excluded by `μ > 0`.

**Saturated base.** At coordinate `2`, the same unit gains produce a strict gap whose size is below `10^{-6}`. The weights are not small. The LayerNorm derivative is small. A report that quoted only this base would be a near-zero response statement. The zero base is the order-one case, and it is inside the same theorem.

**More than one direction.** The line has two orientations. The forward segment is proved; the backward segment has its marginal maxima at the opposite ends for the same reason, because `f'` and `g'` do not change sign. Off the line the special bound fails. The edit `(1, 1.2)` produces a positive branch sum `0.181627925474685`. Constant edits are in the kernel of both LayerNorms. With `w = n` they also lie in the kernel of the raw readout, so both sides of the comparison are zero. That direction does not exhibit a gap and does not violate the line theorem.

**Two heads.** Each head carries half of the isotropic value map. Zeroing the first head's output block halves the attention increment. The construction is not a single-head identity with a second head written in and ignored.

**Earlier token.** Theorem S2 and Theorem S3 include token `0`. At step `1` the earlier-token branch sum is `-0.181279889763888`, with attention `0.7070997102248` and MLP `-0.888379599988688`. The read-token skip is zero, so this is the whole raw response.
