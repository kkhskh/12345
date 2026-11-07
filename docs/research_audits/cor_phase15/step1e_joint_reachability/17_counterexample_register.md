# 17. Counterexample register

Each item says what was varied and which claim it limits. The block weights stay fixed inside an item unless the item is explicitly a comparison across weights.

1. **Joint set equals the product.** `μ = 0`, other special gains, read-token segment. The attention increment is `0`. The joint set is `{0} × S_M`.

2. **Strict subset, support unchanged.** `s_gain = -1`, `μ = 1/2`, `ρ = 1`. The image is a curve, so it is a strict subset of the box `[0, f(1)] × [0, g(1)]`. Both readouts are maximized at `t = 1`, and the sum equals `1.595484248969981`. Empty interior is not a support gap.

3. **Strict subset and a strict support gap.** Theorem S3. The corner `(f(ρ), 0)` is not attained. At `ρ = 1` the joint branch maximum is `0` and the marginal sum is `0.7070997102248`.

4. **Anti-alignment fails off the construction.** Edit `(1, 1.2)` from the zero state. Attention readout `-0.7067534927402195`, MLP readout `0.8883814182149048`, sum `0.181627925474685`. The line orientation is not a property of every direction.

5. **Base cancellation does not survive.** `R'(0) = -50157.11388300843` and `R(1) = 0.818715171479619`.

6. **Large gains do not restore the product corner.** `μ = 2`, `s_gain = 2`, `ρ = 2`. Theorem S2 still separates the argmaxima. The grid maximum of the joint branch is about `1.052`, below `2.828`. A loose spectral split can become worse at large gain; this monotone gap does not rely on that split.

7. **Final LayerNorm reverses the raw picture.** At step `0.2`, raw response `0.018548133077188` and normalized response `1.374816323880916`.

8. **Same attention projection, different MLP increment.** Width-3 singular value map, reflections at radius `0.4` and angle `0.7`. The attention rows match exactly in the test. The MLP rows are negatives, with norm above `10^{-3}`. Constant fibers do the opposite: they preserve both increments. In width 2 the image of LayerNorm is a single line, so an invertible value map cannot separate `b` from `a`.

9. **Skip surrogate large, response small.** Step `0.2`: skip `0.2`, raw response `0.018548133077188`.

10. **Skip surrogate small, response not small.** Earlier-token step `1`: the read-token skip is `0` and the raw response is `-0.181279889763888`.

The linear model supplies the same lessons in closed form. Nonnegative-parallel maps have gap zero. The diagonal pair in note 5 has gap `4` at radius `2`. A base Jacobian gap is Theorem S4 and is not item 5.
