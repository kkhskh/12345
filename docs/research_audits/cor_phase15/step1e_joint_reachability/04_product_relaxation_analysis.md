# 4. Three relaxations

Set A is the graph of the lemma, restricted further to a segment when a theorem says so. Set B is `S_A × S_M`. Set C replaces each marginal by a Euclidean ball.

On the structural line at step `1`, with the special gains,

```text
||a_r|| = 0.7070997102248,    ||b_r|| = 0.888384538745181,
w · (a_r + b_r) = -0.181284828520381.
```

The Set C ceiling that uses those two radii and the readout direction on each ball is their sum, `1.595484248969981`. It is a valid upper bound and a poor one. It permits a pair in which both increments point the same way. The block never produces that pair on this segment: `a_r` is a positive multiple of `n` and `b_r` is a negative multiple of `n`.

Set B on the same segment is sharper than Set C and still larger than Set A. The attention readout ranges over `[0, 0.7070997102248]` and the MLP readout ranges over `[-0.888384538745181, 0]`. The product includes the corner `(0.7070997102248, 0)`. The image of the segment does not. That missing corner is the support gap.

Replacing Set B by Set C and then beating Set C is not a relational result. The comparisons in `12_strict_relational_advantage.md` use Set B.
