# Reviewer 1 — Mathematical correctness

Provenance: subagent [mathematical review](3f3cfe6e-8f2a-41c3-aedf-115b3659d77e), run in this session against the draft. Not an external mathematician. The same pass covered Reviewer 6's counterexample brief; the counterexample list is in `06_counterexamples.md`.

## Claims that survived

The `C^2` rectangle equivalence, the Taylor coefficient `t^2 uv F_uv(0,0)`, the cubic identity `3uv(u+v)`, `D=10/101`, the LayerNorm sign pair, the plain-RMSNorm scale lemma, the residual kernel `P_v A-\rho P_u`, clamped-pattern linearity as an implication, and the affine offset all held under the hypotheses written with them.

No sentence said nonlinearity is unnecessary for a nonzero interaction. No finite check was called a proof.

## Defects

1. `11` said a single zero of `I_F` is not necessary for separability. It is necessary. It is not sufficient.
2. `10` said one zero interaction implies separability in the two arguments. `(u+v)^3` at `(1,-1)` is the counterexample.
3. `11`, `10`, and `18` used "iff" between zero value-interaction and the form `AVW_O`. `u^2+v^2` has zero interaction and is not that form.
4. `03` reversed the fiber quantifiers.
5. `11` said residual zero if and only if the child was assigned by the structural equation. A clamp that copies `f(h_u)` has residual zero too.
6. The RMSNorm test comment said `w·h=0.5`. For `h=(1,1)` and `w=(1,-0.2)` the value is `0.8`.

## Disposition

All six were corrected in the files named above, and the comment was corrected in the test. The orthogonal-to-ones sign lemma was added because a later pass showed the sentence "no extra hypothesis saves the sign" was too broad. That lemma is in `proofs/layernorm_readout.md` and `test_readout_orthogonal_to_ones_keeps_its_sign_under_plain_layernorm`. It does not restore the universal sign claim.
