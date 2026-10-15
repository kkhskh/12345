# MVP Theorem Set: Numerical Verification

Random sheaves: 400 trials.

- [PASS] O(s) = s^T L s matches the edge-residual definition — max rel err 8.32e-16
- [PASS] D(s) = ||(I-P)s||_M^2 matches direct minimization — max rel err 3.81e-16
- [PASS] Theorem 1 sandwich holds — 0/400 violations
- [PASS] Theorem 1 bounds are tight at extreme eigenvectors — max rel err 1.23e-15
- [PASS] d s lies in im d^0 for every s — max residual 1.86e-15
- [PASS] a generic 1-cochain does not lie in im d^0 (non-surjective d only) — min residual 0.038 over 253/400 non-surjective trials
- [PASS] Proposition 1 closed form is stationary — all 6 cases
- [PASS] Proposition 1 closed form matches numeric minimum when A > 0 — 4/4 positive-definite cases

## Condition number of the energy-distance equivalence

- median kappa = lambda_max / lambda_min_plus: 378.245
- maximum kappa: 4831615.584

These are properties of the random sheaves, not of any transformer. A large kappa means
the Laplacian residual is a loose proxy for distance to global sections, so kappa should
be reported per experiment.

## Indefinite-Hessian regime

- cases where A was not positive definite: 2/6
- of those, cases where the closed form still found the minimum: 0

This is the point of the positive-definiteness condition in Proposition 1: when
lambda is too small relative to the negative spectrum of H_O + mu F, the closed-form
delta_star is a stationary point that is not a minimizer.