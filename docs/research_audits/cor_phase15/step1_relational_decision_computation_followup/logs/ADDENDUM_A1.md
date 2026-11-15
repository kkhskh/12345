# XIII-B post-freeze addendum A1 (registered BEFORE evaluation on the seed-21 units; NOT part of the frozen set Q1-Q6)

Motivation: math review 1 showed that Q1's POE null family is handicapped (unnormalised exponents), so Q1 cannot discriminate a mixture from
normalised log-linear pooling. On development models (logs/fit_powermean_dev.json, 18 models) the nested power-mean family
(rho = 1 mixture, rho -> 0 normalised log-linear pooling) gives rho 0.58-0.62 for all 8 early-readout models (lik_RO1 >= 0.8),
0.87-0.98 for all 6 late-readout models (lik_RO1 <= 0.06), intermediate 0.62-0.75 for hybrids; corr(lik_RO1, rho) = -0.90.

Addendum prediction AQ1 (exploratory status: registered post-freeze, before the seed-21 power-mean fits are run):
  for each seed-21 unit, with lik_RO1 taken from its frozen class_features output:
    lik_RO1 >= 0.7  ->  rho <= 0.70 ;    lik_RO1 <= 0.2  ->  rho >= 0.85 ;   (units in between: reported only)
  and across the five units corr(lik_RO1, rho) <= -0.7.
Procedure: scripts/fit_powermean.py on the seed-21 specs with FP_STREAM=79_600_000, FP_TAG=_conf13b_addendum.
