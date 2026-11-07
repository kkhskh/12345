# REL-XOR3 intermediate-variable test (PROTOCOL_ADDENDUM_XOR3.md; generated from logs/xor3_eval.json)

Each cell gives the fraction of decisive pairs (opposite agreement **and** opposite p) whose patched decision matches each hypothesis's prediction. H_AB means "U carries agreement", H_y means "U carries the target", and H_AP / H_BP mean "U carries sign(ap)" or "sign(bp)".

| model / site | acc | k | addendum verdict | S_rel: H_AB / H_y / H_AP / H_BP | S_das_AB → H_AB (held-out) | S_das_y → H_y (held-out) |
|---|---|---|---|---|---|---|
| TF1/s2/resid_mid0@R | 1.0000 | 2 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.477 | 1.000 |
| TF1/s3/resid_mid0@R | 0.4983 | 2 | UNRESOLVED | 0.457 / 0.543 / 0.488 / 0.512 | 0.495 | 0.505 |
| TF1/s4/resid_mid0@R | 0.9999 | 2 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.477 | 1.000 |
| TF1/s5/resid_mid0@R | 1.0000 | 1 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.477 | 1.000 |
| TF1/s6/resid_mid0@R | 0.7490 | 2 | UNRESOLVED | 0.280 / 0.720 / 0.505 / 0.495 | 0.635 | 1.000 |
| TF2/s2/resid_mid1@R | 0.9999 | 1 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.000 | 1.000 |
| TF2/s2/resid_post0@ALL | 0.9999 | 3 | UNRESOLVED | 0.375 / 0.625 / 0.490 / 0.510 | 0.812 | 1.000 |
| TF2/s3/resid_mid1@R | 0.9998 | 2 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.000 | 1.000 |
| TF2/s3/resid_post0@ALL | 0.9998 | 2 | UNRESOLVED | 0.277 / 0.723 / 0.532 / 0.468 | 0.945 | 1.000 |
| TF2/s4/resid_mid1@R | 0.9999 | 2 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.142 | 1.000 |
| TF2/s4/resid_post0@ALL | 0.9999 | 3 | UNRESOLVED | 0.215 / 0.785 / 0.483 / 0.517 | 0.193 | 1.000 |
| TF2/s5/resid_mid1@R | 0.9998 | 2 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.000 | 1.000 |
| TF2/s5/resid_post0@ALL | 0.9998 | 3 | UNRESOLVED | 0.383 / 0.617 / 0.498 / 0.502 | 0.927 | 1.000 |
| TF2/s6/resid_mid1@R | 1.0000 | 2 | TARGET_ONLY | 0.000 / 1.000 / 0.492 / 0.508 | 0.000 | 1.000 |
| TF2/s6/resid_post0@ALL | 1.0000 | 3 | UNRESOLVED | 0.288 / 0.712 / 0.493 / 0.507 | 0.565 | 1.000 |
| TF2AO/s2/resid_post0@ALL | 0.9998 | 2 | AGREEMENT_INTERMEDIATE | 0.975 / 0.025 / 0.493 / 0.507 | 1.000 | 1.000 |
| TF2AO/s3/resid_post0@ALL | 0.9999 | 2 | UNRESOLVED | 0.710 / 0.290 / 0.532 / 0.468 | 0.975 | 1.000 |
| TF2AO/s4/resid_post0@ALL | 0.9995 | 2 | AGREEMENT_INTERMEDIATE | 0.920 / 0.080 / 0.502 / 0.498 | 0.842 | 1.000 |
| TF2AO/s5/resid_post0@ALL | 0.9999 | 2 | UNRESOLVED | 0.758 / 0.242 / 0.600 / 0.400 | 0.947 | 1.000 |
| TF2AO/s6/resid_post0@ALL | 0.9997 | 2 | AGREEMENT_INTERMEDIATE | 1.000 / 0.000 / 0.508 / 0.492 | 0.998 | 1.000 |

The content of each S_rel (additive vs interactive) is in `INTERVENTION_RESULTS.md`, section E5. TF1 seeds 3 and 6 did not learn the task (accuracy 0.498 and 0.749), which is an optimisation failure; they are reported but not interpreted.
