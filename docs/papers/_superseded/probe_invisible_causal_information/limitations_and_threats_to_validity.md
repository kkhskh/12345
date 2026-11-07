# Limitations and threats to validity

This file expands Section 6 of the manuscript. It does not add results.

## Selection

Pairs require nonempty became-legal and became-illegal sets and same-player occupancy buckets. They over-represent positions where a target board is a meaningful legal-set counterfactual. Wrong targets are chosen from simulator features before \(Y\). SEM reuses NST pairs after those \(Y\) values were known; it is exploratory. Editor A decoding quality affects any comparison that conditions on successful board edits; NST’s primary gate does not require Editor A success, but the Editor A contrast is still a comparison against a weak baseline (08R2 NEGATIVE).

## Multiplicity

SPEC, COST, forensic, and fixed-JS inspect one 48-pair list. NST and SEM inspect another. FRESH and VNF are later cohorts. A 95% interval on fixed-JS is not family-wise error controlled across the project. We treat preregistered primary contrasts as the claims, and we do not promote secondary JS targets that were never run.

## Effect size

\(Y\) is an additive difference of two probability masses, each in \([0,1]\), so \(Y\in[-1,1]\). A movement of \(+0.02\) is 2% of that raw scale, not a 2% relative improvement in accuracy. Absolute learned \(Y\approx-0.58\) means the typical edited output still puts more mass on became-illegal than became-legal squares.

## Naturalness

No test in the archive establishes that edited residuals are on-manifold. Probe preservation is a linear constraint, not a generative constraint.

## Information budget

The learned map receives the true target legal mask. Comparisons with methods that do not receive that mask are not information-matched. Oracle donors receive \(h_D\). Editor A receives a decoding objective. Random nulls receive neither board nor donor.

## Statistical unit

Clustering by recipient or triplet is appropriate. Treating 7 directions on one recipient as 7 games is not. Bootstrap seed 82 is reused across 48-pair studies; it is a computational seed, not independent evidence.

## External validity

One Othello-GPT checkpoint. No language-model transfer. No claim about other probes, layers, or games.
