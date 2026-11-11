# Open questions

| # | question | evidence needed |
|---|---|---|
| 1 | Is the sign of φ_rel(5) − φ_rel(4) (P-count < 0, full-budget T1 > 0) a reproducible curriculum-dependent approximation bias? | freeze it as a prediction; new seeds of both arms (and P-count with longer transfer) evaluated with the frozen fit code |
| 2 | Does P-count's coherence and operator fit change with longer transfer (quality matched from the other side)? | P-count 6000-step transfers (≈ 45 min each) |
| 3 | What operation inside layer 2 implements T1's readout-local null route (attention to the readout itself, MLP, residual)? | path patching of attention/MLP outputs at the readout, separately for N and R pairs, with resample controls |
| 4 | Why does the route appear in 4 of 6 T1 parents only? | larger T1 cohort; relation to Investigation IV's partial count copy at T1 readouts |
| 5 | Is the out-of-class residual r_M (most of the null response) structured by a richer operator class? | larger classes (soft colluder × reliability interactions, per-group τ) with held-out surfaces |
| 6 | Do Investigation IV's necessity failures (P4 marginalisation, T11 rotated control) persist with covariate-corrected encoders and natural-marginalisation references? | prospective rerun of the Inv IV necessity protocol with corrected encoders on the cohort |
| 7 | Does probability-space vs logit-space disagreement reflect saturation regions? | stratify by base posterior confidence; analysis on the archived arrays |
| 8 | How do training stages shape the T1 coherence (it appears between 500 and 2000 transfer steps)? | a pre-specified checkpoint schedule within one transfer run |
