# 16. Theorem register

Statuses used here: `PROVED`, `COUNTEREXAMPLE_FOUND`, `REDUCTION`, `WITHDRAWN_AS_OVERBROAD`, `OPEN`. A `PROVED` row has an analytic argument and a second numerical route. `REDUCTION` means the statement is true and is an existing theorem or an immediate factorization, not a new mechanism.

## Dependency graph

```text
Theorem U (raw, unbounded branch scale)
  --PROVED--> no finite Regime 0 bound on that class
  --does not support--> "no Regime 1 bound at fixed weights"
  --blocked by--> final-LN diameter when γ, β, w are fixed

Theorem L
  --PROVED--> isotropic Λ ≥ ||∇m||
  --does not support--> B(H, δ) ≥ ||∇m|| ||δ|| for every directional B

Theorem S  (Regime 1 Lipschitz composition + exact skip)
  --PROVED--> |R| ≤ B_dir
  --PROVED--> B_dir ≤ B_iso, gap = ||w|| ||δ|| - |skip|
  --does not support--> a joint-path mechanism
  --does not support--> final-LN readout
  --REDUCTION--> mean value inequality + triangle inequality

Theorem Align
  --PROVED--> B_align ≤ B_dir
  --factors as--> two unary gains ||W_O w|| and ||W2.T w||

Theorem Kernel-out
  --PROVED--> R = 1_{t=r} w·δ when both gains vanish
  --does not beat--> B_align, which is already 0
  --COUNTEREXAMPLE_FOUND on final LN--> CE-final-kernel

Theorem Persist
  --PROVED--> {rate < 1/2} is a nonempty open set on which B_dir < ||∇m|| ||δ||
              for every H and every nonzero δ ⊥ w
  --does not support--> typicality, tightness against |R|, or a gradient-remainder win

Product-ball ceiling
  --PROVED--> max |a·v_O + h·v_2| = ||v_O|| ρ_c + ||v_2|| ρ_h
  --supports--> no radius-only joint bound beats the separate sum
  --does not support--> "no tighter certificate exists under any future coupling argument"

R1 path sum
  --REDUCTION--> Q_1 = ∇m·δ - skip
  --COUNTEREXAMPLE_FOUND--> base cancellation, |R|=0.169 at step 2

R3
  --REDUCTION--> |R_m - R_s| ≤ B_branch, approximate implementation

Overnight universal B1 impossibility
  --WITHDRAWN_AS_OVERBROAD--> CE-L-scope is a weight-informed finite bound
                              below the Cauchy number on an injective-QKV model
```

## Rows

| id | statement | status | analytic route | independent check |
|---|---|---|---|---|
| U | Raw nonconstant response is unbounded as branch scale varies, weight-free data fixed | PROVED (inherited, scope restated) | overnight closed form | not re-fit; scope limited in `02` |
| U-final | Fixed final LN, fixed `γ, β, w`: `|Δm| ≤ 2 ||w|| R_f` | PROVED | `||z/s||≤√d` | seed-0 number `3.505739880594811`; orthogonal example inside it |
| L | Isotropic Lipschitz constant ≥ gradient norm | PROVED | difference quotients | directional examples sit below the Cauchy number without violating L |
| S | Directional finite bound `B_dir` | PROVED | Lemmas 1–4 and the skip split | loop forward; grid of scales and tokens |
| S-iso | `B_dir ≤ B_iso` with the skip gap | PROVED | expand `(1+L_A L_1)(1+L_M L_2)` | numerical gap equals `||w||` on the orthogonal unit edit |
| Align | `B_align ≤ B_dir` via `W_O w` and `W2.T w` | PROVED | Cauchy–Schwarz on the factorizations | sign-flip preserves both bounds |
| Kernel-out | Both readout gains zero ⇒ raw `R = skip` | PROVED | the two inner products vanish | recomputed gap `< 1e-15` with `||ΔA||>0.1` |
| Persist | `rate < 1/2` ⇒ `B_dir < ||∇m|| ||δ||` on `w⊥`, on an open set | PROVED | Lipschitz control of `∇m - w`, continuity of `rate` | quadratic root `α_*=1.0387929173812708e-4`; derivative gap checked |
| Product | Product-ball max equals the separate sum | PROVED | independent alignment of each ball | square gap identity |
| Sum | `J_A + J_M + skip = ∇m·δ` | PROVED | linearity of the derivative | analytic JVP versus central-difference gradient |
| Cancel | Base-point `J_A + J_M = 0` does not bound `|R|` by the linear term | COUNTEREXAMPLE_FOUND | construction of the rank-one update | step-2 response `-0.16901883579206622` |
| R3 | `|R_m - R_s| ≤ B_branch` | REDUCTION | Theorem S | zero-branch raw response `0` |
| Overnight B1 impossibility | No B1 quantity beats the Cauchy number on injective QKV | WITHDRAWN_AS_OVERBROAD | — | CE-L-scope |
| Joint Level 3 bound | Some examined `Q` with `B_joint < B_align` in Regime 1 | OPEN as a future construction; failed for R1–R3 | product-ball ceiling blocks radius-only attempts | sign-flip shows the slack is real and is seen by `∇m` |

## Downstream withdrawals

Any overnight sentence that used Theorem L to forbid directional certificates, or Theorem U to forbid fixed-weight certificates, is unsupported. The list is in `01_overnight_claim_corrections.md`. No sentence in this directory treats those withdrawn lines as premises. The historical files were not edited.
