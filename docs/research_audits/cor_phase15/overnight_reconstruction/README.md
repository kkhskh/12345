# Overnight reconstruction

Mathematical reconstruction of whether a structural relation among Transformer computations, identifiable without the response, constrains the finite causal response more tightly than the constant-shift identity and Cauchy–Schwarz.

The work lives only in this directory. Phase 1 files, the step 1C packages, and the Path 1 review package were not modified. No pretrained model was run. Phase 2 was not opened.

## Verdict

`PHASE15_NECESSARY_CONDITIONS_AND_BOUNDARIES_ESTABLISHED`

No candidate survives. The strong negative is: for a non-constant edit, budget B0 admits no finite bound, and every Lipschitz certificate is at least the Cauchy–Schwarz number. A conditional kernel equality is exact on a manufactured point and adds no non-constant edit when the `Q,K,V` stack is injective.

`PHASE_2 = NO GO`. `OBSTRUCTION_READY = NO`. `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED` is unchanged.

Read `final/PHASE15_FINAL_VERDICT.md` for the gates and the forty questions.

## Run the checks

From this directory, with no `PYTHONPATH` set:

```text
python3 -m pytest tests -q -p no:cacheprovider
python3 scripts/verify_manifest.py
```

Dependencies are `requirements.txt` (`numpy`, `pytest`, `sympy`). The package does not import the rest of the repository. Pytest puts this directory on `sys.path` because it is the rootdir; that is pytest's default, not an environment variable.

## Map

| path | role |
|---|---|
| `workstream_1/finite_response.md` | Theorems F, U, L, K, J |
| `workstream_2/identification.md` | Germ attacks |
| `workstream_3/compatibility.md` | Residual taxonomy |
| `workstream_4/geometry.md` | `O`/`D` algebra, four operations |
| `workstream_5/prediction.md` | Rejected specification |
| `workstream_6/adversarial_audit.md` | Dropped assumptions |
| `final/` | Reconstruction, mechanisms, candidates, audit, novelty, predictions, coverage, verdict |
| `proofs/` | Block, bounds, obstruction algebra |
| `tests/test_reconstruction.py` | Closed form against recomputation |
| `reviews/` | Passes A–H, same environment |
| `literature/PRIMARY_NOTES.md` | Sources actually read |
| `counterexamples/catalog.md` | Witnesses |
| `logs/` | Extraction exit codes |
