# Data

Everything under `data/` was copied byte for byte from original outputs, which were opened read-only. `data/provenance_manifest.json` records the source of every file and its SHA-256. `audits/provenance_inventory.md` is a readable rendering of that manifest. The analysis code checks the hashes of `data/raw/` before using any file and stops if one differs.

## `data/raw/` — saved arrays (evidence level RECOMPUTED)

Each subdirectory holds the non-smoke members of one result archive from the original Colab runs:

| Directory | Study | Main arrays |
|---|---|---|
| `data/raw/fiber/` | FIBER diagnostic, 32 pairs | full next-move logits, 64×3 probe probabilities, SVD report of the probe contrast matrix |
| `data/raw/spec/` | SPEC, matched L2, 48 pairs | logits and probe probabilities for every cell, nullspace directions, target legal sets, pair manifest |
| `data/raw/fixed_js/` | matched JS divergence on the SPEC pairs, 336 interventions | logits, probe probabilities, calibration and dose-search records, pair manifest |
| `data/raw/nst/` | NST, donor-free map, 48 new pairs | logits, probe probabilities, learned and comparison directions, the frozen PCA/ridge map, symbolic feature names, legal sets, split and leakage records |

The cell names in these files follow the original code. In SPEC, `baseline` is the natural recipient activation \(h_R\) and `editor_A` is the Editor A reference activation \(h_E\). In NST, `unedited` is \(h_R\) and `editor_A` is \(h_E\). Every other cell is a patched activation \(h_E + t\,\delta\).

## `data/protocols/` — preregistrations and configurations (evidence level PROTOCOL)

Copied from the project repository with their git blob identifiers. Thresholds quoted in the paper, such as effect floors, coverage gates and bootstrap settings, are checked against these files by `scripts/verify_paper_claims.py`.

## `data/archival/` — console excerpts (evidence level ARCHIVAL)

The feature-ablation (SEM), switching (FRESH) and factorial (VNF) studies left no result arrays that could be found. `data/archival/console_excerpts/` holds verbatim console output from the original runs. `data/archival/archival_values.json` lists each archival number used in the paper, the excerpt it comes from, and a quoted string that the verifier finds verbatim in that excerpt. These numbers are reported, not recomputed.

## Not included

- Model checkpoint. All runs used `synthetic_model.pth` from the Hugging Face repository `NeelNanda/Othello-GPT-Transformer-Lens`, a redistribution of the Li et al. (2023) research checkpoint. Its SHA-256, as recorded by every run, is `aef97afb1e7ec9dbef1d9007174458830a5c428d62f1e2338128879b179bd5c4`. Its license for redistribution in a paper package has not been verified, so it is not copied here. An authorized researcher can download it from that repository and compare the hash.
- Probe weights. The frozen probe is `08c_selected_probe.pkl` from the project's artifact store, with recorded SHA-256 `d587d61d7cba059ecc16059679120c96c7b660ad9fdf00aa18bab1806f3a2a31`. It is available from the authors on request. The package does not need it, because the probe probabilities of every intervention and the singular values of its contrast matrix are saved.
- The sealed confirmation split and its protocol. They were not opened for this paper.
- The board-edit validation pair tables and the SEM, FRESH and VNF arrays. `data/provenance_manifest.json` lists where they were searched for.

Replaying any intervention requires the checkpoint, the probe and the project repository. Nothing in this package's pipeline needs them.
