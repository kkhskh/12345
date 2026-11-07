"""08R1: audit and repair 0A measurement infrastructure. Does not change the C/PARTIAL letter."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_r1 import (
    FakeResidualForward,
    cluster_bootstrap,
    donor_dependence,
    intervention_probability_metrics,
    paired_cluster_diff,
    softmax,
)
from src.othello_r1_eval import local_source_readiness, prefix_overlap_by_length
from src.othello_runutil import add_labels, load_split, pair_candidates, write_json
from src.toy_othello import (
    SYMMETRY_NAMES,
    BoardState,
    generate_game,
    hook_name,
    legal_squares,
    load_config,
    relative_to_absolute,
    sha256_json,
    to_absolute_labels,
    transform_board,
    transform_square,
)


RESULTS_NAME = "08r1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R1 0A measurement audit.")
    p.add_argument("--config", default="configs/toy_othello.json")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--n-confirm-games", type=int, default=400)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--boot-seed", type=int, default=81)
    p.add_argument("--substitute-train", type=int, default=800)
    p.add_argument("--substitute-test", type=int, default=200)
    return p.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _nonzero(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty deliverable: {path}")


def reconstruct_pairs(test: list[dict], n_pairs: int, seed: int) -> list[dict]:
    pairs = pair_candidates(test, n_pairs, seed)
    out = []
    for p in pairs:
        r = test[p["recip"]]
        d = test[p["donor"]]
        mets = intervention_probability_metrics(
            np.full(61, 1.0 / 61.0),
            np.full(61, 1.0 / 61.0),
            r["legal"],
            d["legal"],
        )
        out.append(
            {
                "recip_game": r["game_id"],
                "donor_game": d["game_id"],
                "recip_t": r["t"],
                "donor_t": d["t"],
                "type": p["type"],
                "hamming": p["hamming"],
                "recip_player": int(r["player"]),
                "donor_player": int(d["player"]),
                "n_became_legal": mets["n_became_legal"],
                "n_became_illegal": mets["n_became_illegal"],
                "n_stayed_legal": mets["n_stayed_legal"],
                "n_stayed_illegal": mets["n_stayed_illegal"],
                "prob_effects": "NOT_EVALUATED_no_saved_logits",
            }
        )
    return out


def hook_verification_rows() -> list[dict]:
    model = FakeResidualForward()
    mid_name = hook_name("resid_mid", 0)
    post_name = hook_name("resid_post", 0)
    mid = model.cache(mid_name)
    post = model.cache(post_name)
    donor_mid = mid.copy()
    donor_post = post.copy()
    donor_mid[:, 0] += 2.5
    donor_post[:, 1] += 3.5
    noop = model.run_patched(post_name, post)
    mid_patch = model.run_patched(mid_name, donor_mid)
    post_patch = model.run_patched(post_name, donor_post)
    donor_logits = model.logits_from_post(donor_post)
    full = model.run_patched(post_name, donor_post)
    rows = [
        {
            "check": "hook_names_distinct",
            "hook": f"{mid_name}|{post_name}",
            "ok": int(mid_name != post_name),
            "mid_post_l2": float(np.linalg.norm(mid - post)),
            "logit_max_abs_diff": float("nan"),
            "pert_norm": float("nan"),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "SYNTHETIC_UNIT_TEST. 08e iterates these exact templates.",
        },
        {
            "check": "cached_mid_post_tensors_differ",
            "hook": mid_name,
            "ok": int(not np.allclose(mid, post)),
            "mid_post_l2": float(np.linalg.norm(mid - post)),
            "logit_max_abs_diff": float("nan"),
            "pert_norm": float("nan"),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "SYNTHETIC_UNIT_TEST fake residual. Not a scientific Othello-GPT result.",
        },
        {
            "check": "noop_final_resid_reproduces_logits",
            "hook": post_name,
            "ok": int(np.allclose(noop["logits_after"], noop["logits_before"], atol=1e-10)),
            "mid_post_l2": 0.0,
            "logit_max_abs_diff": float(noop["logit_max_abs_diff"]),
            "pert_norm": float(noop["pert_norm_post"]),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "SYNTHETIC_UNIT_TEST final-norm analogue. Not real LN.",
        },
        {
            "check": "full_post_patch_reproduces_donor_logits",
            "hook": post_name,
            "ok": int(np.allclose(full["logits_after"], donor_logits, atol=1e-10)),
            "mid_post_l2": float("nan"),
            "logit_max_abs_diff": float(np.max(np.abs(full["logits_after"] - donor_logits))),
            "pert_norm": float(full["pert_norm_post"]),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "SYNTHETIC_UNIT_TEST. Not frozen Othello-GPT.",
        },
        {
            "check": "mid_vs_post_patch_distinguishable",
            "hook": f"{mid_name}|{post_name}",
            "ok": int(not np.allclose(mid_patch["logits_after"], post_patch["logits_after"])),
            "mid_post_l2": float(np.linalg.norm(mid_patch["patched_post"] - post_patch["patched_post"])),
            "logit_max_abs_diff": float(np.max(np.abs(mid_patch["logits_after"] - post_patch["logits_after"]))),
            "pert_norm": float(mid_patch["pert_norm_mid"]),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "SYNTHETIC_UNIT_TEST. Does not explain or confirm the 08e printed identity.",
        },
        {
            "check": "row_labels_match_edited_sites",
            "hook": mid_patch["hook"],
            "ok": int(mid_patch["hook"] == mid_name and post_patch["hook"] == post_name),
            "mid_post_l2": float("nan"),
            "logit_max_abs_diff": float("nan"),
            "pert_norm": float("nan"),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "08e writes r['hook']=hook_name(site, layer) from the same helper",
        },
        {
            "check": "othello_gpt_mid_post_identity",
            "hook": "blocks.{L}.hook_resid_mid vs resid_post",
            "ok": 0,
            "mid_post_l2": float("nan"),
            "logit_max_abs_diff": float("nan"),
            "pert_norm": float("nan"),
            "validation_kind": "PENDING_REAL_MODEL_RUN",
            "note": "PENDING_REAL_MODEL_RUN: run run_08r1_real_model_verification.py on Colab. Do not treat 08e 3-decimal equality as a finding.",
        },
    ]
    _ = softmax
    return rows


def consistency_rows(test: list[dict]) -> list[dict]:
    rows = []
    # Player-relative coordinate invertibility on true boards (not the probe).
    agree = []
    for r in test:
        rel = r["rel_lab"]
        back = relative_to_absolute(rel, int(r["player"]))
        agree.append(float((back == r["abs_lab"]).mean()))
    rows.append(
        {
            "test": "player_relative_coordinate_roundtrip",
            "status": "IMPLEMENTED_AND_VERIFIED",
            "mean": float(np.mean(agree)),
            "n": len(agree),
            "note": "True labels only. Tests the coordinate map, not the 08c probe.",
        }
    )

    # Predicted vs true legal Jaccard requires a saved probe.
    rows.append(
        {
            "test": "decoded_vs_true_legal_jaccard",
            "status": "NOT_EVALUATED",
            "mean": float("nan"),
            "n": 0,
            "note": "Needs 08c_selected_probe.pkl plus model cache. Pickle is not in this workspace.",
        }
    )

    trans = []
    by_game = defaultdict(list)
    for i, r in enumerate(test):
        by_game[r["game_id"]].append(i)
    for items in by_game.values():
        items = sorted(items, key=lambda i: test[i]["t"])
        for a, b in zip(items, items[1:]):
            if test[b]["t"] != test[a]["t"] + 1:
                continue
            if test[a]["after_board"] is None:
                continue
            st = BoardState(test[a]["board_np"].copy(), int(test[a]["player"]))
            move = test[a]["next_square"]
            if move is None or move not in st.legal_moves():
                continue
            stepped = st.play(int(move)).board
            trans.append(float((to_absolute_labels(stepped) == to_absolute_labels(np.asarray(test[b]["board"]).reshape(8, 8))).mean()))
    rows.append(
        {
            "test": "simulator_transition_true_F_vs_next",
            "status": "IMPLEMENTED_AND_VERIFIED",
            "mean": float(np.mean(trans)) if trans else float("nan"),
            "n": len(trans),
            "note": "True boards. Probe transition consistency is NOT_EVALUATED without the pickle.",
        }
    )
    rows.append(
        {
            "test": "probe_transition_decoded_F_vs_next",
            "status": "NOT_EVALUATED",
            "mean": float("nan"),
            "n": 0,
            "note": "Requires decoded boards from 08c_selected_probe.pkl.",
        }
    )

    same = []
    keys = defaultdict(list)
    for i, r in enumerate(test):
        keys[(int(r["player"]), tuple(int(x) for x in r["board"]))].append(i)
    n_groups = 0
    for idxs in keys.values():
        if len(idxs) < 2:
            continue
        n_groups += 1
        labs = [test[i]["abs_lab"] for i in idxs]
        same.append(float((labs[0] == labs[1]).mean()))
    rows.append(
        {
            "test": "same_state_true_label_agreement",
            "status": "NOT_EVALUATED" if n_groups == 0 else "IMPLEMENTED_AND_VERIFIED",
            "mean": float(np.mean(same)) if same else float("nan"),
            "n": n_groups,
            "note": "No repeated (player, board) keys in the subsampled test prefixes."
            if n_groups == 0
            else "True labels of transpositions. Probe agreement NOT_EVALUATED.",
        }
    )

    sym_ok = []
    for r in test[:200]:
        for name in SYMMETRY_NAMES:
            tb = transform_board(r["board_np"], name)
            tlegal = {transform_square(s, name) for s in r["legal"]}
            sym_ok.append(int(tlegal == set(legal_squares(tb, int(r["player"])))))
    rows.append(
        {
            "test": "symmetry_legal_set_consistency",
            "status": "IMPLEMENTED_AND_VERIFIED",
            "mean": float(np.mean(sym_ok)) if sym_ok else float("nan"),
            "n": len(sym_ok),
            "note": "Simulator symmetries on saved test prefixes. Not a probe test.",
        }
    )

    rows.append(
        {
            "test": "legalizer_self_consistent",
            "status": "INVALID_MEASUREMENT",
            "mean": 1.0,
            "n": len(test),
            "note": "08d compared induced_legal(pred) to itself. Removed from evidence.",
        }
    )
    rows.append(
        {
            "test": "whole_board_probe_exact",
            "status": "NOT_EVALUATED",
            "mean": float("nan"),
            "n": 0,
            "note": "08c printed test_exact=0.877 at L6 resid_mid rel; not recomputed here.",
        }
    )
    return rows


def untrained_substitute(train, test, n_train: int, n_test: int) -> list[dict]:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    tr = train[:n_train]
    te = test[:n_test]
    Xtr = rng.normal(size=(len(tr), 512)).astype(np.float32)
    Xte = rng.normal(size=(len(te), 512)).astype(np.float32)
    ytr = np.stack([r["rel_lab"] for r in tr])
    yte = np.stack([r["rel_lab"] for r in te])
    pred = np.zeros_like(yte)
    for j in range(64):
        clf = LogisticRegression(C=1.0, max_iter=40)
        clf.fit(Xtr, ytr[:, j])
        pred[:, j] = clf.predict(Xte)
    exact = float((pred == yte).all(axis=1).mean())
    sq = float((pred == yte).mean())
    return [
        {
            "control": "untrained_othello_gpt",
            "status": "PENDING_REAL_MODEL_RUN",
            "exact_board": float("nan"),
            "square_acc": float("nan"),
            "n_train": 0,
            "n_test": 0,
            "validation_kind": "PENDING_REAL_MODEL_RUN",
            "note": "PENDING_REAL_MODEL_RUN. Generator bug was CPU Generator on CUDA params; fix is in code. Run run_08r1_real_model_verification.py.",
        },
        {
            "control": "iid_gaussian_512_linear_rel_substitute",
            "status": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "exact_board": exact,
            "square_acc": sq,
            "n_train": len(tr),
            "n_test": len(te),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "SYNTHETIC_UNIT_TEST noise floor. Not the untrained Othello-GPT control and not a Gate 1 result.",
        },
    ]


def bootstrap_rows(pairs: list[dict], n_boot: int, seed: int) -> list[dict]:
    rec = [p["recip_game"] for p in pairs]
    don = [p["donor_game"] for p in pairs]
    dep = donor_dependence(rec, don)
    pair_ids = [f"{r}|{d}" for r, d in zip(rec, don)]
    # Constructed clustered series: two games, unequal means. Verifies the CI machinery.
    demo_vals = np.array([0.10] * (len(pairs) // 2) + [0.02] * (len(pairs) - len(pairs) // 2))
    demo_rec = rec[: len(demo_vals)] if rec else ["g0"] * len(demo_vals)
    demo = cluster_bootstrap(demo_vals, demo_rec, n_boot, seed)
    rows = [
        {
            "method": "bootstrap_machinery_check",
            "metric": "constructed_clustered_mean",
            "cluster": "recipient_game",
            **demo,
            "n_pairs": dep["n_pairs"],
            "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
            "n_distinct_donor_games": dep["n_distinct_donor_games"],
            "donor_reuse": int(dep["donor_reuse"]),
            "status": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": dep["note"],
        }
    ]
    if rec:
        card = cluster_bootstrap(np.array([p["n_became_legal"] for p in pairs], dtype=float), rec, n_boot, seed + 1)
        card2 = cluster_bootstrap(np.array([p["n_became_legal"] for p in pairs], dtype=float), pair_ids, n_boot, seed + 2)
        rows.append(
            {
                "method": "reconstructed_08e_pairs",
                "metric": "n_became_legal",
                "cluster": "recipient_game",
                **card,
                "n_pairs": dep["n_pairs"],
                "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
                "n_distinct_donor_games": dep["n_distinct_donor_games"],
                "donor_reuse": int(dep["donor_reuse"]),
                "status": "VERIFIED_WITH_UNIT_TESTS_ONLY",
                "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
                "note": "Cardinalities from deterministic pair_candidates(test, 80, seed+17). No logits.",
            }
        )
        rows.append(
            {
                "method": "reconstructed_08e_pairs",
                "metric": "n_became_legal",
                "cluster": "recipient_and_donor_game",
                **card2,
                "n_pairs": dep["n_pairs"],
                "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
                "n_distinct_donor_games": dep["n_distinct_donor_games"],
                "donor_reuse": int(dep["donor_reuse"]),
                "status": "VERIFIED_WITH_UNIT_TESTS_ONLY",
                "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
                "note": "Secondary clustering for donor reuse.",
            }
        )
    historical = [
        ("08e_L6_resid_mid", "old_mean_donor_legal_lift", 0.096),
        ("08e_L6_resid_post", "old_mean_donor_legal_lift", 0.096),
        ("08f_full_patch", "old_mean_became_legal_dP", 0.106),
        ("08f_pca32_patch", "old_mean_became_legal_dP", 0.072),
        ("08f_donor_probe_subspace", "old_mean_became_legal_dP", 0.007),
        ("08f_linear_board_dirs", "old_mean_became_legal_dP", 0.000),
        ("08f_train_mean_square_delta", "old_mean_became_legal_dP", 0.000),
    ]
    for method, metric, point in historical:
        rows.append(
            {
                "method": method,
                "metric": metric,
                "cluster": "recipient_game",
                "n": 80,
                "n_clusters": dep.get("n_distinct_recipient_games", 0),
                "point": point,
                "boot_mean": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "n_boot": n_boot,
                "seed": seed,
                "n_pairs": dep["n_pairs"],
                "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
                "n_distinct_donor_games": dep["n_distinct_donor_games"],
                "donor_reuse": int(dep["donor_reuse"]),
                "status": "PENDING_REAL_MODEL_RUN",
                "validation_kind": "PENDING_REAL_MODEL_RUN",
                "note": "Historical printed point only. Run run_08r1_real_model_verification.py for CIs.",
            }
        )
    if rec:
        dummy_a = np.array([0.10] * len(pairs))
        dummy_b = np.array([0.07] * len(pairs))
        d = paired_cluster_diff(dummy_a, dummy_b, rec, n_boot, seed + 3)
        rows.append(
            {
                "method": "paired_diff_machinery_check",
                "metric": "constructed_a_minus_pca32_like",
                "cluster": "recipient_game",
                **d,
                "n_pairs": dep["n_pairs"],
                "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
                "n_distinct_donor_games": dep["n_distinct_donor_games"],
                "donor_reuse": int(dep["donor_reuse"]),
                "status": "VERIFIED_WITH_UNIT_TESTS_ONLY",
                "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
                "note": "Machinery only. Not an 08f comparison.",
            }
        )
    return rows


def prereg_rows() -> list[dict]:
    return [
        {
            "ORIGINAL_REQUIREMENT": "Simulator validation must pass before probing",
            "ACTUAL_IMPLEMENTATION": "run_08a.validate_simulator / 08a_simulator_validation.json",
            "SCIENTIFIC_CONSEQUENCE": "Eligible if Colab 08a passed. Local 08a md exists; JSON not in this workspace.",
            "REMAINING_WORK": "none if Colab 08a_simulator_validation.json passed=true",
            "STATUS": "IMPLEMENTED_NOT_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Unique game-level train/val/test splits",
            "ACTUAL_IMPLEMENTATION": "run_08a.split_games; prefixes_*.jsonl",
            "SCIENTIFIC_CONSEQUENCE": "Splits exist locally and are usable.",
            "REMAINING_WORK": "none",
            "STATUS": "IMPLEMENTED_AND_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Frozen Nanda synthetic checkpoint + argmax fixture",
            "ACTUAL_IMPLEMENTATION": "load_othello_gpt / verify_nanda_checkpoint",
            "SCIENTIFIC_CONSEQUENCE": "Executed on Colab. Not re-run here (no torch).",
            "REMAINING_WORK": "none for 08R1; do not retune",
            "STATUS": "IMPLEMENTED_NOT_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Val legality gate mean_legal_mass >= 0.70 frozen before test",
            "ACTUAL_IMPLEMENTATION": "run_08b.main gate_min default 0.70",
            "SCIENTIFIC_CONSEQUENCE": "08g later showed 2400/2400 legal argmax; Gate 0 almost certainly passed.",
            "REMAINING_WORK": "confirm 08b_baseline_summary.json on Drive",
            "STATUS": "IMPLEMENTED_NOT_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Linear probes on listed sites; MLP only L4/L6 resid_post",
            "ACTUAL_IMPLEMENTATION": "run_08c.main methods tuple",
            "SCIENTIFIC_CONSEQUENCE": "Decode table (64 rows) is the valid 0A readout.",
            "REMAINING_WORK": "none; do not retune",
            "STATUS": "IMPLEMENTED_AND_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "PASS_DECODE exact>=0.20 or square>=0.85 AND +0.15 vs shuffle and untrained",
            "ACTUAL_IMPLEMENTATION": "run_08c.decide_gate1 uses shuffle only; untrained crashed",
            "SCIENTIFIC_CONSEQUENCE": "Letter used an incomplete negative-control set. Rel exact 0.877 still far above chance.",
            "REMAINING_WORK": "untrained control on GPU after generator fix",
            "STATUS": "IMPLEMENTED_NOT_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Controls: shuffle, metadata, random 16D, untrained on primary site",
            "ACTUAL_IMPLEMENTATION": "_run_primary_controls + _run_untrained_control",
            "SCIENTIFIC_CONSEQUENCE": "Untrained INVALID until generator fix is executed on GPU.",
            "REMAINING_WORK": "run fixed untrained control",
            "STATUS": "IMPLEMENTED_NOT_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "08d math consistency of selected abstraction",
            "ACTUAL_IMPLEMENTATION": "run_08d; one tautological row",
            "SCIENTIFIC_CONSEQUENCE": "legalizer_self_consistent is INVALID. Other 08d numbers not in this workspace.",
            "REMAINING_WORK": "recompute probe Jaccard/transition if pickle available",
            "STATUS": "INVALID_MEASUREMENT",
        },
        {
            "ORIGINAL_REQUIREMENT": "Full interchange lift >= 0.10 vs no-op on Type B pairs",
            "ACTUAL_IMPLEMENTATION": "08e all pair types; mean_prob_on (mean not mass); 08i uses all rows or 08f full_patch",
            "SCIENTIFIC_CONSEQUENCE": "Printed +0.096/+0.106 is the old mean metric, not total mass, not Type-B-only.",
            "REMAINING_WORK": "regenerate 08r1 causal tables with logits on GPU",
            "STATUS": "INVALID_MEASUREMENT",
        },
        {
            "ORIGINAL_REQUIREMENT": "resid_mid vs resid_post reported separately",
            "ACTUAL_IMPLEMENTATION": "08e loop uses hook_name; printed lifts identical to 3 decimals",
            "SCIENTIFIC_CONSEQUENCE": "Hook helper is distinct. Real Othello identity NOT verified. Do not interpret as science.",
            "REMAINING_WORK": "GPU hook verification on L6 mid/post",
            "STATUS": "IMPLEMENTED_NOT_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "State-specific became_legal>=0.08 and became_illegal<=-0.08, decoder agree>=0.30, beat attr/steer by 0.05",
            "ACTUAL_IMPLEMENTATION": "08i.decide checks became_legal>=0.08 vs attr/steer only",
            "SCIENTIFIC_CONSEQUENCE": "Did not change C: state edits were ~0. Official letter must stay C/PARTIAL.",
            "REMAINING_WORK": "none for the letter; missing thresholds remain missing",
            "STATUS": "NOT_IMPLEMENTED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Val selects one edit method; test once",
            "ACTUAL_IMPLEMENTATION": "08f runs all methods on test",
            "SCIENTIFIC_CONSEQUENCE": "No clean confirmatory edit test. 08R2/08R3 must use the fresh holdout.",
            "REMAINING_WORK": "08R2 method selection on val only",
            "STATUS": "NOT_IMPLEMENTED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Wrong-square control",
            "ACTUAL_IMPLEMENTATION": "absent from 08f.methods",
            "SCIENTIFIC_CONSEQUENCE": "Specificity vs a wrong target square was never measured.",
            "REMAINING_WORK": "08R2",
            "STATUS": "NOT_IMPLEMENTED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Iterative probe-constrained editor (edit_steps/edit_lr)",
            "ACTUAL_IMPLEMENTATION": "08f.linear_edit is one closed-form add, norm clip 8",
            "SCIENTIFIC_CONSEQUENCE": "Config advertised an editor that was not run.",
            "REMAINING_WORK": "08R2",
            "STATUS": "NOT_IMPLEMENTED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Bootstrap by game, 95% CIs on causal metrics",
            "ACTUAL_IMPLEMENTATION": "bootstrap_by_game existed; 08e/08f never called it; no game ids in those CSVs",
            "SCIENTIFIC_CONSEQUENCE": "n=80 pairs treated as if independent. CIs missing.",
            "REMAINING_WORK": "GPU regen of per-example outcomes then 08r1 bootstrap",
            "STATUS": "NOT_IMPLEMENTED",
        },
        {
            "ORIGINAL_REQUIREMENT": "08h intervention-locked Δs_m vs Δmargin",
            "ACTUAL_IMPLEMENTATION": "08h only AUROC of s_m and margins vs true legality",
            "SCIENTIFIC_CONSEQUENCE": "0.999 AUROC is a readout, not a germ intervention test.",
            "REMAINING_WORK": "08R3",
            "STATUS": "NOT_IMPLEMENTED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Legal probability mass = sum_P",
            "ACTUAL_IMPLEMENTATION": "mean_prob_on averaged P over the set",
            "SCIENTIFIC_CONSEQUENCE": "All 08e/08f 'mass' numbers are mislabeled means. Corrected functions are in src/othello_r1.py.",
            "REMAINING_WORK": "apply to real logits on GPU; do not overwrite old CSVs",
            "STATUS": "INVALID_MEASUREMENT",
        },
        {
            "ORIGINAL_REQUIREMENT": "OBSTRUCTION_READY = NO regardless",
            "ACTUAL_IMPLEMENTATION": "08i always writes NO",
            "SCIENTIFIC_CONSEQUENCE": "Followed. Do not change.",
            "REMAINING_WORK": "none",
            "STATUS": "IMPLEMENTED_AND_VERIFIED",
        },
        {
            "ORIGINAL_REQUIREMENT": "Official 0A letter C / PARTIAL",
            "ACTUAL_IMPLEMENTATION": "08i.decide on executed 08f means",
            "SCIENTIFIC_CONSEQUENCE": "Descriptively supported by executed methods. Not silently amended.",
            "REMAINING_WORK": "none — do not change the letter",
            "STATUS": "IMPLEMENTED_AND_VERIFIED",
        },
    ]


def scientific_validation_markdown() -> str:
    rows = [
        ("Corrected mass/mean/became-legal functions", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
        ("Game-cluster bootstrap clustering", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
        ("CPU-then-copy untrained generator fill", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
        ("Hook-name templates resid_mid vs resid_post", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
        ("Fake residual no-op / mid-vs-post / donor-logit", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
        ("Simulator coordinate / transition / symmetry on saved prefixes", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
        ("Nanda checkpoint identity fixture on this machine", "PENDING_REAL_MODEL_RUN"),
        ("Real resid_mid vs resid_post tensors and firing", "PENDING_REAL_MODEL_RUN"),
        ("Real no-op last-token resid_post logit reproduction", "PENDING_REAL_MODEL_RUN"),
        ("Real final-residual donor-logit reproduction", "PENDING_REAL_MODEL_RUN"),
        ("Real patched-activation readback", "PENDING_REAL_MODEL_RUN"),
        ("Corrected probability metrics on real 08e pairs", "PENDING_REAL_MODEL_RUN"),
        ("Game-cluster CIs on real causal metrics", "PENDING_REAL_MODEL_RUN"),
        ("Untrained Othello-GPT decode control", "PENDING_REAL_MODEL_RUN"),
        ("08f corrected mass metrics if probe pickle present", "PENDING_REAL_MODEL_RUN"),
        ("IID Gaussian feature decode row", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
        ("legalizer_self_consistent as evidence", "FAILED"),
        ("Official 0A letter left C / PARTIAL (not amended)", "VERIFIED_WITH_UNIT_TESTS_ONLY"),
    ]
    lines = [
        "# 08R1 scientific validation",
        "",
        "Official 0A letter remains **C / PARTIAL**.",
        "",
        "`08R1_INCOMPLETE — REAL_MODEL_VERIFICATION_PENDING`",
        "",
        "Synthetic residual numbers are written only under `08r1_engineering_*`.",
        "They are not real-model results and must not enter a scientific effect-size summary.",
        "No fabricated numeric results are recorded for unexecuted real-model checks.",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for name, status in rows:
        lines.append(f"| {name} | `{status}` |")
    lines += [
        "",
        "Real-model artifacts, once executed, must be:",
        "`08r1_real_hook_verification.csv`, `08r1_real_activation_inspection.csv`,",
        "`08r1_real_corrected_probability_metrics.csv`, `08r1_real_bootstrap_results.csv`,",
        "`08r1_real_untrained_control.csv`, `08r1_real_model_status.json`.",
        "",
        "Those files are absent or pending until `run_08r1_real_model_verification.py` runs",
        "against the frozen Nanda checkpoint.",
        "",
    ]
    return "\n".join(lines) + "\n"


def audit_markdown(prereg: list[dict], extra: dict) -> str:
    lines = [
        "# 08R1 implementation audit — Experiment 0A",
        "",
        "Official 0A decision remains **C / PARTIAL**. This file does not amend it.",
        "",
        "Statuses: `IMPLEMENTED_AND_VERIFIED`, `IMPLEMENTED_NOT_VERIFIED`, `NOT_IMPLEMENTED`, `INVALID_MEASUREMENT`.",
        "",
        "## A. Requirement matrix",
        "",
        "| Requirement | File / function | Implemented | Executed | Artifact | Interpretable | Missing controls | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in prereg:
        lines.append(
            f"| {r['ORIGINAL_REQUIREMENT']} | {r['ACTUAL_IMPLEMENTATION']} | "
            f"{'yes' if r['STATUS'] != 'NOT_IMPLEMENTED' else 'no'} | "
            f"{'partial' if r['STATUS'] in {'IMPLEMENTED_NOT_VERIFIED', 'INVALID_MEASUREMENT'} else ('yes' if r['STATUS']=='IMPLEMENTED_AND_VERIFIED' else 'no')} | "
            f"see 08r1_prereg_compliance.csv | {r['SCIENTIFIC_CONSEQUENCE']} | {r['REMAINING_WORK']} | `{r['STATUS']}` |"
        )
    lines += [
        "",
        "## B. Hook verification",
        "",
        "08e calls `hook_name('resid_mid', L)` and `hook_name('resid_post', L)` which are distinct strings.",
        "Fake residual checks are `SYNTHETIC_UNIT_TEST` / `VERIFIED_WITH_UNIT_TESTS_ONLY`. They are not",
        "scientific verification of frozen Othello-GPT. Real mid/post tensors, no-op, donor-logit",
        "reproduction, and corrected mass CIs are `PENDING_REAL_MODEL_RUN` until",
        "`run_08r1_real_model_verification.py` executes on the Nanda checkpoint.",
        "Do not treat the Colab 08e 3-decimal mid/post identity as a finding.",
        "",
        "## C. Probability metrics",
        "",
        "Historical `mean_prob_on` = mean P(token) over a set. It was labeled mass. That is `INVALID_MEASUREMENT`.",
        "Corrected metric *functions* are `VERIFIED_WITH_UNIT_TESTS_ONLY`.",
        "Corrected metric *values on real interventions* are `PENDING_REAL_MODEL_RUN`.",
        "Historical CSVs were not overwritten.",
        "",
        "## D. 08d consistency",
        "",
        "`legalizer_self_consistent` compared `induced_legal(pred)` to itself. `INVALID_MEASUREMENT`.",
        "Probe Jaccard / probe transitions: `PENDING_REAL_MODEL_RUN` (pickle + model cache).",
        "True-label coordinate, simulator transition, and symmetry checks are local simulator tests.",
        "",
        "## E. Untrained control",
        "",
        "Cause: `torch.Generator()` is CPU-only; `p.normal_(generator=g)` on CUDA parameters raises.",
        "Fix: `apply_cpu_normal_to_tensors` draws on CPU and copies to the parameter device.",
        "That fix is `VERIFIED_WITH_UNIT_TESTS_ONLY`. The untrained Othello-GPT decode numbers are",
        "`PENDING_REAL_MODEL_RUN`. The Gaussian-feature row is `SYNTHETIC_UNIT_TEST` and is not Gate 1.",
        "",
        "## F. Bootstrap",
        "",
        "`bootstrap_by_game` clustering is `VERIFIED_WITH_UNIT_TESTS_ONLY`.",
        "Causal probability CIs are `PENDING_REAL_MODEL_RUN`.",
        f"Reconstructed 08e pairs: n={extra.get('n_pairs')} recip_games={extra.get('n_recip')} "
        f"donor_games={extra.get('n_donor')} donor_reuse={extra.get('donor_reuse')}.",
        "",
        "## G–H. Gates and holdout",
        "",
        "08i omitted became-illegal, decoder-agree, val-only edit selection, wrong-square, untrained, and germ Δs_m.",
        "The old test set is contaminated for any new editor. 08R2/08R3 confirmation games are hashed in",
        "`08r1_fresh_holdout_manifest.json` and were not used for method development.",
        "",
        "## Official letter",
        "",
        "C / PARTIAL remains descriptively supported by the methods that actually ran: decode worked;",
        "state-specific edits did not beat controls on the (misnamed) mean metric.",
        "Scientific completion of 08R1 requires the real-model runner.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_holdout(existing_games: set[tuple[int, ...]], n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    seqs = []
    guard = 0
    while len(seqs) < n:
        guard += 1
        if guard > n * 80:
            raise RuntimeError("could not sample enough confirmation games")
        sq = generate_game(rng, li_compatible=True, max_moves=60)
        key = tuple(sq)
        if len(sq) < 8 or key in existing_games or key in {tuple(s) for s in seqs}:
            continue
        seqs.append(list(map(int, sq)))
    games = [{"game_id": f"confirm_{i:05d}", "squares": sq, "sha256": sha256_json(sq)} for i, sq in enumerate(seqs)]
    return {
        "split": "08r2_08r3_confirmation",
        "n_games": len(games),
        "seed": seed,
        "generator": "src.toy_othello.generate_game li_compatible=True max_moves=60",
        "disjoint_from": ["train", "val", "test"],
        "used_in_08r1_method_development": False,
        "manifest_sha256": sha256_json([g["sha256"] for g in games]),
        "games": games,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    results_dir = ensure_dir(cfg["results_dir"])
    data_dir = Path(cfg["data_dir"])

    print("=== 08R1 load prefixes ===")
    train = add_labels(load_split(data_dir, "train"))
    val = add_labels(load_split(data_dir, "val"))
    test = add_labels(load_split(data_dir, "test"))
    games_obj = json.loads((data_dir / "games.json").read_text(encoding="utf-8"))
    existing = set()
    for split in games_obj:
        for g in games_obj[split]:
            existing.add(tuple(g["squares"]))
    print(f"  train={len(train)} val={len(val)} test={len(test)} unique_games={len(existing)}")

    print("=== 08R1 hooks ===")
    hook_rows = hook_verification_rows()
    write_csv(results_dir / "08r1_engineering_hook_verification.csv", hook_rows)

    print("=== 08R1 reconstruct 08e pairs / corrected metric cardinalities ===")
    pairs = reconstruct_pairs(test, int(cfg["n_pairs"]), int(cfg["seed"]) + 17)
    dep = donor_dependence([p["recip_game"] for p in pairs], [p["donor_game"] for p in pairs])
    metric_rows = []
    # Exact hand-constructed distribution (same as unit test) so the CSV has numeric mass/mean.
    from src.othello_r1 import mean_prob_per_move, total_legal_mass
    from src.toy_othello import square_to_token

    p = np.zeros(61, dtype=np.float64)
    p[square_to_token(0)] = 0.50
    p[square_to_token(1)] = 0.20
    p[square_to_token(2)] = 0.10
    p[0] = 0.05
    p[square_to_token(3)] = 1.0 - p.sum()
    q = p.copy()
    q[square_to_token(2)] += 0.08
    q[square_to_token(0)] -= 0.08
    hand = intervention_probability_metrics(p, q, [0, 1], [1, 2])
    metric_rows.append(
        {
            "source": "unit_hand_distribution",
            "type": "constructed",
            **hand,
            "old_mean_on_target_before": mean_prob_per_move(p, [1, 2]),
            "total_mass_on_target_before": total_legal_mass(p, [1, 2]),
            "validation_kind": "VERIFIED_WITH_UNIT_TESTS_ONLY",
            "note": "SYNTHETIC_UNIT_TEST hand distribution. Expected became_legal_d_mass=+0.08, became_illegal_d_mass=-0.08",
        }
    )
    type_counts = defaultdict(int)
    for pr in pairs:
        type_counts[pr["type"]] += 1
        metric_rows.append(
            {
                "source": "reconstructed_08e_pair",
                "type": pr["type"],
                "n_became_legal": pr["n_became_legal"],
                "n_became_illegal": pr["n_became_illegal"],
                "n_stayed_legal": pr["n_stayed_legal"],
                "n_stayed_illegal": pr["n_stayed_illegal"],
                "became_legal_d_mass": float("nan"),
                "became_illegal_d_mass": float("nan"),
                "stayed_legal_d_mass": float("nan"),
                "stayed_illegal_d_mass": float("nan"),
                "donor_legal_mass_lift": float("nan"),
                "old_mean_donor_legal_lift": float("nan"),
                "recip_game": pr["recip_game"],
                "donor_game": pr["donor_game"],
                "validation_kind": "PENDING_REAL_MODEL_RUN",
                "note": "Cardinalities only. PENDING_REAL_MODEL_RUN for probability effects.",
            }
        )
    write_csv(results_dir / "08r1_engineering_corrected_probability_metrics.csv", metric_rows)
    print(f"  pairs={len(pairs)} types={dict(type_counts)} TypeB={type_counts.get('B_legal_diff', 0)}")

    print("=== 08R1 consistency ===")
    cons = consistency_rows(test)
    write_csv(results_dir / "08r1_engineering_consistency_checks.csv", cons)
    for r in cons:
        print(f"  {r['test']} status={r['status']} mean={r['mean']} n={r['n']}")

    print("=== 08R1 untrained / substitute ===")
    untr = untrained_substitute(train, test, args.substitute_train, args.substitute_test)
    write_csv(results_dir / "08r1_engineering_untrained_control.csv", untr)
    for r in untr:
        print(f"  {r['control']} exact={r['exact_board']} sq={r['square_acc']}")

    print("=== 08R1 bootstrap ===")
    boot = bootstrap_rows(pairs, args.n_boot, args.boot_seed)
    write_csv(results_dir / "08r1_engineering_bootstrap_results.csv", boot)

    print("=== 08R1 prereg / audit / holdout ===")
    pre = prereg_rows()
    write_csv(results_dir / "08r1_prereg_compliance.csv", pre)
    extra = {
        "n_pairs": dep["n_pairs"],
        "n_recip": dep["n_distinct_recipient_games"],
        "n_donor": dep["n_distinct_donor_games"],
        "donor_reuse": dep["donor_reuse"],
    }
    _write_text(results_dir / "08r1_implementation_audit.md", audit_markdown(pre, extra))
    hold = build_holdout(existing, args.n_confirm_games, seed=8801)
    write_json(results_dir / "08r1_fresh_holdout_manifest.json", hold)
    # Sealed repo copy used by Colab checkout. Do not use this split in 08R1 method work.
    write_json(Path("configs/08r1_fresh_holdout_manifest.json"), hold)
    print(f"  confirmation games={hold['n_games']} manifest={hold['manifest_sha256'][:12]}")
    existing_seqs = []
    for split in games_obj:
        for g in games_obj[split]:
            existing_seqs.append(list(map(int, g["squares"])))
    hold_seqs = [list(map(int, g["squares"])) for g in hold["games"]]
    overlap = prefix_overlap_by_length(hold_seqs, existing_seqs, max_len=20, reference="existing_games")
    prefix_seqs = [list(map(int, r["squares"])) for r in (*train, *val, *test)]
    overlap += prefix_overlap_by_length(hold_seqs, prefix_seqs, max_len=20, reference="existing_prefixes_jsonl")
    write_csv(results_dir / "08r1_engineering_holdout_prefix_overlap.csv", overlap)
    game_l1 = [r for r in overlap if r["reference"] == "existing_games" and r["prefix_length"] == 1][0]
    game_l12 = [r for r in overlap if r["reference"] == "existing_games" and r["prefix_length"] == 12][0]
    print(f"  prefix L=1 overlap={game_l1['frac_overlap']:.3f} L=12 overlap={game_l12['frac_overlap']:.3f}")

    runner_text = Path("run_08r1_real_model_verification.py").read_text(encoding="utf-8")
    helper_text = Path("src/othello_r1_eval.py").read_text(encoding="utf-8")
    ready = local_source_readiness(runner_text, helper_text, tests_passed=False)
    write_json(results_dir / "08r1_engineering_local_readiness.json", ready)
    print(f"  local_readiness={ready['decision']}")

    required = [
        "08r1_implementation_audit.md",
        "08r1_engineering_hook_verification.csv",
        "08r1_engineering_corrected_probability_metrics.csv",
        "08r1_engineering_consistency_checks.csv",
        "08r1_engineering_untrained_control.csv",
        "08r1_engineering_bootstrap_results.csv",
        "08r1_engineering_holdout_prefix_overlap.csv",
        "08r1_prereg_compliance.csv",
        "08r1_fresh_holdout_manifest.json",
    ]
    sci = scientific_validation_markdown()
    _write_text(results_dir / "08r1_scientific_validation.md", sci)
    required.append("08r1_scientific_validation.md")
    for name in required:
        path = results_dir / name
        _nonzero(path)
        print(f"  ok {path} bytes={path.stat().st_size}")
    print("08R1 engineering artifacts written. Official 0A letter unchanged: C / PARTIAL.")
    print("08R1_INCOMPLETE — REAL_MODEL_VERIFICATION_PENDING")


if __name__ == "__main__":
    main()
