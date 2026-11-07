"""Read-only 08R2 post-validation audit. Does not load Othello-GPT."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.othello_r1 import cluster_bootstrap, paired_cluster_diff
from src.othello_runutil import write_json


PRIMARY = {"A_legal_diff", "B_similar_board_legal_change"}
EDITORS = ("editor_A_fullspace", "editor_B_rowspace", "editor_C_reusable")
SPEC_METHODS = EDITORS + ("probe_direction",)
COST_METHODS = EDITORS + ("probe_direction", "random_same_norm", "full_donor_patch", "noop")
FLOORS = {"changed": 0.40, "unchanged": 0.80, "causal": 0.05}

SIGN_ONLY = (
    "valid_edit == 1 AND changed_acc >= 0.40 AND unchanged_acc >= 0.80 "
    "AND joint_metric_eligible AND dP_gain > 0 AND dP_loss < 0"
)
PREREG = (
    "valid_edit == 1 AND changed_acc >= 0.40 AND unchanged_acc >= 0.80 "
    "AND joint_metric_eligible AND dP_gain >= 0.05 AND dP_loss <= -0.05"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2 saved-data audit. No model execution.")
    p.add_argument(
        "--validation-dir",
        default="/content/drive/MyDrive/ml_research_artifacts/toy_othello/results/08r2/validation",
    )
    p.add_argument("--out-dir", default=None, help="Defaults to <validation-dir>/postvalidation")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=82)
    return p.parse_args()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _valid(df: pd.DataFrame) -> pd.Series:
    return _num(df.get("valid_edit", 0)).isin({1, 1.0})


def _state_ok(df: pd.DataFrame) -> pd.Series:
    return (
        _valid(df)
        & (_num(df.get("changed_acc")) >= FLOORS["changed"])
        & (_num(df.get("unchanged_acc")) >= FLOORS["unchanged"])
    )


def _sign_only(df: pd.DataFrame) -> pd.Series:
    elig = _num(df.get("joint_metric_eligible")).isin({1, 1.0})
    return _state_ok(df) & elig & (_num(df.get("dP_gain")) > 0) & (_num(df.get("dP_loss")) < 0)


def _prereg(df: pd.DataFrame) -> pd.Series:
    elig = _num(df.get("joint_metric_eligible")).isin({1, 1.0})
    return (
        _state_ok(df)
        & elig
        & (_num(df.get("dP_gain")) >= FLOORS["causal"])
        & (_num(df.get("dP_loss")) <= -FLOORS["causal"])
    )


def _boot(values, clusters, n_boot, seed) -> dict:
    out = cluster_bootstrap(np.asarray(values, dtype=np.float64), list(clusters), n_boot, seed)
    return {k: out[k] for k in ("n", "n_clusters", "point", "boot_mean", "ci95_low", "ci95_high")}


def load_pairs(path: Path) -> pd.DataFrame:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for p in doc.get("pairs", []):
        rows.append(
            {
                "recip": p.get("recip"),
                "donor": p.get("donor"),
                "r2_type": p.get("r2_type"),
                "recip_game": p.get("recip_game"),
                "target_game": p.get("target_game"),
                "intended_target_game": p.get("intended_target_game"),
                "hamming": p.get("hamming"),
                "n_became_legal": p.get("n_became_legal"),
                "n_became_illegal": p.get("n_became_illegal"),
            }
        )
    return pd.DataFrame(rows)


def accounting_table(causal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in EDITORS:
        for slice_name, mask in (
            ("A", causal["r2_type"] == "A_legal_diff"),
            ("B", causal["r2_type"] == "B_similar_board_legal_change"),
            ("D", causal["r2_type"] == "D_wrong_target"),
            ("primary_AB", causal["r2_type"].isin(PRIMARY)),
            ("all_including_D", causal["r2_type"].isin(PRIMARY | {"D_wrong_target"})),
        ):
            sub = causal[(causal["method"] == method) & mask]
            for budget, g in list(sub.groupby("budget", dropna=False)) + [("all_budgets", sub)]:
                if isinstance(budget, str) and budget == "all_budgets":
                    gg = g
                    bval = "all"
                else:
                    gg = g
                    bval = float(budget)
                rows.append(
                    {
                        "method": method,
                        "slice": slice_name,
                        "budget": bval,
                        "attempted": int(len(gg)),
                        "n_unique_recip_target": int(gg[["recip_game", "target_game"]].drop_duplicates().shape[0]) if len(gg) else 0,
                        "n_unique_recipient_games": int(gg["recip_game"].nunique()) if len(gg) else 0,
                        "n_unique_target_games": int(gg["target_game"].nunique()) if len(gg) else 0,
                        "valid_executions": int(_valid(gg).sum()),
                        "sign_only_successes": int(_sign_only(gg).sum()),
                        "prereg_joint_successes": int(_prereg(gg).sum()),
                        "state_target_successes": int(_state_ok(gg).sum()),
                        "mean_dP_gain": float(np.nanmean(_num(gg.get("dP_gain")))) if len(gg) else float("nan"),
                        "mean_dP_loss": float(np.nanmean(_num(gg.get("dP_loss")))) if len(gg) else float("nan"),
                        "mean_S_legal": float(np.nanmean(_num(gg.get("S_legal")))) if len(gg) else float("nan"),
                        "mean_changed_acc": float(np.nanmean(_num(gg.get("changed_acc")))) if len(gg) else float("nan"),
                        "mean_board_exact": float(np.nanmean(_num(gg.get("board_exact")))) if len(gg) else float("nan"),
                        "mean_l2": float(np.nanmean(_num(gg.get("delta_norm")))) if len(gg) else float("nan"),
                        "mean_kl": float(np.nanmean(_num(gg.get("kl")))) if len(gg) else float("nan"),
                        "sign_only_predicate": SIGN_ONLY,
                        "prereg_predicate": PREREG,
                        "attempts_construction": "64 pairs (24A+8B+32D) x 3 budgets = 192 per editor",
                    }
                )
    return pd.DataFrame(rows)


def wrong_target_table(causal: pd.DataFrame, pairs: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict]:
    d_manifest = pairs[pairs["r2_type"] == "D_wrong_target"].copy()
    missing = {
        "edited_output_probs_pe": True,
        "base_output_probs_pb": True,
        "intended_target_legal_move_list": True,
        "predicted_board_labels": True,
        "intended_target_rel_lab": True,
        "causal_row_intended_target_game": "intended_target_game" not in causal.columns,
    }
    note = (
        "D editor rows were scored against the wrong-target legal set and wrong-target board. "
        "The CSV does not store pe/pb or legal-move lists, so ΔP cannot be recomputed "
        "against the intended target. This table matches same recipient+method+budget and "
        "reports each row's OWN-target scores. That is not the preregistered intended-legal-set contrast."
    )
    rows = []
    flags = []
    for method in SPEC_METHODS:
        for budget, _ in causal[causal["method"] == method].groupby("budget"):
            prim = causal[(causal["method"] == method) & (causal["r2_type"].isin(PRIMARY)) & (_num(causal["budget"]) == float(budget))]
            wrong = causal[(causal["method"] == method) & (causal["r2_type"] == "D_wrong_target") & (_num(causal["budget"]) == float(budget))]
            prim = prim[_valid(prim)]
            wrong = wrong[_valid(wrong)]
            matched = 0
            unmatched_p = 0
            diffs = defaultdict(list)
            clusters = []
            for _, pr in prim.iterrows():
                wr = wrong[wrong["recip_game"] == pr["recip_game"]]
                if len(wr) != 1:
                    unmatched_p += 1
                    flags.append(
                        {
                            "method": method,
                            "budget": float(budget),
                            "recip_game": pr["recip_game"],
                            "status": "UNMATCHED_PRIMARY",
                            "n_d_rows": int(len(wr)),
                        }
                    )
                    continue
                w = wr.iloc[0]
                man = d_manifest[d_manifest["recip_game"] == pr["recip_game"]]
                intended = man.iloc[0]["intended_target_game"] if len(man) else None
                matched += 1
                rec = {
                    "method": method,
                    "budget": float(budget),
                    "r2_type_primary": pr["r2_type"],
                    "recip_game": pr["recip_game"],
                    "correct_target_game": pr["target_game"],
                    "wrong_target_game": w["target_game"],
                    "intended_target_game": intended,
                    "intended_matches_correct": int(intended == pr["target_game"]) if intended is not None else None,
                    "scored_against_intended_legal_set": 0,
                    "correct_dP_gain_own_target": float(pr["dP_gain"]) if pd.notna(pr.get("dP_gain")) else float("nan"),
                    "correct_dP_loss_own_target": float(pr["dP_loss"]) if pd.notna(pr.get("dP_loss")) else float("nan"),
                    "wrong_dP_gain_own_target": float(w["dP_gain"]) if pd.notna(w.get("dP_gain")) else float("nan"),
                    "wrong_dP_loss_own_target": float(w["dP_loss"]) if pd.notna(w.get("dP_loss")) else float("nan"),
                    "diff_dP_gain_own_targets": float(pr["dP_gain"] - w["dP_gain"]) if pd.notna(pr.get("dP_gain")) and pd.notna(w.get("dP_gain")) else float("nan"),
                    "diff_S_legal_own_targets": float(pr["S_legal"] - w["S_legal"]) if pd.notna(pr.get("S_legal")) and pd.notna(w.get("S_legal")) else float("nan"),
                    "correct_kl": float(pr["kl"]) if pd.notna(pr.get("kl")) else float("nan"),
                    "wrong_kl": float(w["kl"]) if pd.notna(w.get("kl")) else float("nan"),
                    "correct_l2": float(pr["delta_norm"]) if pd.notna(pr.get("delta_norm")) else float("nan"),
                    "wrong_l2": float(w["delta_norm"]) if pd.notna(w.get("delta_norm")) else float("nan"),
                    "correct_changed_acc_own_target": float(pr["changed_acc"]) if pd.notna(pr.get("changed_acc")) else float("nan"),
                    "wrong_changed_acc_own_target": float(w["changed_acc"]) if pd.notna(w.get("changed_acc")) else float("nan"),
                    "correct_unchanged_acc_own_target": float(pr["unchanged_acc"]) if pd.notna(pr.get("unchanged_acc")) else float("nan"),
                    "wrong_unchanged_acc_own_target": float(w["unchanged_acc"]) if pd.notna(w.get("unchanged_acc")) else float("nan"),
                    "limitation": "own-target scoring only; intended-legal-set contrast NOT_COMPUTABLE",
                }
                rows.append(rec)
                for key in ("diff_dP_gain_own_targets", "diff_S_legal_own_targets"):
                    diffs[key].append(rec[key])
                clusters.append(pr["recip_game"])
            summary = {
                "method": method,
                "budget": float(budget),
                "matched_pairs": matched,
                "unmatched_primary": unmatched_p,
                "scored_against_intended_legal_set": 0,
            }
            if matched:
                for key, vals in diffs.items():
                    b = _boot(vals, clusters, n_boot, seed)
                    summary[f"{key}_point"] = b["point"]
                    summary[f"{key}_ci95_low"] = b["ci95_low"]
                    summary[f"{key}_ci95_high"] = b["ci95_high"]
                    summary[f"{key}_n_clusters"] = b["n_clusters"]
            flags.append(summary)
    meta = {
        "requested_estimand": "correct and wrong edits scored on the SAME intended legal-move set",
        "available_estimand": "correct edit vs its intended target; wrong edit vs its wrong target",
        "missing_fields_for_requested_estimand": missing,
        "note": note,
        "unmatched_or_batch_flags": flags,
    }
    return pd.DataFrame(rows), meta


def ci_table(causal: pd.DataFrame, stored: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    if len(stored):
        for _, r in stored.iterrows():
            rec = r.to_dict()
            rec["ci_source"] = "stored_validation_bootstrap"
            rec["slice"] = "pooled_AB_unless_method_label_says_otherwise"
            rows.append(rec)
    metrics = ["dP_gain", "dP_loss", "S_legal", "changed_acc", "kl", "delta_norm"]
    for method in COST_METHODS:
        for slice_name, types in (("A", {"A_legal_diff"}), ("B", {"B_similar_board_legal_change"}), ("pooled_AB", PRIMARY)):
            sub = causal[(causal["method"] == method) & (causal["r2_type"].isin(types)) & _valid(causal)]
            for budget, g in sub.groupby("budget"):
                for metric in metrics:
                    gg = g[pd.notna(_num(g.get(metric)))]
                    if metric == "S_legal":
                        gg = gg[_num(gg.get("joint_metric_eligible")).isin({1, 1.0})]
                    if gg.empty:
                        continue
                    b = _boot(_num(gg[metric]).tolist(), gg["recip_game"].tolist(), n_boot, seed)
                    tb = _boot(_num(gg[metric]).tolist(), (gg["recip_game"].astype(str) + "|" + gg["target_game"].astype(str)).tolist(), n_boot, seed + 1)
                    rows.append(
                        {
                            "method": method,
                            "budget": float(budget),
                            "slice": slice_name,
                            "metric": metric,
                            "ci_source": "recomputed_from_saved_pair_rows",
                            "cluster": "recipient_game",
                            **b,
                        }
                    )
                    rows.append(
                        {
                            "method": method,
                            "budget": float(budget),
                            "slice": slice_name,
                            "metric": metric,
                            "ci_source": "recomputed_from_saved_pair_rows",
                            "cluster": "recipient_pipe_target_sensitivity",
                            **tb,
                        }
                    )
    return pd.DataFrame(rows)


def cost_table(causal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in COST_METHODS:
        sub = causal[(causal["method"] == method) & (causal["r2_type"].isin(PRIMARY)) & _valid(causal)]
        for (rtype, budget), g in sub.groupby(["r2_type", "budget"]):
            rows.append(
                {
                    "method": method,
                    "r2_type": rtype,
                    "nominal_budget": float(budget),
                    "n": int(len(g)),
                    "n_recipient_games": int(g["recip_game"].nunique()),
                    "n_target_games": int(g["target_game"].nunique()),
                    "mean_achieved_l2": float(np.nanmean(_num(g.get("delta_norm")))),
                    "mean_kl": float(np.nanmean(_num(g.get("kl")))),
                    "mean_dP_gain": float(np.nanmean(_num(g.get("dP_gain")))),
                    "mean_dP_loss": float(np.nanmean(_num(g.get("dP_loss")))),
                    "mean_S_legal": float(np.nanmean(_num(g.get("S_legal")))),
                    "mean_changed_acc": float(np.nanmean(_num(g.get("changed_acc")))),
                    "mean_board_exact": float(np.nanmean(_num(g.get("board_exact")))),
                    "used_donor_activation": int(_num(g.get("used_donor_activation")).fillna(0).max()),
                    "comparison_class": (
                        "donor_access"
                        if method in {"full_donor_patch"}
                        else "same_nominal_budget_not_same_achieved_l2"
                    ),
                    "note": "A/B often use less than the nominal 16/32 budget. Full donor L2 is not budget-matched.",
                }
            )
    return pd.DataFrame(rows)


def pair_type_numeric(pairs: pd.DataFrame) -> pd.DataFrame:
    num = pairs.copy()
    for col in ("hamming", "n_became_legal", "n_became_illegal"):
        if col in num.columns:
            num[col] = _num(num[col])
    rows = []
    for r2, g in num.groupby("r2_type"):
        rows.append(
            {
                "r2_type": r2,
                "n_pairs": int(len(g)),
                "n_unique_recip_games": int(g["recip_game"].nunique()),
                "n_unique_target_games": int(g["target_game"].nunique()),
                "hamming_mean": float(g["hamming"].mean()) if "hamming" in g else float("nan"),
                "hamming_min": float(g["hamming"].min()) if "hamming" in g else float("nan"),
                "hamming_max": float(g["hamming"].max()) if "hamming" in g else float("nan"),
                "n_became_legal_mean": float(g["n_became_legal"].mean()) if "n_became_legal" in g else float("nan"),
                "n_became_illegal_mean": float(g["n_became_illegal"].mean()) if "n_became_illegal" in g else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_interpretation(path: Path, acc: pd.DataFrame, cost: pd.DataFrame, wt_meta: dict, pair_sum: pd.DataFrame) -> None:
    def cell(method, slice_, budget, field):
        hit = acc[(acc["method"] == method) & (acc["slice"] == slice_) & (acc["budget"].astype(str) == str(budget))]
        if hit.empty:
            return None
        return hit.iloc[0][field]

    lines = [
        "# 08R2 Post-validation Scientific Interpretation",
        "",
        "Read-only audit of saved validation artifacts. No model rerun. No threshold change.",
        "",
        "Official labels unchanged:",
        "",
        "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
        "- Original 0A: **C / PARTIAL**",
        "- Section 5.3: **PARTIAL**",
        "- `OBSTRUCTION_READY = NO`",
        "",
        "## Success predicates",
        "",
        f"- Sign-only joint success: `{SIGN_ONLY}`",
        f"- Preregistered joint success: `{PREREG}`",
        "",
        "Sign-only is not a preregistered success.",
        "",
        "## How 192 attempts are built",
        "",
        "Each editor is applied to every pair in the frozen ABD sample, at each of three budgets.",
        "64 pairs (24 A + 8 B + 32 D) × 3 budgets = **192** attempted edits per editor.",
        "Primary experimental units are the 32 A/B recipient games. D rows are paired controls on those same recipients.",
        "The original accounting JSON mixed A, B, and D. This audit splits them.",
        "",
        "## Pair counts (numeric fields only; game IDs counted with nunique)",
        "",
        pair_sum.to_string(index=False),
        "",
        "## Wrong-target limitation",
        "",
        wt_meta["note"],
        "",
        f"Missing for the requested estimand: `{json.dumps(wt_meta['missing_fields_for_requested_estimand'])}`",
        "",
        "## Claim-by-claim",
        "",
        "### A. The linear probe decodes board state accurately.",
        "",
        "**SUPPORTED.** Frozen 08c probe, torch–sklearn parity passed, and full-donor residuals on these pairs give changed-square 1.0 and board-exact 1.0.",
        "",
        "### B. Editor A can induce exact target-board decoding.",
        "",
        "**SUPPORTED** on primary pairs at budgets 16 and 32 (changed-square 1.0, board-exact 1.0). Budget 8 is only partial. This is a representation finding, not a causal finding.",
        "",
        "### C. Editor A's induced board is behaviorally effective.",
        "",
        "**NOT_SUPPORTED.** On A pairs at 16/32, mean ΔP_gain ≈ 0.019 and on B pairs ≈ 0.002, below the 0.05 floor. Full donor on the same pairs has ΔP_gain typically 0.3–1.0. The site can move behavior; the decoder-optimal δ does not.",
        "",
        "### D. Probe-direction edits can produce causal legal-move changes.",
        "",
        "**SUPPORTED** at budgets 16 and 32. On A pairs, probe-direction mean ΔP_gain ≈ 0.34 at L2 16 and ≈ 0.49 at L2 ≈ 20.5. Do not call probe directions causally ineffective.",
        "",
        "### E. Editor C's changes are specific to the intended board.",
        "",
        "**INCONCLUSIVE** for the requested estimand. Saved D rows are scored against the wrong target's legal set, and pe/pb were not stored, so correct-minus-wrong ΔP on the intended legal set cannot be computed. Own-target D scores are in `08r2_postvalidation_wrong_target.csv` and must not be treated as the intended-legal-set contrast.",
        "",
        "### F. A reusable, causally validated mathematical state abstraction has been demonstrated.",
        "",
        "**NOT_SUPPORTED.** No editor passed representation and the 0.05 behavioral floor together. Editor C moves legal-set mass without decoding the target board. That is not a validated state editor.",
        "",
        "## What the negative experiment establishes",
        "",
        "1. A linear residual probe can be satisfied exactly without producing a preregistered legal-set shift.",
        "2. The same site still supports a large full-donor effect, so the failure is the editor, not a dead site or empty legal-set difference.",
        "3. A larger, unregularized probe-direction shove can move legal-set mass; the regularized decoder-optimal edit does not.",
        "",
        "## What it does not establish",
        "",
        "1. It does not show that probe directions are useless.",
        "2. It does not show that Editor C is a better state editor.",
        "3. It does not establish or refute same-legal invariance (no category C pairs).",
        "4. It does not confirm or refute Section 5.3. Confirmation remains sealed.",
        "5. It does not disprove COR. One negative local-edit test is not a theory refutation.",
        "6. Recipient-game bootstrap with only six target games understates target-sharing dependence if only recipient clusters are used. See `recipient_pipe_target_sensitivity` rows.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    val_dir = Path(args.validation_dir)
    out_dir = Path(args.out_dir) if args.out_dir else val_dir / "postvalidation"
    out_dir.mkdir(parents=True, exist_ok=True)
    causal = pd.read_csv(val_dir / "08r2_validation_causal_effects.csv")
    stored_boot = pd.read_csv(val_dir / "08r2_validation_bootstrap.csv") if (val_dir / "08r2_validation_bootstrap.csv").is_file() else pd.DataFrame()
    pairs = load_pairs(val_dir / "08r2_validation_pairs.json")

    acc = accounting_table(causal)
    wt, wt_meta = wrong_target_table(causal, pairs, args.n_boot, args.boot_seed)
    ci = ci_table(causal, stored_boot, args.n_boot, args.boot_seed)
    cost = cost_table(causal)
    pair_sum = pair_type_numeric(pairs)

    acc_path = out_dir / "08r2_postvalidation_accounting_audit.csv"
    wt_path = out_dir / "08r2_postvalidation_wrong_target.csv"
    ci_path = out_dir / "08r2_postvalidation_ci_summary.csv"
    cost_path = out_dir / "08r2_postvalidation_cost_comparison.csv"
    md_path = out_dir / "08r2_postvalidation_scientific_interpretation.md"
    acc.to_csv(acc_path, index=False)
    wt.to_csv(wt_path, index=False)
    ci.to_csv(ci_path, index=False)
    cost.to_csv(cost_path, index=False)
    write_interpretation(md_path, acc, cost, wt_meta, pair_sum)
    write_json(out_dir / "08r2_postvalidation_wrong_target_meta.json", wt_meta)
    write_json(
        out_dir / "08r2_postvalidation_manifest.json",
        {
            "status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
            "official_0A_letter": "C / PARTIAL",
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
            "confirmation_evaluated": 0,
            "validation_dir": str(val_dir),
            "out_dir": str(out_dir),
            "row_counts": {
                "accounting": int(len(acc)),
                "wrong_target": int(len(wt)),
                "ci": int(len(ci)),
                "cost": int(len(cost)),
                "pair_types": int(len(pair_sum)),
            },
            "predicates": {"sign_only": SIGN_ONLY, "prereg": PREREG},
            "overwrote_historical": 0,
        },
    )
    print("08R2 postvalidation audit written")
    print(f"  out_dir={out_dir}")
    print(f"  accounting_rows={len(acc)}")
    print(f"  wrong_target_rows={len(wt)}")
    print(f"  ci_rows={len(ci)}")
    print(f"  cost_rows={len(cost)}")
    print("  official letter unchanged: C / PARTIAL")
    print("  08R2_VALIDATION_COMPLETE — NEGATIVE")


if __name__ == "__main__":
    main()
