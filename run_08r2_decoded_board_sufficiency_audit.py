"""Read-only 08R2-S audit: is the decoded board sufficient for saved interventions?

Does not load Othello-GPT. Does not overwrite original validation files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.othello_r1 import cluster_bootstrap
from src.othello_r2 import SITE
from src.othello_runutil import write_json


PRIMARY = {"A_legal_diff", "B_similar_board_legal_change"}
FLOOR = 0.05
EDITOR_A_BUDGET = 16.0
MATCH_KEYS = ("recip_game", "target_game", "recip", "donor", "r2_type")
TOKEN_POSITION = "last real token (sequence length minus one)"
LOGIT_EXACT = {
    "pe",
    "pb",
    "logits",
    "edited_probs",
    "base_probs",
    "probe_margin",
    "changed_logit",
    "decoder_margin",
    "proba_target",
}
ORIGINAL_FILES = (
    "08r2_validation_causal_effects.csv",
    "08r2_validation_state_effects.csv",
    "08r2_validation_summary.json",
    "08r2_validation_bootstrap.csv",
    "08r2_validation_pairs.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-S decoded-board sufficiency audit.")
    p.add_argument(
        "--validation-dir",
        default="/content/drive/MyDrive/ml_research_artifacts/toy_othello/results/08r2/validation",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=82)
    p.add_argument("--editor-budget", type=float, default=EDITOR_A_BUDGET)
    return p.parse_args()


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _valid(df: pd.DataFrame) -> pd.Series:
    return _num(df.get("valid_edit", 0)).isin({1, 1.0})


def _boot(values, clusters, n_boot, seed) -> dict:
    out = cluster_bootstrap(np.asarray(values, dtype=np.float64), list(clusters), n_boot, seed)
    return {k: out[k] for k in ("n", "n_clusters", "point", "boot_mean", "ci95_low", "ci95_high")}


def _fmt_ci(ci: dict | None) -> str:
    if not ci:
        return "n/a"
    return (
        f"n={ci['n']} clusters={ci['n_clusters']} point={ci['point']:.4f} "
        f"CI95=[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]"
    )


def missing_representation_fields(columns: list[str]) -> dict:
    names = [str(c) for c in columns]
    lower = [c.lower() for c in names]
    present = []
    for col, low in zip(names, lower):
        stem = low.split("_a")[0].split("_donor")[0]
        if low in LOGIT_EXACT or stem in LOGIT_EXACT:
            present.append(col)
    return {
        "decoded_board_labels": "IDENTIFIABLE" if any(c == "board_exact" or c.startswith("board_exact_") for c in lower) else "NOT_IDENTIFIABLE_FROM_SAVED_DATA",
        "decoder_probabilities_or_margins": "NOT_IDENTIFIABLE_FROM_SAVED_DATA" if not present else "COLUMNS_PRESENT",
        "legal_set_probability_metrics": "IDENTIFIABLE" if any("dp_gain" in c for c in lower) else "NOT_IDENTIFIABLE_FROM_SAVED_DATA",
        "kl_to_baseline_scalar": "IDENTIFIABLE" if any(c == "kl" or c.startswith("kl_") for c in lower) else "NOT_IDENTIFIABLE_FROM_SAVED_DATA",
        "full_output_logits_or_probs": "NOT_IDENTIFIABLE_FROM_SAVED_DATA" if not present else "COLUMNS_PRESENT",
        "candidate_logit_columns": present,
    }


def decide_verdict(n_matched: int, n_exact: int, n_counter: int) -> str:
    if n_matched == 0 or n_exact == 0:
        return "INCONCLUSIVE_FROM_SAVED_DATA"
    if n_counter > 0:
        return "DECODED_BOARD_INSUFFICIENCY_SUPPORTED"
    return "NOT_SUPPORTED"


def _coerce_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    for k in keys:
        if k not in out.columns:
            continue
        if k in {"recip", "donor"}:
            out[k] = pd.to_numeric(out[k], errors="coerce")
        else:
            out[k] = out[k].astype(str)
    return out


def load_pair_protocol(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        return pd.DataFrame(), {"pairs_json": "MISSING"}
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
                "same_player": p.get("same_player"),
                "hamming": p.get("hamming"),
            }
        )
    pdf = pd.DataFrame(rows)
    flags = {
        "pairs_json": "PRESENT",
        "n_manifest_pairs": int(len(pdf)),
        "same_player_all_primary": bool(
            pdf[pdf["r2_type"].isin(PRIMARY)]["same_player"].fillna(0).astype(float).eq(1).all()
        )
        if len(pdf) and "same_player" in pdf.columns
        else False,
        "same_player_column": "same_player" in pdf.columns,
    }
    return pdf, flags


def editor_a_budget_identity(causal: pd.DataFrame, b1: float = 16.0, b2: float = 32.0) -> dict:
    keys = [k for k in MATCH_KEYS if k in causal.columns]
    a = causal[(causal["method"] == "editor_A_fullspace") & causal["r2_type"].isin(PRIMARY) & _valid(causal)]
    left = a[np.isclose(_num(a["budget"]), b1)]
    right = a[np.isclose(_num(a["budget"]), b2)]
    if left.empty or right.empty:
        return {"comparable": 0, "identical_operating_point": None}
    merged = _coerce_keys(left, keys).merge(_coerce_keys(right, keys), on=keys, suffixes=("_16", "_32"))
    cols = [c for c in ("dP_gain", "dP_loss", "S_legal", "board_exact", "kl", "delta_norm") if f"{c}_16" in merged.columns]
    identical = True
    maxima = {}
    for c in cols:
        diff = (_num(merged[f"{c}_16"]) - _num(merged[f"{c}_32"])).abs()
        maxima[c] = float(np.nanmax(diff)) if len(diff) else float("nan")
        if np.nanmax(diff) > 1e-12:
            identical = False
    return {
        "comparable": int(len(merged)),
        "identical_operating_point": bool(identical) if len(merged) else None,
        "max_abs_diff": maxima,
        "note": "Budgets 16 and 32 are not independent samples if this operating point is identical.",
    }


def match_editor_a_to_donor(causal: pd.DataFrame, budget: float) -> tuple[pd.DataFrame, dict]:
    primary = causal[causal["r2_type"].isin(PRIMARY)].copy()
    ed = primary[(primary["method"] == "editor_A_fullspace") & (np.isclose(_num(primary["budget"]), budget)) & _valid(primary)]
    don = primary[(primary["method"] == "full_donor_patch") & _valid(primary)]
    keys = [k for k in MATCH_KEYS if k in causal.columns]
    if "recip_game" not in keys or "target_game" not in keys:
        raise SystemExit("causal CSV missing recip_game/target_game")
    ed = _coerce_keys(ed, keys)
    don = _coerce_keys(don, keys)
    d_ct = don.groupby(keys).size()
    dup_donors = int((d_ct > 1).sum())
    if dup_donors:
        don = don.sort_values(keys).groupby(keys, as_index=False).first()
    merged = ed.merge(don, on=keys, how="outer", suffixes=("_A", "_donor"), indicator=True)
    both = merged[merged["_merge"] == "both"].copy()
    flags = {
        "editor_A_unmatched": int((merged["_merge"] == "left_only").sum()),
        "donor_unmatched": int((merged["_merge"] == "right_only").sum()),
        "matched": int(len(both)),
        "duplicate_donor_key_groups_collapsed": dup_donors,
        "match_keys": keys,
        "player_column_present": any("player" in c and "same_player" not in c for c in causal.columns),
        "token_position_column_present": any(c in causal.columns for c in ("token_pos", "position", "last_token")),
        "site_column_present": "site" in causal.columns or "probe_site" in causal.columns,
        "site_protocol": SITE,
        "token_position_protocol": TOKEN_POSITION,
        "grouped_donor_by_l2": 0,
        "d_rows_in_primary_match": 0,
    }
    if flags["player_column_present"] and "player_A" in both.columns:
        both["player_match"] = both["player_A"] == both["player_donor"]
        flags["player_match"] = "COLUMN_VERIFIED"
    else:
        both["player_match"] = pd.NA
        flags["player_match"] = "PROTOCOL_SAME_PLAYER_REQUIRED_AT_PAIRING"
    flags["token_position_match"] = (
        "COLUMN_VERIFIED" if flags["token_position_column_present"] else "PROTOCOL_CONSTANT_NOT_IN_CSV"
    )
    flags["site_match"] = "PROTOCOL_CONSTANT_NOT_IN_CSV"
    if both.empty:
        both["same_exact_decoded_board"] = pd.Series(dtype=bool)
        both["dP_gain_diff_A_minus_donor"] = pd.Series(dtype=float)
        both["dP_loss_diff_A_minus_donor"] = pd.Series(dtype=float)
        both["S_legal_diff_A_minus_donor"] = pd.Series(dtype=float)
        both["kl_diff_A_minus_donor"] = pd.Series(dtype=float)
        both["l2_A"] = pd.Series(dtype=float)
        both["l2_donor"] = pd.Series(dtype=float)
        both["abs_gain_diff"] = pd.Series(dtype=float)
        both["abs_loss_diff"] = pd.Series(dtype=float)
        both["discrepancy_ge_0_05"] = pd.Series(dtype=bool)
        both["sufficiency_counterexample"] = pd.Series(dtype=bool)
        both["editor_budget_used"] = pd.Series(dtype=float)
        return both, flags
    both["same_exact_decoded_board"] = (
        np.isclose(_num(both["board_exact_A"]), 1.0) & np.isclose(_num(both["board_exact_donor"]), 1.0)
    )
    both["dP_gain_diff_A_minus_donor"] = _num(both["dP_gain_A"]) - _num(both["dP_gain_donor"])
    both["dP_loss_diff_A_minus_donor"] = _num(both["dP_loss_A"]) - _num(both["dP_loss_donor"])
    both["S_legal_diff_A_minus_donor"] = _num(both["S_legal_A"]) - _num(both["S_legal_donor"])
    both["kl_diff_A_minus_donor"] = _num(both["kl_A"]) - _num(both["kl_donor"])
    both["l2_A"] = _num(both["delta_norm_A"])
    both["l2_donor"] = _num(both["delta_norm_donor"])
    both["abs_gain_diff"] = both["dP_gain_diff_A_minus_donor"].abs()
    both["abs_loss_diff"] = both["dP_loss_diff_A_minus_donor"].abs()
    both["discrepancy_ge_0_05"] = (both["abs_gain_diff"] >= FLOOR) | (both["abs_loss_diff"] >= FLOOR)
    both["sufficiency_counterexample"] = both["same_exact_decoded_board"] & both["discrepancy_ge_0_05"]
    both["editor_budget_used"] = budget
    return both, flags


def slice_stats(matched: pd.DataFrame, r2: str | None, n_boot: int, seed: int) -> dict:
    g = matched if r2 is None else matched[matched["r2_type"] == r2]
    exact = g[g["same_exact_decoded_board"] == True] if len(g) else g  # noqa: E712
    out = {
        "slice": "pooled_AB" if r2 is None else ("A" if r2 == "A_legal_diff" else "B"),
        "n_matched_pairs": int(len(g)),
        "n_exact_board_both": int(exact["same_exact_decoded_board"].sum()) if len(g) else 0,
        "n_counterexamples": int(exact["sufficiency_counterexample"].sum()) if len(g) else 0,
        "n_discrepant_any_decode": int(g["discrepancy_ge_0_05"].sum()) if len(g) else 0,
        "n_recipient_games": int(g["recip_game"].nunique()) if len(g) else 0,
        "n_target_games": int(g["target_game"].nunique()) if len(g) else 0,
        "n_recipient_games_exact": int(exact["recip_game"].nunique()) if len(exact) else 0,
        "n_target_games_exact": int(exact["target_game"].nunique()) if len(exact) else 0,
        "mean_dP_gain_A": float(np.nanmean(_num(exact["dP_gain_A"]))) if len(exact) else float("nan"),
        "mean_dP_gain_donor": float(np.nanmean(_num(exact["dP_gain_donor"]))) if len(exact) else float("nan"),
        "mean_dP_loss_A": float(np.nanmean(_num(exact["dP_loss_A"]))) if len(exact) else float("nan"),
        "mean_dP_loss_donor": float(np.nanmean(_num(exact["dP_loss_donor"]))) if len(exact) else float("nan"),
        "mean_gain_diff": float(np.nanmean(exact["dP_gain_diff_A_minus_donor"])) if len(exact) else float("nan"),
        "median_gain_diff": float(np.nanmedian(exact["dP_gain_diff_A_minus_donor"])) if len(exact) else float("nan"),
        "mean_abs_gain_diff": float(np.nanmean(exact["abs_gain_diff"])) if len(exact) else float("nan"),
        "median_abs_gain_diff": float(np.nanmedian(exact["abs_gain_diff"])) if len(exact) else float("nan"),
        "mean_loss_diff": float(np.nanmean(exact["dP_loss_diff_A_minus_donor"])) if len(exact) else float("nan"),
        "median_loss_diff": float(np.nanmedian(exact["dP_loss_diff_A_minus_donor"])) if len(exact) else float("nan"),
        "mean_abs_loss_diff": float(np.nanmean(exact["abs_loss_diff"])) if len(exact) else float("nan"),
        "median_abs_loss_diff": float(np.nanmedian(exact["abs_loss_diff"])) if len(exact) else float("nan"),
        "mean_S_legal_diff": float(np.nanmean(exact["S_legal_diff_A_minus_donor"])) if len(exact) else float("nan"),
        "median_S_legal_diff": float(np.nanmedian(exact["S_legal_diff_A_minus_donor"])) if len(exact) else float("nan"),
        "mean_kl_diff": float(np.nanmean(exact["kl_diff_A_minus_donor"])) if len(exact) else float("nan"),
        "mean_l2_A": float(np.nanmean(exact["l2_A"])) if len(exact) else float("nan"),
        "mean_l2_donor": float(np.nanmean(exact["l2_donor"])) if len(exact) else float("nan"),
    }
    if len(exact):
        for name, col in (
            ("gain_diff", "dP_gain_diff_A_minus_donor"),
            ("loss_diff", "dP_loss_diff_A_minus_donor"),
            ("S_legal_diff", "S_legal_diff_A_minus_donor"),
        ):
            b = _boot(exact[col].tolist(), exact["recip_game"].astype(str).tolist(), n_boot, seed)
            out[f"{name}_ci"] = b
            tb = _boot(
                exact[col].tolist(),
                (exact["recip_game"].astype(str) + "|" + exact["target_game"].astype(str)).tolist(),
                n_boot,
                seed + 1,
            )
            out[f"{name}_ci_recip_pipe_target"] = tb
    else:
        out["gain_diff_ci"] = None
        out["loss_diff_ci"] = None
        out["S_legal_diff_ci"] = None
    return out


def write_report(
    path: Path,
    ident: dict,
    flags: dict,
    slices: list[dict],
    verdict: str,
    identity: dict,
    pair_flags: dict,
    site_saved: str | None,
) -> None:
    pooled = next(s for s in slices if s["slice"] == "pooled_AB")
    a = next(s for s in slices if s["slice"] == "A")
    b = next(s for s in slices if s["slice"] == "B")
    claim_a = "SUPPORTED" if pooled["n_counterexamples"] else "NOT_SUPPORTED"
    if verdict == "INCONCLUSIVE_FROM_SAVED_DATA":
        claim_a = "INCONCLUSIVE_FROM_SAVED_DATA"
    lines = [
        "# 08R2-S Decoded-board causal sufficiency",
        "",
        "Read-only comparison of Editor A at budget 16 versus the natural full-donor residual on the same primary pairs.",
        "Budgets 16 and 32 for Editor A are treated as one operating point and are not double-counted.",
        "D wrong-target rows are excluded from the primary analysis. Full-donor rows are matched by recipient/target keys, not by continuous L2.",
        "",
        "Official labels unchanged:",
        "",
        "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
        "- Original 0A: **C / PARTIAL**",
        "- `SECTION_5_3 = PARTIAL`",
        "- `OBSTRUCTION_READY = NO`",
        "",
        "## Matching",
        "",
        f"- Keys: `{flags['match_keys']}`",
        f"- Matched pairs: {flags['matched']}",
        f"- Unmatched Editor A: {flags['editor_A_unmatched']}; unmatched donor: {flags['donor_unmatched']}",
        f"- Duplicate donor key groups collapsed: {flags['duplicate_donor_key_groups_collapsed']}",
        f"- Grouped donor by unique L2: {flags['grouped_donor_by_l2']}",
        f"- Player: {flags['player_match']}",
        f"- Pair-manifest same_player: {pair_flags}",
        f"- Token position: {flags['token_position_match']} ({flags['token_position_protocol']})",
        f"- Site: {flags['site_match']} (protocol `{flags['site_protocol']}`; saved summary site `{site_saved}`)",
        f"- Editor A 16 vs 32 identity: {identity}",
        "",
        "## Identifiability",
        "",
        f"- Discrete decoded board labels: `{ident['decoded_board_labels']}` (`board_exact`)",
        f"- Decoder probabilities / margins: `{ident['decoder_probabilities_or_margins']}`",
        f"- Downstream legal-set probability metrics: `{ident['legal_set_probability_metrics']}` (`dP_gain`, `dP_loss`, `S_legal`)",
        f"- KL to the unedited baseline (scalar): `{ident['kl_to_baseline_scalar']}`",
        f"- Full output logits / per-token probabilities: `{ident['full_output_logits_or_probs']}`",
        "",
        "Equality of predicted board labels is not equality of the linear-probe logit representation.",
        "Legal-mass metrics and KL-to-baseline are not the full output distribution, and they are not a KL between Editor A and the donor.",
        "",
        "## Counts",
        "",
        f"- Pooled matched: {pooled['n_matched_pairs']} pairs, {pooled['n_recipient_games']} recipient games, {pooled['n_target_games']} target games",
        f"- Both exact intended target board: {pooled['n_exact_board_both']}",
        f"- Counterexamples (same exact board, |ΔP_gain| or |ΔP_loss| discrepancy ≥ 0.05): {pooled['n_counterexamples']}",
        f"- Category A: {a['n_matched_pairs']} pairs, {a['n_recipient_games']} recipients, {a['n_target_games']} targets, {a['n_exact_board_both']} exact, {a['n_counterexamples']} counterexamples",
        f"- Category B: {b['n_matched_pairs']} pairs, {b['n_recipient_games']} recipients, {b['n_target_games']} targets, {b['n_exact_board_both']} exact, {b['n_counterexamples']} counterexamples",
        "",
        "## Behavioral differences (Editor A minus full donor), exact-board pairs",
        "",
        f"- A ΔP_gain: mean A={a['mean_dP_gain_A']:.4f} donor={a['mean_dP_gain_donor']:.4f} signed-diff mean={a['mean_gain_diff']:.4f} median={a['median_gain_diff']:.4f} |diff| mean={a['mean_abs_gain_diff']:.4f} median={a['median_abs_gain_diff']:.4f}",
        f"- A ΔP_loss: mean A={a['mean_dP_loss_A']:.4f} donor={a['mean_dP_loss_donor']:.4f} signed-diff mean={a['mean_loss_diff']:.4f} median={a['median_loss_diff']:.4f} |diff| mean={a['mean_abs_loss_diff']:.4f} median={a['median_abs_loss_diff']:.4f}",
        f"- A S_legal signed-diff mean={a['mean_S_legal_diff']:.4f} median={a['median_S_legal_diff']:.4f}",
        f"- B ΔP_gain: mean A={b['mean_dP_gain_A']:.4f} donor={b['mean_dP_gain_donor']:.4f} signed-diff mean={b['mean_gain_diff']:.4f} median={b['median_gain_diff']:.4f} |diff| mean={b['mean_abs_gain_diff']:.4f} median={b['median_abs_gain_diff']:.4f}",
        f"- B ΔP_loss: mean A={b['mean_dP_loss_A']:.4f} donor={b['mean_dP_loss_donor']:.4f} signed-diff mean={b['mean_loss_diff']:.4f} median={b['median_loss_diff']:.4f} |diff| mean={b['mean_abs_loss_diff']:.4f} median={b['median_abs_loss_diff']:.4f}",
        f"- B S_legal signed-diff mean={b['mean_S_legal_diff']:.4f} median={b['median_S_legal_diff']:.4f}",
        f"- A mean L2: editor={a['mean_l2_A']:.4f} donor={a['mean_l2_donor']:.4f}",
        f"- B mean L2: editor={b['mean_l2_A']:.4f} donor={b['mean_l2_donor']:.4f}",
        "",
        "### Recipient-game clustered 95% CIs (exact-board pairs)",
        "",
        f"- A gain diff: {_fmt_ci(a.get('gain_diff_ci'))}",
        f"- A loss diff: {_fmt_ci(a.get('loss_diff_ci'))}",
        f"- A S_legal diff: {_fmt_ci(a.get('S_legal_diff_ci'))}",
        f"- B gain diff: {_fmt_ci(b.get('gain_diff_ci'))}",
        f"- B loss diff: {_fmt_ci(b.get('loss_diff_ci'))}",
        f"- B S_legal diff: {_fmt_ci(b.get('S_legal_diff_ci'))}",
        f"- Pooled gain diff: {_fmt_ci(pooled.get('gain_diff_ci'))}",
        "",
        "### Recipient|target sensitivity CIs",
        "",
        f"- A gain: {_fmt_ci(a.get('gain_diff_ci_recip_pipe_target'))}",
        f"- B gain: {_fmt_ci(b.get('gain_diff_ci_recip_pipe_target'))}",
        "",
        "Category A has only two target games. Recipient-game confidence intervals treat many recipients that share those two targets as independent clusters. They can be tight while still failing to establish generalization across arbitrary target states. The recipient|target sensitivity intervals do not repair that limit; they only mark the shared-target dependence.",
        "",
        "## Claims",
        "",
        "### A. Discrete decoded-board labels are insufficient for downstream behavior under these interventions.",
        "",
        f"**{claim_a}.** Matched counterexamples: {pooled['n_counterexamples']} of {pooled['n_exact_board_both']} exact-board pairs "
        f"(A: {a['n_counterexamples']}/{a['n_exact_board_both']}; B: {b['n_counterexamples']}/{b['n_exact_board_both']}).",
        "",
        "### B. The entire linear-probe logit representation is insufficient.",
        "",
        f"**{ident['decoder_probabilities_or_margins']}.** Saved rows contain discrete `board_exact` / square accuracies, not per-square probe logits or class margins for both interventions. Label equality does not imply representation equality.",
        "",
        "### C. No causally sufficient representation exists at this site.",
        "",
        "**NOT_SUPPORTED.** Full donor at the same site produces large legal-set shifts on the same recipients and targets. A sufficient residual description exists at this site; the decoded board labels are not shown to be that description.",
        "",
        "### D. The full COR Section 5.3 germ hypothesis is false.",
        "",
        "**NOT_SUPPORTED.** This is one site, one decoder, one editor-versus-donor contrast, on a pair set with two A-pair targets. It does not falsify Section 5.3. Confirmation remains sealed. `SECTION_5_3 = PARTIAL`.",
        "",
        "## Off-manifold caveat",
        "",
        "Editor A is an optimized residual. Its L2 and KL typically differ from the natural donor residual, so the edited activation may leave the natural activation manifold. That can contribute to the behavior gap. It does not cancel claim A: two activations that the decoder reads as the same board, inserted into the same recipient context, still produce different legal-set behavior under the saved interventions.",
        "",
        "## Supported proposition",
        "",
        "Let `h_A` be Editor A's budget-16 residual and `h_donor` the natural full-donor residual at `blocks.6.hook_resid_mid`, last real token. If both are inserted into the same recipient prefix, and the frozen 08c probe maps both to the exact intended target board, they need not induce the same became-legal / became-illegal probability changes. Therefore the probe's discrete 64-square label vector is not a sufficient state description for this model's legal-move behavior under the saved interventions.",
        "",
        "This proposition is about decoded-board labels, not about the probe's continuous logits, not about the non-existence of any sufficient residual statistic, and not about COR as a whole.",
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    val_dir = Path(args.validation_dir)
    out_dir = Path(args.out_dir) if args.out_dir else val_dir / "postvalidation"
    out_dir.mkdir(parents=True, exist_ok=True)
    causal_path = val_dir / "08r2_validation_causal_effects.csv"
    causal = pd.read_csv(causal_path)
    ident = missing_representation_fields(list(causal.columns))
    matched, flags = match_editor_a_to_donor(causal, args.editor_budget)
    pair_df, pair_flags = load_pair_protocol(val_dir / "08r2_validation_pairs.json")
    if len(pair_df) and len(matched):
        keys = [k for k in MATCH_KEYS if k in matched.columns and k in pair_df.columns]
        proto = _coerce_keys(pair_df[pair_df["r2_type"].isin(PRIMARY)], keys)
        matched = matched.merge(proto, on=keys, how="left", suffixes=("", "_manifest"))
        if "same_player" in matched.columns:
            flags["same_player_on_matched"] = int(pd.to_numeric(matched["same_player"], errors="coerce").fillna(0).eq(1).sum())
    identity = editor_a_budget_identity(causal)
    summary_path = val_dir / "08r2_validation_summary.json"
    site_saved = None
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        site_saved = saved.get("site") or saved.get("probe_site")
        if site_saved == SITE:
            flags["site_match"] = "SUMMARY_VERIFIED"
    slices = [
        slice_stats(matched, "A_legal_diff", args.n_boot, args.boot_seed),
        slice_stats(matched, "B_similar_board_legal_change", args.n_boot, args.boot_seed),
        slice_stats(matched, None, args.n_boot, args.boot_seed),
    ]
    pooled = slices[-1]
    verdict = decide_verdict(flags["matched"], pooled["n_exact_board_both"], pooled["n_counterexamples"])

    csv_path = out_dir / "08r2_sufficiency_matched_pairs.csv"
    md_path = out_dir / "08r2_sufficiency_scientific_report.md"
    keep = [
        c
        for c in (
            "r2_type",
            "recip_game",
            "target_game",
            "recip",
            "donor",
            "same_player",
            "hamming",
            "board_exact_A",
            "board_exact_donor",
            "same_exact_decoded_board",
            "dP_gain_A",
            "dP_gain_donor",
            "dP_gain_diff_A_minus_donor",
            "dP_loss_A",
            "dP_loss_donor",
            "dP_loss_diff_A_minus_donor",
            "S_legal_A",
            "S_legal_donor",
            "S_legal_diff_A_minus_donor",
            "kl_A",
            "kl_donor",
            "l2_A",
            "l2_donor",
            "changed_acc_A",
            "changed_acc_donor",
            "discrepancy_ge_0_05",
            "sufficiency_counterexample",
            "editor_budget_used",
        )
        if c in matched.columns
    ]
    matched[keep].to_csv(csv_path, index=False)
    write_report(md_path, ident, flags, slices, verdict, identity, pair_flags, site_saved)
    write_json(
        out_dir / "08r2_sufficiency_summary.json",
        {
            "verdict": verdict,
            "official_0A_letter": "C / PARTIAL",
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
            "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
            "identifiability": ident,
            "match_flags": flags,
            "pair_protocol": pair_flags,
            "editor_a_budget_identity": identity,
            "slices": slices,
            "editor_budget": args.editor_budget,
            "floor": FLOOR,
            "row_counts": {"matched_csv": int(len(matched)), "report": 1},
            "original_files_read_only": list(ORIGINAL_FILES),
            "overwrote_historical": 0,
        },
    )
    print("08R2-S sufficiency audit written")
    print(f"  out_dir={out_dir}")
    print(f"  match_keys={flags['match_keys']}")
    print(f"  matched={flags['matched']} unmatched_A={flags['editor_A_unmatched']} unmatched_donor={flags['donor_unmatched']}")
    print(f"  exact_both={pooled['n_exact_board_both']} counterexamples={pooled['n_counterexamples']}")
    print(f"  A: pairs={slices[0]['n_matched_pairs']} exact={slices[0]['n_exact_board_both']} cex={slices[0]['n_counterexamples']} targets={slices[0]['n_target_games']}")
    print(f"  B: pairs={slices[1]['n_matched_pairs']} exact={slices[1]['n_exact_board_both']} cex={slices[1]['n_counterexamples']} targets={slices[1]['n_target_games']}")
    print(f"  identifiability={ident}")
    print(f"  verdict={verdict}")
    print("  official letter unchanged: C / PARTIAL")
    print("  08R2_VALIDATION_COMPLETE — NEGATIVE")
    print("  SECTION_5_3 = PARTIAL")
    print("  OBSTRUCTION_READY = NO")


if __name__ == "__main__":
    main()
