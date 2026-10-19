"""Read-only census failure investigation for 08R2-SYMMETRIC-TARGET.

Does not overwrite SYM/NST/SEM. No Othello-GPT. No confirmation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_07f_modular_causal_sanity_validation import ensure_dir
from src.othello_r2 import OFFICIAL_0A_LETTER, load_holdout_game_ids, require_no_confirmation
from src.othello_r2_fiber_cost import adopt_drive_defaults
from src.othello_r2_null_state import prior_eval_games
from src.othello_r2_symmetric_target import NST_PAIR_HASH, STATUS_INFEASIBLE
from src.othello_r2_symmetric_target_census_audit import (
    compare_to_production,
    eligible_indices,
    exclusion_breakdown,
    fresh_cohort_rows,
    integrity_row,
    isolate_audit_dir,
    progressive_funnel,
    synthetic_b_triplet_rows,
)
from src.othello_runutil import add_labels, load_split, write_json
from src.toy_othello import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-SYMMETRIC-TARGET census audit")
    p.add_argument("--config", default="configs/toy_othello_08r2_symmetric_target.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--fresh-games", type=int, default=400)
    p.add_argument("--fresh-seed", type=int, default=8182)
    p.add_argument("--fresh-prefix-seed", type=int, default=8282)
    return p.parse_args()


def merge_cfg(*parts):
    out = {}
    for p in parts:
        out.update(p)
    return out


def _funnel_block(name: str, funnel: dict) -> list[str]:
    p, b, a = funnel["pairs"], funnel["B"], funnel["A"]
    return [
        f"## {name}",
        "",
        f"- Eligible recipient prefixes: {p['eligible_recipients']}",
        f"- Eligible target prefixes: {p['eligible_targets']}",
        f"- Distinct-game ordered pairs: {p['distinct_game']}",
        f"- Same player to move: {p['same_player']}",
        f"- Hamming 2–8: {p['ham_B_2_8']}",
        f"- Hamming >8: {p['ham_A_gt_8']}",
        f"- Hamming 0 or 1: {p['ham_0_or_1']}",
        f"- Bidirectional legal change vs recipient: {p['bidir_legal_vs_recip']}",
        f"- Category-B recipient–target pairs (`_kind`): {p['B_pairs']}",
        f"- Category-A recipient–target pairs (`_kind`): {p['A_pairs']}",
        f"- Loose B pairs (`_pair_kind`: ham≤8 and any legal difference): {p['B_pairs_loose_pair_kind']}",
        "",
        "B pairs / B triplets / disjoint packing are not interchangeable.",
        "",
        f"- Recipients with ≥2 B targets: {b['recipients_with_ge2_targets']}",
        f"- Recipients with ≥2 equal-occupancy B targets: {b['recipients_with_ge2_equal_occupancy_targets']}",
        f"- T/W same occupancy: {b['tw_same_occupancy']}",
        f"- T/W legal-count slack: {b['tw_legal_count_slack']}",
        f"- T/W distinct games: {b['tw_distinct_games']}",
        f"- T/W legal sets differ: {b['tw_legal_sets_differ']}",
        f"- Category-B T/W triplets: {b['n_triplets']}",
        f"- B packing: {b['packing']['max_packing']} / upper {b['packing']['upper_bound']} ({b['packing']['method']})",
        f"- Category-A T/W triplets: {a.get('n_triplets')}",
        f"- A packing: {a.get('packing', {}).get('max_packing')} / upper {a.get('packing', {}).get('upper_bound')} ({a.get('packing', {}).get('method')})",
        "",
    ]


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(load_config(args.parent_config), load_config(args.r2_config), load_config(args.config))
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    adopt_drive_defaults(cfg, args)
    out_dir = ensure_dir(isolate_audit_dir(cfg["results_dir"])["results_dir"])
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    require_no_confirmation(val, holdout["game_ids"])
    excl = exclusion_breakdown(val, holdout["game_ids"])
    leftover = eligible_indices(val, holdout["game_ids"], excl["games"])
    unused_before_nst = eligible_indices(val, holdout["game_ids"], set(map(str, prior_eval_games(val, holdout["game_ids"])["games"])))
    holdout_only = eligible_indices(val, holdout["game_ids"], set())

    integrity = [integrity_row(val[i]) for i in holdout_only]
    integ = {
        "n": len(integrity),
        "occ_mismatch": int(sum(1 - r["occ_match"] for r in integrity)),
        "legal_mismatch": int(sum(1 - r["legal_match"] for r in integrity)),
    }

    print("  funnel leftover...", flush=True)
    funnels = {
        "leftover_after_332": progressive_funnel(val, leftover, expand_a=True),
    }
    print("  funnel before_nst...", flush=True)
    funnels["unused_after_fiber_spec_before_nst"] = progressive_funnel(val, unused_before_nst, expand_a=False)
    print("  funnel val-holdout...", flush=True)
    funnels["val_minus_holdout_only"] = progressive_funnel(val, holdout_only, expand_a=False)
    print("  production recount leftover...", flush=True)
    prod = compare_to_production(val, holdout["game_ids"], excl["games"])
    syn = add_labels(synthetic_b_triplet_rows())
    syn_funnel = progressive_funnel(syn, list(range(len(syn))))

    existing_ids = {str(r["game_id"]) for r in val}
    for split in ("train", "test"):
        try:
            existing_ids |= {str(r["game_id"]) for r in load_split(Path(cfg["data_dir"]), split)}
        except FileNotFoundError:
            pass
    existing_ids |= set(holdout["game_ids"]) | set(excl["games"])
    print("  fresh cohort...", flush=True)
    fresh_rows = add_labels(fresh_cohort_rows(int(args.fresh_games), int(args.fresh_seed), int(args.fresh_prefix_seed), existing_ids))
    expand_fresh_a = len(fresh_rows) <= 500
    fresh_funnel = progressive_funnel(fresh_rows, list(range(len(fresh_rows))), expand_a=expand_fresh_a)
    fresh_prod = {
        "n_eligible": len(fresh_rows),
        "n_candidate_A": fresh_funnel["A"]["n_triplets"],
        "n_candidate_B": fresh_funnel["B"]["n_triplets"],
        "note": "fresh uses independent funnel, not leftover production enumerator",
    }
    b_pack = int(fresh_funnel["B"]["packing"]["max_packing"] or 0)
    a_ok = int((fresh_funnel["A"]["n_triplets"] or 0) >= 16 or (fresh_funnel["A"].get("recipients_with_ge2_targets") or 0) >= 16)
    fresh_ok = int(b_pack >= 16 and a_ok)

    cause = "absence_of_B_triplets"
    if funnels["leftover_after_332"]["pairs"]["B_pairs"] > 0 and funnels["leftover_after_332"]["B"]["n_triplets"] == 0:
        cause = "B_pairs_exist_but_triplet_constraints_eliminate_them"
    elif funnels["leftover_after_332"]["B"]["n_triplets"] > 0 and funnels["leftover_after_332"]["B"]["packing"]["max_packing"] < 16:
        cause = "B_triplets_exist_but_disjoint_packing_below_16"
    elif funnels["leftover_after_332"]["pairs"]["B_pairs"] == 0:
        cause = "absence_of_B_pairs"

    bug = 0
    if integ["occ_mismatch"] or integ["legal_mismatch"]:
        bug = 1
    if excl["nst_hash_ok"] != 1:
        bug = 1
    if prod["n_candidate_B"] != funnels["leftover_after_332"]["B"]["n_triplets"]:
        bug = 1
    if prod["n_eligible"] != funnels["leftover_after_332"]["pairs"]["eligible_recipients"]:
        bug = 1
    if syn_funnel["B"]["n_triplets"] < 1:
        bug = 1

    unchanged = int(not bug and prod["n_candidate_B"] == 0)
    audit = {
        "status_retained": STATUS_INFEASIBLE if unchanged else "CENSUS_IMPLEMENTATION_ERROR_FOUND",
        "original_status_not_rewritten": 1,
        "implementation_error": int(bug),
        "cause": cause,
        "exclusion": {k: v for k, v in excl.items() if k != "games"},
        "integrity": integ,
        "production_recount": prod,
        "funnels": funnels,
        "synthetic_B_triplets": syn_funnel["B"]["n_triplets"],
        "fresh": {
            "seed": int(args.fresh_seed),
            "prefix_seed": int(args.fresh_prefix_seed),
            "n_games": int(args.fresh_games),
            "n_prefixes": len(fresh_rows),
            "n_games_generated": len({r["game_id"] for r in fresh_rows}),
            "feasible_16A_16B_packing": fresh_ok,
            "funnel": fresh_funnel,
            "production_style": fresh_prod,
            "note": "Proposed NEW evaluation cohort only. Does not change the failed leftover-val study.",
        },
        "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "sym_preregistration_unchanged": 1,
    }
    write_json(out_dir / "census_audit.json", audit)
    report = [
        "# 08R2-SYMMETRIC-TARGET census failure investigation",
        "",
        "Read-only. Original `SYMMETRIC_TARGET_CENSUS_INFEASIBLE` is not rewritten.",
        "No Othello-GPT run. No retraining. No confirmation access.",
        "",
        f"Implementation error: {bug}",
        f"Cause: `{cause}`",
        f"Retained status: `{audit['status_retained']}`",
        "",
        "## Exclusion reconstruction",
        "",
        f"- Fiber+SPEC prior games: {excl['n_prior_fiber_spec']}",
        f"- NST pair games: {excl['n_nst_pair_games']}",
        f"- NST wrong-target games: {excl['n_nst_wrong_games']}",
        f"- Total excluded: {excl['n_total_excluded']}",
        f"- NST hash ok: {excl['nst_hash_ok']} (`{excl['nst_hash']}` vs `{NST_PAIR_HASH}`)",
        "",
        "## Integrity",
        "",
        f"- Occupancy mismatches vs board: {integ['occ_mismatch']}/{integ['n']}",
        f"- Legal-mask mismatches vs simulator: {integ['legal_mismatch']}/{integ['n']}",
        "",
        "## Production recount vs independent funnel",
        "",
        f"- Eligible: production {prod['n_eligible']} / independent {funnels['leftover_after_332']['pairs']['eligible_recipients']}",
        f"- A triplets: production {prod['n_candidate_A']} / independent {funnels['leftover_after_332']['A']['n_triplets']}",
        f"- B triplets: production {prod['n_candidate_B']} / independent {funnels['leftover_after_332']['B']['n_triplets']}",
        f"- Selected: {prod['selected_A']} A + {prod['selected_B']} B hash `{prod['triplet_sha256']}`",
        "",
        "Greedy matching is not used as a proof of infeasibility. B packing uses an empty-set certificate or exact backtrack.",
        "",
    ]
    for name in ("leftover_after_332", "unused_after_fiber_spec_before_nst", "val_minus_holdout_only"):
        report += _funnel_block(name, funnels[name])
    report += [
        "## Synthetic counterexample",
        "",
        f"Injected 3-game B fixture produced {syn_funnel['B']['n_triplets']} B triplets. Enumerator can detect B when it exists.",
        "",
        "## Fresh independent simulator cohort (not the failed study)",
        "",
        f"- Seeds: games {args.fresh_seed}, prefixes {args.fresh_prefix_seed}",
        f"- Generated games / prefixes: {audit['fresh']['n_games_generated']} / {audit['fresh']['n_prefixes']}",
        f"- B pairs: {fresh_funnel['pairs']['B_pairs']}",
        f"- B triplets: {fresh_funnel['B']['n_triplets']}",
        f"- B packing: {fresh_funnel['B']['packing']['max_packing']}",
        f"- A packing: {fresh_funnel['A']['packing']['max_packing']}",
        f"- 16A+16B packing feasible on this fresh draw: {fresh_ok}",
        "",
        "This cohort is a proposed new evaluation pool. It does not alter the leftover-validation study or its preregistration.",
        "",
        "Preserved: LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY, DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED,",
        "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED, TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED,",
        "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE, 08R2_VALIDATION_COMPLETE — NEGATIVE,",
        "0A C / PARTIAL, SECTION_5_3 = PARTIAL, OBSTRUCTION_READY = NO.",
        "",
    ]
    (out_dir / "census_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print("08R2-SYMMETRIC-TARGET CENSUS AUDIT")
    print(f"  excluded={excl['n_total_excluded']} leftover={len(leftover)} occ_bad={integ['occ_mismatch']} legal_bad={integ['legal_mismatch']}")
    print(f"  leftover B_pairs={funnels['leftover_after_332']['pairs']['B_pairs']} B_triplets={funnels['leftover_after_332']['B']['n_triplets']} B_pack={funnels['leftover_after_332']['B']['packing']['max_packing']}")
    print(f"  before_nst B_pairs={funnels['unused_after_fiber_spec_before_nst']['pairs']['B_pairs']} B_triplets={funnels['unused_after_fiber_spec_before_nst']['B']['n_triplets']}")
    print(f"  production A={prod['n_candidate_A']} B={prod['n_candidate_B']} selected={prod['selected_A']}+{prod['selected_B']}")
    print(f"  cause={cause} implementation_error={bug}")
    print(f"  fresh B_triplets={fresh_funnel['B']['n_triplets']} B_pack={fresh_funnel['B']['packing']['max_packing']} A_pack={fresh_funnel['A']['packing']['max_packing']} feasible16={fresh_ok}")
    print(f"  retained={audit['status_retained']}")


if __name__ == "__main__":
    main()
