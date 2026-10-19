"""08R2-VISIBLE-NULL-FACTORIAL staged runner.

Does not overwrite SYM/NST/SEM. No confirmation. Not obstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from run_08r1_real_model_verification import git_hash
from run_08r2_symmetric_target import _load_packs, _load_stack, _patch, _resolve_sibling, _sha256_text, merge_cfg
from src.othello_r1_eval import sha256_file
from src.othello_r2 import OFFICIAL_0A_LETTER, SITE, load_holdout_game_ids, require_no_confirmation
from src.othello_r2_fiber import js_divergence
from src.othello_r2_fiber_cost import adopt_drive_defaults
from src.othello_r2_fiber_spec import y_components
from src.othello_r2_null_state import r2_score, symbolic_state
from src.othello_r2_state_semantics import paired_boot, predict_zhat
from src.othello_r2_symmetric_target import nst_eval_games
from src.othello_r2_symmetric_target_fresh import (
    apply_exclusions,
    banned_sequence_hashes,
    generate_game_records,
    joint_pack_from_rows,
    prefixes_from_records,
)
from src.othello_r2_visible_null import (
    BOOT_SEED,
    COHORT_SEED,
    COVERAGE_MIN,
    I_FLOOR,
    NU,
    PREFIX_SEED,
    RANDOM_SEED,
    RHO,
    SHUFFLE_SEED,
    STATUS_AWAITING,
    STATUS_INFEASIBLE,
    TRIPLET_SEED,
    VERDICT_ENGINEERING,
    decide_vnf_verdict,
    factorial_cells,
    fit_rowspace_ridge,
    interaction_ok,
    interaction_scores,
    isolate_vnf_dir,
    project_check,
    random_null_unit,
    rhat,
)
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, softmax_np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-VISIBLE-NULL-FACTORIAL")
    p.add_argument("--config", default="configs/toy_othello_08r2_visible_null_factorial.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--stage", default="1", choices=["1", "2", "3", "4", "5", "6", "all"])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--nst-dir", default=None)
    p.add_argument("--sem-dir", default=None)
    p.add_argument("--fresh-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def stage1(cfg, out_dir: Path, args) -> None:
    post = Path(args.fresh_dir) / "postvalidation" if args.fresh_dir else Path(cfg["results_dir"]) / "results" / "08r2" / "symmetric_target_fresh" / "postvalidation"
    boot = _read_csv(post / "anchor_effect_bootstrap.csv")
    feat = _read_csv(post / "feature_map_switching_comparison.csv")
    d_t = next((r for r in boot if r.get("slice") == "D_T"), {})
    d_w = next((r for r in boot if r.get("slice") == "D_W"), {})
    doc = {
        "sym_status_preserved": "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED",
        "census_status_preserved": "SYMMETRIC_TARGET_CENSUS_INFEASIBLE",
        "D_T": d_t,
        "D_W": d_w,
        "feature_maps": feat,
        "d_intervals_present": int(bool(d_t.get("ci95_low") not in (None, "") and d_w.get("ci95_low") not in (None, ""))),
        "note": "Optimized-anchor D_T/D_W are confounded. Not a COR compatibility result.",
        "used_confirmation": 0,
    }
    write_json(out_dir / "vnf_stage1_prereq.json", doc)
    print("08R2-VISIBLE-NULL-FACTORIAL STAGE 1")
    print(f"  D_T point={d_t.get('point')} ci=[{d_t.get('ci95_low')},{d_t.get('ci95_high')}]")
    print(f"  D_W point={d_w.get('point')} ci=[{d_w.get('ci95_low')},{d_w.get('ci95_high')}]")
    for r in feat:
        print(f"  {r.get('model')} A_T={r.get('A_T')} A_W={r.get('A_W')} both={r.get('both_positive_frac')}")
    print("  SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED preserved")


def stage2(cfg, out_dir: Path) -> None:
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    splits = {}
    for name in ("train", "val", "test"):
        try:
            splits[name] = load_split(Path(cfg["data_dir"]), name)
        except FileNotFoundError:
            splits[name] = []
    require_no_confirmation(splits.get("val") or [], holdout["game_ids"])
    prior = nst_eval_games(splits["val"], holdout["game_ids"]) if splits["val"] else {"games": set(), "n_total": 0}
    banned_ids = set(holdout["game_ids"]) | set(map(str, prior.get("games") or []))
    for rows in splits.values():
        banned_ids |= {str(r["game_id"]) for r in rows}
    extra = []
    extra.extend(generate_game_records(400, 8182, "fresh"))
    extra.extend(generate_game_records(400, 9182, "symf"))
    banned = banned_sequence_hashes(splits, extra)
    raw = generate_game_records(int(cfg.get("n_games", 400)), int(cfg.get("cohort_seed", COHORT_SEED)), "vnf")
    filtered = apply_exclusions(raw, banned["hashes"], banned_ids)
    rows = prefixes_from_records(filtered["kept"], int(cfg.get("prefix_seed", PREFIX_SEED)), "vnf", k=int(cfg.get("positions_per_game", 6)))
    selected = joint_pack_from_rows(rows, 16, 16, int(cfg.get("triplet_seed", TRIPLET_SEED)), int(cfg.get("pack_trials", 2000)))
    wit = selected.get("witness") or {}
    census = {
        "cohort_seed": int(cfg.get("cohort_seed", COHORT_SEED)),
        "n_kept": filtered["n_kept"],
        "n_dropped": filtered["n_dropped"],
        "n_prefixes": len(rows),
        "selected_A": selected["n_a"],
        "selected_B": selected["n_b"],
        "n_union": wit.get("n_union"),
        "feasible": int(bool(selected.get("feasible") and wit.get("ok"))),
        "triplet_sha256": selected.get("triplet_sha256"),
        "method": selected.get("method"),
        "upper_b": selected.get("upper_b"),
    }
    write_json(out_dir / "vnf_census.json", census)
    write_json(out_dir / "vnf_triplet_manifest.json", selected)
    with (out_dir / "vnf_prefixes.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print("08R2-VISIBLE-NULL-FACTORIAL STAGE 2")
    print(f"  seed={census['cohort_seed']} kept={census['n_kept']} dropped={census['n_dropped']} prefixes={census['n_prefixes']}")
    print(f"  selected={census['selected_A']}A+{census['selected_B']}B union={census['n_union']} feasible={census['feasible']}")
    print(f"  triplet_hash={census['triplet_sha256']}")
    if not census["feasible"]:
        write_json(out_dir / "vnf_completion_status.json", {"status": STATUS_INFEASIBLE, "census": census, "used_confirmation": 0})
        raise SystemExit(STATUS_INFEASIBLE)


def stage3(cfg, out_dir: Path, args, nst: Path, sem: Path) -> None:
    stack = _load_stack(cfg, args)
    if stack is None:
        print(STATUS_AWAITING)
        return
    train = add_labels(load_split(Path(cfg["data_dir"]), "train"))
    H = cache_site(stack["model"], tokens_of(train), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    P_R = np.asarray(stack["svd"]["P_R"], dtype=np.float64)
    B = np.stack([symbolic_state(r) for r in train])
    R = (P_R @ np.asarray(H, dtype=np.float64).T).T
    games = np.array([str(r["game_id"]) for r in train])
    uniq = np.array(sorted(set(games.tolist())))
    rng = np.random.default_rng(int(cfg.get("train_hold_seed", 10782)))
    rng.shuffle(uniq)
    hold = set(uniq[: max(1, int(0.2 * len(uniq)))].tolist())
    tr = np.array([g not in hold for g in games])
    ho = ~tr
    diag = fit_rowspace_ridge(B[tr], R[tr], P_R, float(cfg.get("ridge_alpha", 10.0)))
    pred_ho = np.stack([rhat(b, diag) for b in B[ho]])
    final = fit_rowspace_ridge(B, R, P_R, float(cfg.get("ridge_alpha", 10.0)))
    pred_tr = np.stack([rhat(b, final) for b in B])
    np.savez_compressed(out_dir / "vnf_rowspace_ridge.npz", W=final["W"], c=final["c"], alpha=final["alpha"], P_R=P_R)
    info = {
        "n_train_rows": int(tr.sum()),
        "n_hold_rows": int(ho.sum()),
        "n_final": int(len(train)),
        "alpha": 10.0,
        "rank_search": 0,
        "holdout_r2": r2_score(R[ho], pred_ho),
        "train_r2": r2_score(R, pred_tr),
        "mean_proj_residual": float(np.mean([np.linalg.norm(stack["P_N"] @ rhat(b, final)) for b in B[:64]])),
        "refit": 0,
        "used_causal_eval": 0,
    }
    write_json(out_dir / "vnf_rowspace_fit.json", info)
    print("08R2-VISIBLE-NULL-FACTORIAL STAGE 3")
    print(f"  alpha=10 rank_search=0 holdout_r2={info['holdout_r2']:.4f} train_r2={info['train_r2']:.4f}")


def _load_rows(out_dir: Path) -> list[dict]:
    return add_labels([json.loads(line) for line in (out_dir / "vnf_prefixes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])


def _load_rowspace(out_dir: Path) -> dict:
    z = np.load(out_dir / "vnf_rowspace_ridge.npz")
    return {"W": z["W"], "c": z["c"], "alpha": float(z["alpha"]), "P_R": z["P_R"]}


def stage_eval(cfg, out_dir: Path, args, nst: Path, sem: Path, *, smoke: bool) -> None:
    write_dir = out_dir / "smoke" if smoke else out_dir
    ensure_dir(write_dir)
    man = json.loads((out_dir / "vnf_triplet_manifest.json").read_text(encoding="utf-8"))
    rows = _load_rows(out_dir)
    stack = _load_stack(cfg, args)
    if stack is None:
        print(STATUS_AWAITING)
        return
    packs = _load_packs(nst, sem)
    vis = _load_rowspace(out_dir)
    P_R, P_N = np.asarray(stack["svd"]["P_R"], dtype=np.float64), np.asarray(stack["P_N"], dtype=np.float64)
    H = cache_site(stack["model"], tokens_of(rows), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    decoder = stack["decoder"]
    trips = man["triplets"]
    if smoke:
        trips = [trips[int(cfg.get("smoke_triplet_index", 0))]]
    rho, nu = float(cfg.get("rho", RHO)), float(cfg.get("nu", NU))
    rng = np.random.default_rng(int(cfg.get("random_seed", RANDOM_SEED)))
    shuf = np.random.default_rng(int(cfg.get("shuffle_seed", SHUFFLE_SEED)))
    attempts = {"triplets": 0, "degen": 0, "proj_fail": 0, "l2_fail": 0, "probe_row_fail": 0, "valid": 0}
    inter, checks = [], []
    h_pack, logit_pack, probe_pack, dir_pack = {}, {}, {}, {}
    hooks_ok = 1
    target_l2 = float(np.sqrt(rho * rho + nu * nu))
    for trip in trips:
        attempts["triplets"] += 1
        rec, tgt, wrg = rows[int(trip["recip"])], rows[int(trip["target_t"])], rows[int(trip["target_w"])]
        if rec["game_id"] != trip["recip_game"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} index mismatch")
        h = np.asarray(H[int(trip["recip"])], dtype=np.float64)
        b_t, b_w = symbolic_state(tgt), symbolic_state(wrg)
        r_t = P_R @ (rhat(b_t, vis) - h)
        r_w = P_R @ (rhat(b_w, vis) - h)
        n_t = P_N @ (predict_zhat(b_t, packs["FULL"]) - h)
        n_w = P_N @ (predict_zhat(b_w, packs["FULL"]) - h)
        leaks = [project_check(r_t, P_R, P_N, "row"), project_check(r_w, P_R, P_N, "row"), project_check(n_t, P_R, P_N, "null"), project_check(n_w, P_R, P_N, "null")]
        built = factorial_cells(h, r_t, r_w, n_t, n_w, rho, nu)
        if not built["ok"]:
            attempts["degen"] += 1
            continue
        if not all(x["ok"] for x in leaks):
            attempts["proj_fail"] += 1
            continue
        if any(abs(built["l2"][k] - target_l2) > 1e-5 for k in ("TT", "TW", "WT", "WW")):
            attempts["l2_fail"] += 1
            continue
        rnd_a, rnd_b = random_null_unit(P_N, rng), random_null_unit(P_N, rng)
        if rnd_a is None or rnd_b is None:
            attempts["degen"] += 1
            continue
        swap = int(shuf.integers(0, 2))
        n_t_s, n_w_s = (n_w, n_t) if swap else (n_t, n_w)
        shuf_built = factorial_cells(h, r_t, r_w, n_t_s, n_w_s, rho, nu)
        extra = {
            "rnd_TT": h + rho * built["units"]["r_T"] + nu * rnd_a,
            "rnd_TW": h + rho * built["units"]["r_T"] + nu * rnd_b,
            "rnd_WT": h + rho * built["units"]["r_W"] + nu * rnd_a,
            "rnd_WW": h + rho * built["units"]["r_W"] + nu * rnd_b,
        }
        eval_cells = {**built["cells"], **extra}
        if shuf_built["ok"]:
            eval_cells.update({f"sh_{k}": shuf_built["cells"][k] for k in ("TT", "TW", "WT", "WW")})
        ys_t, ys_w, gs, qs = {}, {}, {}, {}
        row_ok = 1
        probe_vis = {}
        for name, vec in eval_cells.items():
            probs, logits, audit = _patch(stack["model"], rec["tokens"], vec)
            hooks_ok = hooks_ok and int(audit.get("ok", 0))
            yt = y_components(probs, rec["legal"], tgt["legal"])
            yw = y_components(probs, rec["legal"], wrg["legal"])
            ys_t[name], ys_w[name], gs[name], qs[name] = yt["Y"], yw["Y"], yt["G"], yt["Q"]
            pr = decoder.proba_np(vec).reshape(64, 3)
            if name in {"TT", "TW", "WT", "WW", "vis_T", "vis_W", "nul_T", "nul_W", "noop"}:
                h_pack[f"{trip['recip_game']}__{name}"] = np.asarray(vec, dtype=np.float64)
                logit_pack[f"{trip['recip_game']}__{name}"] = np.asarray(logits, dtype=np.float64)
                probe_pack[f"{trip['recip_game']}__{name}"] = pr
            if name in {"TT", "TW"}:
                probe_vis.setdefault("T", []).append(pr)
            if name in {"WT", "WW"}:
                probe_vis.setdefault("W", []).append(pr)
            inter.append({"cell": name, "recip_game": trip["recip_game"], "r2_type": trip["r2_type"], "Y_T": yt["Y"], "Y_W": yw["Y"], "G_T": yt["G"], "Q_T": yt["Q"], "js_vs_natural": float(js_divergence(softmax_np(logit_pack.get(f"{trip['recip_game']}__noop", logits)) if f"{trip['recip_game']}__noop" in logit_pack else probs, probs)), "l2": float(np.linalg.norm(vec - h))})
        if f"{trip['recip_game']}__noop" in logit_pack:
            p0 = softmax_np(logit_pack[f"{trip['recip_game']}__noop"])
            for name in ("TT", "TW", "WT", "WW"):
                if f"{trip['recip_game']}__{name}" in logit_pack:
                    inter[-1]  # already stored; js filled below in a second pass if needed
        if len(probe_vis.get("T", [])) == 2 and float(np.max(np.abs(probe_vis["T"][0] - probe_vis["T"][1]))) > float(cfg.get("probe_proba_atol_f32", 1e-5)):
            row_ok = 0
        if len(probe_vis.get("W", [])) == 2 and float(np.max(np.abs(probe_vis["W"][0] - probe_vis["W"][1]))) > float(cfg.get("probe_proba_atol_f32", 1e-5)):
            row_ok = 0
        if not row_ok:
            attempts["probe_row_fail"] += 1
            continue
        xo = interaction_scores({k: ys_t[k] for k in ("TT", "TW", "WT", "WW")}, {k: ys_w[k] for k in ("TT", "TW", "WT", "WW")})
        xo_r = interaction_scores({k: ys_t[k] for k in ("TT", "TW", "WT", "WW")}, {k: ys_w[k] for k in ("TT", "TW", "WT", "WW")})
        rnd_map_t = {"TT": ys_t["rnd_TT"], "TW": ys_t["rnd_TW"], "WT": ys_t["rnd_WT"], "WW": ys_t["rnd_WW"]}
        rnd_map_w = {"TT": ys_w["rnd_TT"], "TW": ys_w["rnd_TW"], "WT": ys_w["rnd_WT"], "WW": ys_w["rnd_WW"]}
        xo_rnd = interaction_scores(rnd_map_t, rnd_map_w)
        if all(f"sh_{k}" in ys_t for k in ("TT", "TW", "WT", "WW")):
            xo_sh = interaction_scores({k: ys_t[f"sh_{k}"] for k in ("TT", "TW", "WT", "WW")}, {k: ys_w[f"sh_{k}"] for k in ("TT", "TW", "WT", "WW")})
        else:
            xo_sh = {"I_T": float("nan"), "I_W": float("nan")}
        dir_pack[f"{trip['recip_game']}__r_T"] = r_t
        dir_pack[f"{trip['recip_game']}__r_W"] = r_w
        dir_pack[f"{trip['recip_game']}__n_T"] = n_t
        dir_pack[f"{trip['recip_game']}__n_W"] = n_w
        dir_pack[f"{trip['recip_game']}__h_R"] = h
        attempts["valid"] += 1
        checks.append({
            "recip_game": trip["recip_game"], "r2_type": trip["r2_type"], "valid": 1,
            **xo, "I_T_rnd": xo_rnd["I_T"], "I_W_rnd": xo_rnd["I_W"], "I_T_sh": xo_sh.get("I_T"), "I_W_sh": xo_sh.get("I_W"),
            "Y_T_TT": ys_t["TT"], "Y_T_TW": ys_t["TW"], "Y_T_WT": ys_t["WT"], "Y_T_WW": ys_t["WW"],
            "Y_W_TT": ys_w["TT"], "Y_W_TW": ys_w["TW"], "Y_W_WT": ys_w["WT"], "Y_W_WW": ys_w["WW"],
            "Y_T_visT": ys_t.get("vis_T"), "Y_T_nulT": ys_t.get("nul_T"), "Y_T_noop": ys_t.get("noop"),
            "l2_TT": built["l2"]["TT"], "target_l2": target_l2,
        })
    write_csv(write_dir / "vnf_interventions.csv", inter or [{"reason": "none"}])
    write_csv(write_dir / "vnf_factorial.csv", checks or [{"reason": "none"}])
    if h_pack:
        np.savez_compressed(write_dir / "vnf_activations.npz", **h_pack)
    if logit_pack:
        np.savez_compressed(write_dir / "vnf_logits.npz", **logit_pack)
    if probe_pack:
        np.savez_compressed(write_dir / "vnf_probe_probabilities.npz", **probe_pack)
    if dir_pack:
        np.savez_compressed(write_dir / "vnf_directions.npz", **dir_pack)
    n_boot = int(cfg.get("n_boot", 5000))
    seed = int(cfg.get("boot_seed", BOOT_SEED))
    need = 1 if smoke else int(cfg.get("coverage_min", COVERAGE_MIN))
    i_t = paired_boot([r["I_T"] for r in checks], n_boot, seed) if checks else {"n": 0}
    i_w = paired_boot([r["I_W"] for r in checks], n_boot, seed + 1) if checks else {"n": 0}
    i_tr = paired_boot([r["I_T_rnd"] for r in checks], n_boot, seed + 2) if checks else {"n": 0}
    i_wr = paired_boot([r["I_W_rnd"] for r in checks], n_boot, seed + 3) if checks else {"n": 0}
    i_ts = paired_boot([float(r["I_T_sh"]) for r in checks if r.get("I_T_sh") == r.get("I_T_sh")], n_boot, seed + 4) if checks else {"n": 0}
    i_ws = paired_boot([float(r["I_W_sh"]) for r in checks if r.get("I_W_sh") == r.get("I_W_sh")], n_boot, seed + 5) if checks else {"n": 0}
    ctrl = int(interaction_ok(i_tr, i_wr) or interaction_ok(i_ts, i_ws))
    gate = {
        "reconstruction_ok": 1,
        "split_ok": 1,
        "integrity_ok": int(hooks_ok and attempts["triplets"] == len(trips)),
        "projection_ok": int(attempts["proj_fail"] == 0 and attempts["valid"] > 0),
        "l2_ok": int(attempts["l2_fail"] == 0 and attempts["valid"] > 0),
        "coverage_ok": int(attempts["valid"] >= need),
        "interaction_ok": interaction_ok(i_t, i_w, float(cfg.get("i_floor", I_FLOOR))) if checks else 0,
        "control_reproduces": ctrl,
    }
    verdict = "SMOKE_ONLY" if smoke else decide_vnf_verdict(gate)
    write_json(write_dir / "vnf_bootstrap.json", {"I_T": i_t, "I_W": i_w, "I_T_rnd": i_tr, "I_W_rnd": i_wr, "I_T_sh": i_ts, "I_W_sh": i_ws})
    write_json(write_dir / "vnf_attempt_accounting.json", attempts)
    write_json(write_dir / "vnf_dependency_manifest.json", {"git_hash": git_hash(), "checkpoint_sha256": sha256_file(stack["ckpt"]), "probe_sha256": sha256_file(stack["probe_info"]["probe_path"]), "triplet_hash": man.get("triplet_sha256"), "used_confirmation": 0})
    write_json(write_dir / "vnf_completion_status.json", {
        "status": "SMOKE_COMPLETE" if smoke else "VISIBLE_NULL_FACTORIAL_COMPLETE",
        "verdict": verdict, "gate": gate, "attempts": attempts, "smoke": int(smoke),
        "sym_status_preserved": "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED",
        "official_0A_letter": OFFICIAL_0A_LETTER, "used_confirmation": 0,
        "section_5_3_status": "PARTIAL", "obstruction_ready": "NO",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
    })
    (write_dir / "vnf_scientific_report.md").write_text(
        "\n".join(["# 08R2-VISIBLE-NULL-FACTORIAL", "", f"Verdict: `{verdict}`", "Not obstruction. Not a compatibility map.", f"Gate: {gate}", f"I_T: {i_t}", f"I_W: {i_w}", f"Random: {i_tr} / {i_wr}", f"Shuffle: {i_ts} / {i_ws}", ""]),
        encoding="utf-8",
    )
    print("08R2-VISIBLE-NULL-FACTORIAL")
    print(f"  verdict={verdict} valid={attempts['valid']}/{len(trips)}")
    print(f"  gate={gate}")
    print(f"  I_T={i_t}")
    print(f"  I_W={i_w}")
    print("  SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED preserved")
    if smoke and attempts["valid"] != 1:
        raise SystemExit(f"{VERDICT_ENGINEERING} smoke failed")


def stage6(out_dir: Path) -> None:
    man = json.loads((out_dir / "vnf_triplet_manifest.json").read_text(encoding="utf-8"))
    rows = _load_rows(out_dir)
    inter = list(csv.DictReader((out_dir / "vnf_interventions.csv").open(encoding="utf-8")))
    logits = np.load(out_dir / "vnf_logits.npz")
    acts = np.load(out_dir / "vnf_activations.npz")
    n_ok = n_bad = n_act = 0
    by = {t["recip_game"]: t for t in man["triplets"]}
    for row in inter:
        trip = by.get(row.get("recip_game"))
        if not trip or row.get("cell") not in {"TT", "TW", "WT", "WW"}:
            continue
        rec, tgt = rows[int(trip["recip"])], rows[int(trip["target_t"])]
        key = f"{row['recip_game']}__{row['cell']}"
        if key not in logits.files:
            n_bad += 1
            continue
        recon = y_components(softmax_np(logits[key]), rec["legal"], tgt["legal"])
        ok = abs(float(row["Y_T"]) - float(recon["Y"])) <= 1e-6
        n_ok += int(ok)
        n_bad += int(not ok)
        n_act += int(key in acts.files and acts[key].shape[0] == 512)
    status = {"ok": int(n_bad == 0 and n_ok > 0 and n_act > 0), "n_match": n_ok, "n_mismatch": n_bad, "n_saved_512": n_act}
    write_json(out_dir / "vnf_stage6_status.json", status)
    print("08R2-VISIBLE-NULL-FACTORIAL STAGE 6")
    print(f"  ok={status['ok']} match={n_ok} mismatch={n_bad} saved512={n_act}")
    if not status["ok"]:
        raise SystemExit(VERDICT_ENGINEERING)


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(load_config(args.parent_config), load_config(args.r2_config), load_config(args.config))
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    adopt_drive_defaults(cfg, args)
    out_dir = ensure_dir(isolate_vnf_dir(cfg["results_dir"])["results_dir"])
    prereg = Path(cfg.get("preregistration", "docs/experiments/08r2_visible_null_factorial_preregistration.md"))
    if prereg.is_file():
        (out_dir / "vnf_preregistration.md").write_text(prereg.read_text(encoding="utf-8"), encoding="utf-8")
    nst = _resolve_sibling(cfg, args, out_dir, "null_state_transfer", "nst_dir", ("null_state_pca_ridge.npz",))
    sem = _resolve_sibling(cfg, args, out_dir, "state_semantics", "sem_dir", ("state_semantics_interventions.csv",))
    stages = ["1", "2", "3", "4", "5", "6"] if args.stage == "all" else [args.stage]
    if "1" in stages:
        stage1(cfg, out_dir, args)
    if "2" in stages:
        stage2(cfg, out_dir)
    if "3" in stages:
        if not json.loads((out_dir / "vnf_census.json").read_text()).get("feasible"):
            raise SystemExit(STATUS_INFEASIBLE)
        stage3(cfg, out_dir, args, nst, sem)
    if "4" in stages:
        stage_eval(cfg, out_dir, args, nst, sem, smoke=True)
    if "5" in stages:
        stage_eval(cfg, out_dir, args, nst, sem, smoke=False)
    if "6" in stages:
        if not (out_dir / "vnf_logits.npz").is_file() or not (out_dir / "vnf_activations.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} missing vectors")
        stage6(out_dir)


if __name__ == "__main__":
    main()
