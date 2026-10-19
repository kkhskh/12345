"""08R2-SYMMETRIC-TARGET three-anchor switching study.

Does not overwrite NST/SEM/SPEC/COST. No confirmation. Letter C / PARTIAL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from run_08r1_real_model_verification import git_hash, patch_and_audit, resolve_checkpoint
from src.othello_r1_eval import resolve_probe_path, sha256_file
from src.othello_r2 import (
    OFFICIAL_0A_LETTER,
    SITE,
    LinearBoardDecoder,
    decoder_state_effects,
    iterative_state_edit,
    load_holdout_game_ids,
    require_no_confirmation,
)
from src.othello_r2_fiber import independent_contrasts, js_divergence, reject_unexpected_preprocessing, svd_probe_geometry
from src.othello_r2_fiber_cost import adopt_drive_defaults
from src.othello_r2_fiber_cost_fixed import SCALE_GRID, calibrate_fixed_js, scaled_displacement
from src.othello_r2_fiber_spec import y_components
from src.othello_r2_null_state import symbolic_state
from src.othello_r2_state_semantics import (
    ABLATIONS,
    ablation_delta,
    crossover_scores,
    pack_full_from_nst,
    paired_boot,
    predict_zhat,
    probe_image_residual,
)
from src.othello_r2_symmetric_target import (
    ANCHORS,
    AT_FLOOR,
    AW_FLOOR,
    BOTH_POS_MIN,
    COVERAGE_MIN,
    FIXED_JS,
    FIXED_JS_TOL,
    N_BOOT,
    NST_PAIR_HASH,
    STATUS_AWAITING,
    STATUS_INFEASIBLE,
    STATUS_STAGE0_FAIL,
    TRIPLET_SEED,
    VERDICT_ENGINEERING,
    WANT_A,
    WANT_B,
    anchor_dependence_ok,
    arithmetic_levels,
    decide_sym_verdict,
    enumerate_triplets,
    isolate_sym_dir,
    natural_symmetric_ok,
    nst_eval_games,
    select_triplets,
    stage_only_identical,
)
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, load_othello_gpt, softmax_np, verify_nanda_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-SYMMETRIC-TARGET")
    p.add_argument("--config", default="configs/toy_othello_08r2_symmetric_target.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--stage", default="0", choices=["0", "1", "2", "3", "4", "5", "6", "all"])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--nst-dir", default=None)
    p.add_argument("--sem-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def merge_cfg(*parts):
    out = {}
    for p in parts:
        out.update(p)
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_sibling(cfg, args, out_dir: Path, name: str, flag: str, required: tuple[str, ...]) -> Path:
    cands = []
    raw = getattr(args, flag, None)
    if raw:
        cands.append(Path(raw))
    cands.append(out_dir.parent / name)
    cands.append(Path(cfg["results_dir"]) / "results" / "08r2" / name)
    for path in cands:
        if path.is_dir() and all((path / f).is_file() for f in required):
            return path
    return cands[0] if raw else cands[1]


def _load_stack(cfg, args):
    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not probe_info["exists"] or ckpt is None or not Path(ckpt).is_file():
        return None
    with Path(probe_info["probe_path"]).open("rb") as handle:
        pack = pickle.load(handle)
    prep = reject_unexpected_preprocessing(pack)
    if not prep["ok"] or str(pack.get("hook")) != SITE:
        raise SystemExit(f"{VERDICT_ENGINEERING} probe {prep}")
    decoder = LinearBoardDecoder(pack)
    contrasts = independent_contrasts(decoder)
    model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=str(ckpt))
    ident = verify_nanda_checkpoint(model)
    if not ident.get("nanda_argmax_match"):
        raise SystemExit(f"{VERDICT_ENGINEERING} checkpoint {ident}")
    for p in model.parameters():
        p.requires_grad_(False)
    svd = svd_probe_geometry(contrasts["W"], float(cfg.get("primary_rank_tol", 1e-8)), [1e-8])
    return {"probe_info": probe_info, "ckpt": ckpt, "decoder": decoder, "contrasts": contrasts, "model": model, "ident": ident, "svd": svd, "P_N": svd["P_N"], "W_probe": contrasts["W"]}


def _patch(model, tokens, h):
    vals = __import__("torch").tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
    logits, _, audit = patch_and_audit(model, [tokens], SITE, vals, batch_size=1)
    return softmax_np(logits.numpy()[0]), logits.numpy()[0], audit


def _editor(cfg, decoder, rec, tgt, h_rec):
    return iterative_state_edit(
        h_rec, tgt["rel_lab"], rec["rel_lab"], decoder.as_torch(),
        steps=int(cfg.get("edit_steps", 40)), lr=float(cfg.get("edit_lr", 0.4)),
        budget=float(cfg.get("editor_budget", 16.0)), lambda_norm=float(cfg["lambda_norm"]),
        eta_preserve=float(cfg["eta_preserve"]), mu_manifold=float(cfg.get("mu_manifold", 0.0)),
    )


def _mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def stage0(cfg, out_dir: Path, sem: Path) -> dict:
    if not (sem / "state_semantics_interventions.csv").is_file():
        write_json(out_dir / "symmetric_target_stage0_status.json", {"ok": 0, "reason": f"missing_sem:{sem}"})
        raise SystemExit(STATUS_STAGE0_FAIL)
    inter = list(csv.DictReader((sem / "state_semantics_interventions.csv").open(encoding="utf-8")))
    contrasts = [r for r in inter if r.get("cell") == "pair_contrast" and str(r.get("valid")) in {"1", "1.0", "True"}]
    if not contrasts:
        raise SystemExit(STATUS_STAGE0_FAIL)

    def slice_rows(pred):
        g = [r for r in contrasts if pred(r)]
        if not g:
            return {"n": 0}
        yf = _mean([float(r["Y_FULL"]) for r in g])
        ys = _mean([float(r["Y_STAGE_ONLY"]) for r in g])
        ye = _mean([float(r["Y_editor"]) for r in g])
        return {"n": len(g), **arithmetic_levels(yf, ys, ye), "Y_FULL": yf, "Y_STAGE_ONLY": ys, "Y_editor": ye}

    slices = {
        "pooled": slice_rows(lambda r: True),
        "A": slice_rows(lambda r: r.get("r2_type") == "A_legal_diff"),
        "B": slice_rows(lambda r: r.get("r2_type") == "B_similar_board_legal_change"),
        "exact": slice_rows(lambda r: r.get("board_exact_editor") in {"1", "1.0", "True", 1, 1.0}),
        "nonexact": slice_rows(lambda r: r.get("board_exact_editor") not in {"1", "1.0", "True", 1, 1.0}),
    }
    write_json(out_dir / "symmetric_target_stage0_arithmetic.json", slices)
    report = [
        "# 08R2-SYMMETRIC-TARGET Stage 0 correction",
        "",
        "Read-only. Does not change STATE-SEMANTICS artifacts or letters.",
        "",
        "Residual R² and causal ΔY measure different properties.",
        "The quantities below are an arithmetic decomposition of mean response levels,",
        "not a decomposition into independent causal mechanisms.",
        "STAGE_ONLY is target-stage-conditioned (player + occupancy of the intended target),",
        "not a target-independent natural-state prior.",
        "",
    ]
    for name, sl in slices.items():
        report += [
            f"## {name} n={sl.get('n')}",
            f"- Y FULL / STAGE / Editor A: {sl.get('Y_FULL')} / {sl.get('Y_STAGE_ONLY')} / {sl.get('Y_editor')}",
            f"- FULL − Editor A: {sl.get('FULL_minus_EditorA')}",
            f"- STAGE_ONLY − Editor A: {sl.get('STAGE_ONLY_minus_EditorA')}",
            f"- FULL − STAGE_ONLY: {sl.get('FULL_minus_STAGE_ONLY')}",
            f"- STAGE share of FULL−EditorA: {sl.get('stage_share_of_full_editor')}",
            "",
        ]
    (out_dir / "symmetric_target_stage0_correction.md").write_text("\n".join(report), encoding="utf-8")
    write_json(out_dir / "symmetric_target_stage0_status.json", {"ok": 1, "n": slices["pooled"]["n"], "sem_dir": str(sem)})
    print("08R2-SYMMETRIC-TARGET STAGE 0 CORRECTION")
    p = slices["pooled"]
    print(f"  n={p['n']} FULL-EditorA={p['FULL_minus_EditorA']:.6f} STAGE-EditorA={p['STAGE_ONLY_minus_EditorA']:.6f} share={p['stage_share_of_full_editor']:.4f}")
    for name in ("A", "B", "exact", "nonexact"):
        sl = slices[name]
        print(f"  {name} n={sl.get('n')} share={sl.get('stage_share_of_full_editor')}")
    return slices


def _load_packs(nst: Path, sem: Path) -> dict:
    nst_pack = np.load(nst / "null_state_pca_ridge.npz")
    packs = {"FULL": pack_full_from_nst(nst_pack)}
    for name in ("BOARD_ONLY", "LEGAL_ONLY", "STAGE_ONLY"):
        z = np.load(sem / f"state_semantics_{name.lower()}.npz")
        packs[name] = {
            "name": name, "mu": z["mu"], "V": z["V"], "W": z["W"], "c": z["c"],
            "idx": z["idx"], "P_N": z["P_N"], "k": int(z["k"]), "alpha": float(z["alpha"]),
        }
    return packs


def stage1(cfg, out_dir: Path, nst: Path, sem: Path) -> dict:
    if not (nst / "null_state_pca_ridge.npz").is_file() or not (sem / "state_semantics_stage_only.npz").is_file():
        raise SystemExit(f"{VERDICT_ENGINEERING} missing frozen models")
    packs = _load_packs(nst, sem)
    if int(packs["FULL"]["k"]) != 64 or abs(float(packs["FULL"]["alpha"]) - 10.0) > 1e-12:
        raise SystemExit(f"{VERDICT_ENGINEERING} FULL k/alpha mismatch")
    ident = {
        "full_k": int(packs["FULL"]["k"]),
        "full_alpha": float(packs["FULL"]["alpha"]),
        "nst_pca_sha256": sha256_file(nst / "null_state_pca_ridge.npz"),
        "board_sha256": sha256_file(sem / "state_semantics_board_only.npz"),
        "legal_sha256": sha256_file(sem / "state_semantics_legal_only.npz"),
        "stage_sha256": sha256_file(sem / "state_semantics_stage_only.npz"),
        "refit": 0,
        "ok": 1,
    }
    write_json(out_dir / "symmetric_target_model_hashes.json", ident)
    write_json(out_dir / "symmetric_target_stage1_status.json", {"ok": 1, **ident})
    print("08R2-SYMMETRIC-TARGET STAGE 1")
    print(f"  frozen FULL k={ident['full_k']} alpha={ident['full_alpha']} refit=0")
    return {"packs": packs}


def stage2(cfg, out_dir: Path) -> dict:
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    require_no_confirmation(val, holdout["game_ids"])
    prior = nst_eval_games(val, holdout["game_ids"])
    enum = enumerate_triplets(val, holdout["game_ids"], prior["games"])
    selected = select_triplets(enum["found"], int(cfg.get("n_a", WANT_A)), int(cfg.get("n_b", WANT_B)), int(cfg.get("triplet_seed", TRIPLET_SEED)))
    census = {
        "n_prior_excluded": prior["n_total"],
        "n_eligible": enum["n_eligible"],
        "n_candidate_A": len(enum["found"]["A_legal_diff"]),
        "n_candidate_B": len(enum["found"]["B_similar_board_legal_change"]),
        "selected_A": selected["n_a"],
        "selected_B": selected["n_b"],
        "n_recipients": selected["n_recipients"],
        "n_targets": selected["n_targets"],
        "recip_target_overlap": selected["recip_target_overlap"],
        "feasible": int(bool(selected.get("feasible")) and selected["n_a"] == int(cfg.get("n_a", WANT_A)) and selected["n_b"] == int(cfg.get("n_b", WANT_B)) and selected["n_recipients"] == 32 and selected["n_targets"] == 64),
        "triplet_sha256": selected["triplet_sha256"],
        "seed": selected["seed"],
        "nst_hash": prior["nst_hash"],
    }
    write_json(out_dir / "symmetric_target_census.json", census)
    write_json(out_dir / "symmetric_target_triplet_manifest.json", {**selected, "wrong_targets": [], "prior_excluded": prior["n_total"]})
    print("08R2-SYMMETRIC-TARGET STAGE 2 CENSUS")
    print(f"  excluded={prior['n_total']} elig={enum['n_eligible']} A_cand={census['n_candidate_A']} B_cand={census['n_candidate_B']}")
    print(f"  selected_A={selected['n_a']} selected_B={selected['n_b']} recip={selected['n_recipients']} targets={selected['n_targets']} feasible={census['feasible']}")
    print(f"  triplet_hash={selected['triplet_sha256']}")
    if not census["feasible"]:
        write_json(out_dir / "symmetric_target_completion_status.json", {
            "status": STATUS_INFEASIBLE,
            "verdict": VERDICT_ENGINEERING,
            "census": census,
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "note": "Strict 16A+16B unused-val design is infeasible. Pairing rules were not relaxed. No new interventions were run.",
            "sem_status_preserved": "LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY",
            "nst_status_preserved": "DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED",
            "fixed_js_status_preserved": "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED",
            "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
            "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
            "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
        })
        (out_dir / "symmetric_target_scientific_report.md").write_text(
            "\n".join([
                "# 08R2-SYMMETRIC-TARGET Scientific Report",
                "",
                f"Verdict: `{VERDICT_ENGINEERING}`",
                f"Status: `{STATUS_INFEASIBLE}`",
                "",
                "The prespecified unused-val 16A+16B design is infeasible after excluding prior causal games.",
                "Pairing rules were not relaxed. No Editor A, learned-direction, or JS-calibrated intervention was run.",
                "This is not a scientific target-switching result.",
                "",
                f"- Prior excluded games: {census['n_prior_excluded']}",
                f"- Eligible unused-val prefixes: {census['n_eligible']}",
                f"- Candidate A triplets: {census['n_candidate_A']}",
                f"- Candidate B triplets: {census['n_candidate_B']}",
                f"- Maximum selected under frozen rules: {census['selected_A']} A + {census['selected_B']} B",
                f"- Recipients / targets in that maximum: {census['n_recipients']} / {census['n_targets']}",
                f"- Required: 16 A + 16 B, 32 recipients, 64 targets",
                f"- Triplet hash of maximum selected: `{census['triplet_sha256']}`",
                "",
                "Preserved: LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY, DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED,",
                "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED, TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED,",
                "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE, 08R2_VALIDATION_COMPLETE — NEGATIVE,",
                "0A C / PARTIAL, SECTION_5_3 = PARTIAL, OBSTRUCTION_READY = NO.",
                "",
            ]),
            encoding="utf-8",
        )
        raise SystemExit(STATUS_INFEASIBLE)
    return {"selected": selected, "val": val, "prior": prior}


def stage3(cfg, out_dir: Path) -> None:
    protocol = {
        "experiment": "08R2-SYMMETRIC-TARGET",
        "not_independent_confirmation": 1,
        "triplet_seed": int(cfg.get("triplet_seed", TRIPLET_SEED)),
        "n_a": int(cfg.get("n_a", WANT_A)),
        "n_b": int(cfg.get("n_b", WANT_B)),
        "fixed_js": float(cfg.get("fixed_js", FIXED_JS)),
        "a_t_floor": float(cfg.get("a_t_floor", AT_FLOOR)),
        "a_w_floor": float(cfg.get("a_w_floor", AW_FLOOR)),
        "both_pos_min": float(cfg.get("both_pos_min", BOTH_POS_MIN)),
        "coverage_min": int(cfg.get("coverage_min", COVERAGE_MIN)),
        "n_boot": int(cfg.get("n_boot", N_BOOT)),
        "anchors": list(ANCHORS),
        "models": list(ABLATIONS),
        "primary": "FULL natural anchor",
        "scale_grid": list(SCALE_GRID),
    }
    write_json(out_dir / "symmetric_target_protocol.json", protocol)
    write_json(out_dir / "symmetric_target_protocol_hash.json", {"sha256": _sha256_text(json.dumps(protocol, sort_keys=True))})
    print("08R2-SYMMETRIC-TARGET STAGE 3 FROZEN")
    print(f"  protocol_sha256={_sha256_text(json.dumps(protocol, sort_keys=True))}")


def _slice(rows, pred, n_boot, seed):
    g = [r for r in rows if pred(r)]
    if not g:
        return {"n": 0}
    out = {"n": len(g)}
    for key in ("A_T", "A_W", "C"):
        boot = paired_boot([r[key] for r in g], n_boot, seed)
        out[key] = boot
    out["both_positive_frac"] = float(np.mean([r.get("both_positive", 0) for r in g]))
    return out


def stage_eval(cfg, out_dir: Path, args, nst: Path, sem: Path, *, smoke: bool) -> None:
    write_dir = out_dir / "smoke" if smoke else out_dir
    ensure_dir(write_dir)
    man = json.loads((out_dir / "symmetric_target_triplet_manifest.json").read_text(encoding="utf-8"))
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    require_no_confirmation(val, holdout["game_ids"])
    stack = _load_stack(cfg, args)
    if stack is None:
        print(STATUS_AWAITING)
        return
    packs = _load_packs(nst, sem)
    H = cache_site(stack["model"], tokens_of(val), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    decoder = stack["decoder"]
    trips = man["triplets"]
    if smoke:
        trips = [trips[int(cfg.get("smoke_triplet_index", 0))]]
    js_target = float(cfg.get("fixed_js", FIXED_JS))
    inv_atol = float(cfg.get("probe_proba_atol_f32", 1e-5))
    n_boot = int(cfg.get("n_boot", N_BOOT))
    attempts = {"triplets": 0, "editor_fail": 0, "js_fail": 0, "invariance_fail": 0, "valid_full": 0}
    rows, cals, doses, idents = [], [], [], []
    logit_pack, probe_pack = {}, {}
    hooks_ok = 1
    models = ("FULL", "LEGAL_ONLY", "BOARD_ONLY", "STAGE_ONLY")

    for trip in trips:
        attempts["triplets"] += 1
        rec, tgt, wrg = val[int(trip["recip"])], val[int(trip["target_t"])], val[int(trip["target_w"])]
        if rec["game_id"] != trip["recip_game"] or tgt["game_id"] != trip["target_t_game"] or wrg["game_id"] != trip["target_w_game"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} index/game mismatch")
        b_t, b_w = symbolic_state(tgt), symbolic_state(wrg)
        z_t = predict_zhat(b_t, packs["STAGE_ONLY"])
        z_w = predict_zhat(b_w, packs["STAGE_ONLY"])
        idents.append({
            "recip_game": trip["recip_game"],
            "stage_only_identical": int(stage_only_identical(b_t, b_w)),
            "stage_only_zhat_identical": int(bool(np.allclose(z_t, z_w))),
            "occ_t": int(tgt["occupied"]),
            "occ_w": int(wrg["occupied"]),
        })
        h_r = np.asarray(H[int(trip["recip"])], dtype=np.float64)
        ed_t = _editor(cfg, decoder, rec, tgt, h_r)
        ed_w = _editor(cfg, decoder, rec, wrg, h_r)
        if not ed_t.get("valid_edit") or not ed_w.get("valid_edit"):
            attempts["editor_fail"] += 1
            continue
        anchors = {
            "N": h_r,
            "T": np.asarray(ed_t["edited"], dtype=np.float64),
            "W": np.asarray(ed_w["edited"], dtype=np.float64),
        }
        exact = {
            "N": decoder_state_effects(decoder, h_r, rec["rel_lab"], rec["rel_lab"])["board_exact"],
            "T": decoder_state_effects(decoder, anchors["T"], rec["rel_lab"], tgt["rel_lab"])["board_exact"],
            "W": decoder_state_effects(decoder, anchors["W"], rec["rel_lab"], wrg["rel_lab"])["board_exact"],
        }
        full_ok = 1
        recs = {}
        for anc_name, a in anchors.items():
            probs_a, logits_a, audit_a = _patch(stack["model"], rec["tokens"], a)
            hooks_ok = hooks_ok and int(audit_a.get("ok", 0))
            probe_a = decoder.proba_np(a).reshape(64, 3)
            logit_pack[f"{trip['recip_game']}__{anc_name}__anchor"] = np.asarray(logits_a, dtype=np.float64)
            probe_pack[f"{trip['recip_game']}__{anc_name}__anchor"] = probe_a
            y_a_t = y_components(probs_a, rec["legal"], tgt["legal"])
            y_a_w = y_components(probs_a, rec["legal"], wrg["legal"])
            for model in models:
                d_t = ablation_delta(a, b_t, packs[model])
                d_w = ablation_delta(a, b_w, packs[model])
                got = {}
                for tag, delta, score_tgt in (("t", d_t, tgt), ("w", d_w, wrg)):
                    def ev(t, _d=delta, _a=a):
                        h = _a + float(t) * np.asarray(_d)
                        probs, logits, audit = _patch(stack["model"], rec["tokens"], h)
                        return {"js": float(js_divergence(probs_a, probs)), "probs": probs, "logits": logits, "audit": audit, "h": h}

                    matched = calibrate_fixed_js(ev, target_js=js_target, grid=SCALE_GRID, tol=float(cfg.get("fixed_js_tol", FIXED_JS_TOL)))
                    sc = scaled_displacement(delta, matched.get("t", float("nan"))) if matched.get("t") == matched.get("t") else {"l2": float("nan")}
                    probe = decoder.proba_np(matched["h"]).reshape(64, 3) if matched.get("ok") and matched.get("h") is not None else None
                    inv = float(np.max(np.abs(probe - probe_a))) if probe is not None else float("nan")
                    cell = f"{model}_{anc_name}_{tag}"
                    cals.append({"cell": cell, "recip_game": trip["recip_game"], "ok": matched.get("ok", 0), "reason": matched.get("reason"), "js": matched.get("js"), "t": matched.get("t"), "l2": sc.get("l2"), "invariance_max_err": inv, "probe_image": probe_image_residual(delta, stack["W_probe"])})
                    if not matched.get("ok"):
                        if model == "FULL":
                            full_ok = 0
                            attempts["js_fail"] += 1
                        got[tag] = None
                        continue
                    if inv > inv_atol:
                        attempts["invariance_fail"] += 1
                        if model == "FULL":
                            full_ok = 0
                    y_t = y_components(matched["probs"], rec["legal"], tgt["legal"])
                    y_w = y_components(matched["probs"], rec["legal"], wrg["legal"])
                    logit_pack[f"{trip['recip_game']}__{cell}"] = np.asarray(matched["logits"], dtype=np.float64)
                    if probe is not None:
                        probe_pack[f"{trip['recip_game']}__{cell}"] = probe
                    got[tag] = {"Y_T": y_t, "Y_W": y_w, "js": matched["js"], "l2": sc.get("l2")}
                    rows.append({"cell": cell, "r2_type": trip["r2_type"], "recip_game": trip["recip_game"], "valid": 1, "Y_T": y_t["Y"], "Y_W": y_w["Y"], "G_T": y_t["G"], "Q_T": y_t["Q"], "js": matched["js"], "l2": sc.get("l2"), "anchor_exact": exact[anc_name]})
                if got.get("t") and got.get("w"):
                    xo = crossover_scores(got["t"]["Y_T"]["Y"], got["w"]["Y_T"]["Y"], got["w"]["Y_W"]["Y"], got["t"]["Y_W"]["Y"])
                    recs[(model, anc_name)] = {
                        **xo,
                        "Y_T_ht": got["t"]["Y_T"]["Y"],
                        "Y_T_hw": got["w"]["Y_T"]["Y"],
                        "Y_W_hw": got["w"]["Y_W"]["Y"],
                        "Y_W_ht": got["t"]["Y_W"]["Y"],
                        "Y_anchor_T": y_a_t["Y"],
                        "Y_anchor_W": y_a_w["Y"],
                    }
        if full_ok and all(("FULL", a) in recs for a in ANCHORS):
            attempts["valid_full"] += 1
            for (model, anc_name), xo in recs.items():
                rows.append({
                    "cell": "triplet_contrast", "model": model, "anchor": anc_name, "r2_type": trip["r2_type"],
                    "recip_game": trip["recip_game"], "target_t_game": trip["target_t_game"], "target_w_game": trip["target_w_game"],
                    "valid": 1, "board_exact_T": exact["T"], "board_exact_W": exact["W"], **xo,
                })

    write_csv(write_dir / "symmetric_target_calibrations.csv", cals or [{"reason": "none"}])
    write_csv(write_dir / "symmetric_target_interventions.csv", rows or [{"reason": "none"}])
    write_csv(write_dir / "symmetric_target_stage_identity.csv", idents or [{"reason": "none"}])
    if logit_pack:
        np.savez_compressed(write_dir / "symmetric_target_full_logits.npz", **logit_pack)
    if probe_pack:
        np.savez_compressed(write_dir / "symmetric_target_probe_probabilities.npz", **probe_pack)
    contrasts = [r for r in rows if r.get("cell") == "triplet_contrast" and r.get("valid") == 1]
    n_boot_s = int(cfg.get("boot_seed", 82))
    slices = {}
    for model in models:
        for anc in ANCHORS:
            key = f"{model}_{anc}"
            slices[key] = _slice(contrasts, lambda r, m=model, a=anc: r.get("model") == m and r.get("anchor") == a, n_boot, n_boot_s)
        slices[f"{model}_N_A"] = _slice(contrasts, lambda r, m=model: r.get("model") == m and r.get("anchor") == "N" and r.get("r2_type") == "A_legal_diff", n_boot, n_boot_s + 1)
        slices[f"{model}_N_B"] = _slice(contrasts, lambda r, m=model: r.get("model") == m and r.get("anchor") == "N" and r.get("r2_type") == "B_similar_board_legal_change", n_boot, n_boot_s + 2)
    nat = slices.get("FULL_N") or {}
    at = slices.get("FULL_T") or {}
    aw = slices.get("FULL_W") or {}
    both = float(nat.get("both_positive_frac") or 0)
    gate = {
        "reconstruction_ok": 1,
        "split_ok": 1,
        "integrity_ok": int(hooks_ok and attempts["triplets"] == len(trips)),
        "invariance_ok": int(attempts["invariance_fail"] == 0 and attempts["valid_full"] > 0),
        "js_match_ok": int(attempts["valid_full"] > 0),
        "coverage_ok": int(attempts["valid_full"] >= (1 if smoke else int(cfg.get("coverage_min", COVERAGE_MIN)))),
        "natural_symmetric_ok": natural_symmetric_ok(nat.get("A_T") or {}, nat.get("A_W") or {}, nat.get("C") or {}, both, float(cfg.get("a_t_floor", AT_FLOOR)), float(cfg.get("both_pos_min", BOTH_POS_MIN))) if nat.get("n") else 0,
        "anchor_dependence_ok": anchor_dependence_ok(nat, at, aw) if nat.get("n") and at.get("n") and aw.get("n") else 0,
    }
    verdict = decide_sym_verdict(gate)
    boot_rows = []
    for name, sl in slices.items():
        for met, val in sl.items():
            if isinstance(val, dict) and "ci95_low" in val:
                boot_rows.append({"slice": name, "metric": met, **val})
    write_csv(write_dir / "symmetric_target_bootstrap.csv", boot_rows or [{"n": 0}])
    write_json(write_dir / "symmetric_target_attempt_accounting.json", attempts)
    write_json(write_dir / "symmetric_target_dependency_manifest.json", {
        "git_hash": git_hash(), "checkpoint_sha256": sha256_file(stack["ckpt"]),
        "probe_sha256": sha256_file(stack["probe_info"]["probe_path"]),
        "triplet_hash": man.get("triplet_sha256"), "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
    })
    write_json(write_dir / "symmetric_target_completion_status.json", {
        "status": "SMOKE_COMPLETE" if smoke else "SYMMETRIC_TARGET_COMPLETE",
        "verdict": "SMOKE_ONLY" if smoke else verdict,
        "gate": gate, "slices": slices, "attempts": attempts, "smoke": int(smoke),
        "used_confirmation": 0, "official_0A_letter": OFFICIAL_0A_LETTER,
        "sem_status_preserved": "LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY",
        "nst_status_preserved": "DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED",
        "fixed_js_status_preserved": "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED",
        "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
        "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "section_5_3_status": "PARTIAL", "obstruction_ready": "NO",
    })
    (write_dir / "symmetric_target_scientific_report.md").write_text(
        "\n".join([
            "# 08R2-SYMMETRIC-TARGET Scientific Report",
            "",
            f"Verdict: `{'SMOKE_ONLY' if smoke else verdict}`",
            "Positive C alone is not bidirectional switching. Natural-anchor FULL is primary.",
            f"Attempts: {attempts}",
            f"Gate: {gate}",
            f"FULL_N: {nat}",
            f"FULL_T: {at}",
            f"FULL_W: {aw}",
            "",
        ]),
        encoding="utf-8",
    )
    print("08R2-SYMMETRIC-TARGET")
    print(f"  verdict={'SMOKE_ONLY' if smoke else verdict}")
    print(f"  valid_full={attempts['valid_full']}/{len(trips)}")
    print(f"  attempts={attempts}")
    print(f"  gate={gate}")
    print(f"  FULL_N={nat}")
    print("  LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY preserved")
    print("  DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED preserved")
    if smoke and attempts["valid_full"] != 1:
        raise SystemExit(f"{VERDICT_ENGINEERING} smoke failed")


def stage6(cfg, out_dir: Path) -> None:
    man = json.loads((out_dir / "symmetric_target_triplet_manifest.json").read_text(encoding="utf-8"))
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    inter = list(csv.DictReader((out_dir / "symmetric_target_interventions.csv").open(encoding="utf-8")))
    logits = np.load(out_dir / "symmetric_target_full_logits.npz")
    n_ok = n_bad = 0
    by = {t["recip_game"]: t for t in man["triplets"]}
    for row in inter:
        if row.get("cell") == "triplet_contrast" or row.get("Y_T") in (None, ""):
            continue
        trip = by.get(row.get("recip_game"))
        if not trip:
            continue
        rec, tgt = val[int(trip["recip"])], val[int(trip["target_t"])]
        lkey = f"{row['recip_game']}__{row['cell']}"
        if lkey not in logits.files:
            continue
        recon = y_components(softmax_np(logits[lkey]), rec["legal"], tgt["legal"])
        try:
            ok = abs(float(row["Y_T"]) - float(recon["Y"])) <= 1e-6
        except (TypeError, ValueError):
            ok = False
        n_ok += int(ok)
        n_bad += int(not ok)
    status = {"ok": int(n_bad == 0 and n_ok > 0), "n_match": n_ok, "n_mismatch": n_bad}
    write_json(out_dir / "symmetric_target_stage6_status.json", status)
    print("08R2-SYMMETRIC-TARGET STAGE 6")
    print(f"  ok={status['ok']} match={n_ok} mismatch={n_bad}")
    if not status["ok"]:
        raise SystemExit(STATUS_STAGE0_FAIL)


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
    out_dir = ensure_dir(isolate_sym_dir(cfg["results_dir"])["results_dir"])
    prereg = Path(cfg.get("preregistration", "docs/experiments/08r2_symmetric_target_preregistration.md"))
    if prereg.is_file():
        (out_dir / "symmetric_target_preregistration.md").write_text(prereg.read_text(encoding="utf-8"), encoding="utf-8")
    nst = _resolve_sibling(cfg, args, out_dir, "null_state_transfer", "nst_dir", ("null_state_pca_ridge.npz",))
    sem = _resolve_sibling(cfg, args, out_dir, "state_semantics", "sem_dir", ("state_semantics_interventions.csv",))
    stages = ["0", "1", "2", "3", "4", "5", "6"] if args.stage == "all" else [args.stage]
    if "0" in stages:
        stage0(cfg, out_dir, sem)
    if "1" in stages:
        stage1(cfg, out_dir, nst, sem)
    if "2" in stages:
        stage2(cfg, out_dir)
    if "3" in stages:
        if not (out_dir / "symmetric_target_triplet_manifest.json").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 2 missing")
        if not json.loads((out_dir / "symmetric_target_census.json").read_text()).get("feasible"):
            raise SystemExit(STATUS_INFEASIBLE)
        stage3(cfg, out_dir)
    if "4" in stages:
        stage_eval(cfg, out_dir, args, nst, sem, smoke=True)
    if "5" in stages:
        stage_eval(cfg, out_dir, args, nst, sem, smoke=False)
    if "6" in stages:
        if not (out_dir / "symmetric_target_full_logits.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 5 logits missing")
        stage6(cfg, out_dir)


if __name__ == "__main__":
    main()
