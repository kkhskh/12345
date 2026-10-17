"""08i: assemble 0A decision from written artifacts only. Does not invent tables."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv, write_text
from src.toy_othello import MODEL_SPEC, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="08i final 0A evaluation.")
    parser.add_argument("--config", default="configs/toy_othello.json")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def decide(art: dict) -> dict:
    sim = art.get("sim") or {}
    b = art.get("b") or {}
    c = art.get("c_gate") or {}
    fspec = art.get("fspec") or []
    e = art.get("e") or []
    if not sim.get("passed"):
        return _pack("D", "Insufficient model quality or inconclusive experiment", "FAIL", "Simulator did not pass.")
    if b.get("status") in {None, "BLOCKED"}:
        return _pack("D", "Insufficient model quality or inconclusive experiment", "FAIL", "Model checkpoint not loaded; later gates not executed.")
    if b.get("gate0") == "FAIL" or b.get("status") == "FAIL_FIXTURE":
        return _pack("D", "Insufficient model quality or inconclusive experiment", "FAIL", "Gate 0 failed.")
    gate1 = c.get("gate1")
    if gate1 in {None, "BLOCKED"}:
        return _pack(
            "D",
            "Insufficient model quality or inconclusive experiment",
            "FAIL",
            "Gate 0 passed; Gate 1 (state probes) has not been executed yet. No Section 5.3 claim.",
        )
    if gate1 == "FAIL_DECODE":
        return _pack("B", "Decodable world state without validated causal abstraction", "FAIL", "Board decode failed controls.")

    full_lift = _mean(e, "donor_legal_mass_after") - _mean(e, "donor_legal_mass_before") if e else float("nan")
    became = {r["method"]: _f(r, "mean_became_legal_dP") for r in fspec}
    state_best = max(
        (became.get(k, float("nan")) for k in ("linear_board_dirs", "train_mean_square_delta", "donor_probe_subspace")),
        default=float("nan"),
    )
    attr = became.get("sign_attribution", float("nan"))
    steer = became.get("logit_steer_plus4", float("nan"))
    full = became.get("full_patch", float("nan"))
    decode_ok = gate1 == "PASS_DECODE"
    causal_reuse = (
        state_best == state_best
        and state_best >= 0.08
        and (attr != attr or state_best >= attr + 0.05)
        and (steer != steer or state_best >= steer + 0.05)
    )
    full_ok = (full == full and full >= 0.10) or (full_lift == full_lift and full_lift >= 0.10)

    if decode_ok and causal_reuse:
        return _pack(
            "A",
            "Causally validated mathematical state abstraction",
            "PASS",
            "Held-out decode plus targeted reusable edits beat attribution/logit-steer.",
        )
    if decode_ok and not causal_reuse:
        letter = "B" if not full_ok else "C"
        title = (
            "Decodable world state without validated causal abstraction"
            if letter == "B"
            else "Causal information flow without reusable state representation"
        )
        return _pack(letter, title, "PARTIAL", "Decode or full patch exists; reusable state-specific edits do not beat controls.")
    if full_ok:
        return _pack("C", "Causal information flow without reusable state representation", "PARTIAL", "Full interchange works without a validated compact state map.")
    return _pack("D", "Insufficient model quality or inconclusive experiment", "FAIL", "Evidence insufficient after executed stages.")


def _mean(rows, key) -> float:
    import numpy as np

    vals = [_f(r, key) for r in rows]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _pack(letter, title, gate, missing):
    return {
        "letter": letter,
        "title": title,
        "section_5_3": gate,
        "obstruction_ready": "NO",
        "missing": missing,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    results_dir = ensure_dir(cfg["results_dir"])
    fig_dir = ensure_dir(cfg["figures_dir"])
    art = {
        "sim": _read_json(results_dir / "08a_simulator_validation.json"),
        "man": _read_json(results_dir / "08a_dataset_manifest.json"),
        "b": _read_json(results_dir / "08b_baseline_summary.json"),
        "c_gate": _read_json(results_dir / "08c_gate1.json"),
        "c_dec": _read_csv(results_dir / "08c_state_decode.csv"),
        "c_ctrl": _read_csv(results_dir / "08c_probe_controls.csv"),
        "d": _read_csv(results_dir / "08d_state_abstraction_consistency.csv"),
        "e": _read_csv(results_dir / "08e_full_interchange.csv"),
        "f": _read_csv(results_dir / "08f_state_edit_effects.csv"),
        "fspec": _read_csv(results_dir / "08f_causal_specificity.csv"),
        "g": _read_csv(results_dir / "08g_state_vs_decision_failures.csv"),
        "h": _read_csv(results_dir / "08h_candidate_germ_validation.csv"),
    }
    executed = {k: art[k] not in (None, []) for k in art}
    decision = decide(art)
    write_csv(results_dir / "08i_decision_summary.csv", [{**decision, **{f"executed_{k}": int(v) for k, v in executed.items()}}])
    (results_dir / "08i_decision.json").write_text(json.dumps({**decision, "executed": executed, "model_spec": MODEL_SPEC}, indent=2) + "\n")
    report = render_report(decision, art, executed, cfg)
    write_text(results_dir / "08_final_scientific_report.md", report)
    write_manifest(results_dir, cfg, decision, executed)
    leftover_figures(fig_dir, art)
    print(f"0A decision {decision['letter']} — {decision['title']}")
    print(f"SECTION_5_3_EMPIRICAL_GATE = {decision['section_5_3']}")
    print("OBSTRUCTION_READY = NO")
    print(decision["missing"])


def write_manifest(results_dir: Path, cfg: dict, decision: dict, executed: dict) -> None:
    import subprocess

    commit = ""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    obj = {
        "experiment": "0A",
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": cfg,
        "decision": decision,
        "executed": executed,
        "model_spec": MODEL_SPEC,
        "obstruction_ready": "NO",
    }
    (results_dir / "08i_experiment_manifest.json").write_text(json.dumps(obj, indent=2) + "\n")


def leftover_figures(fig_dir: Path, art: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = art.get("g") or []
    if g:
        from collections import Counter

        c = Counter(r["label"] for r in g)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar(list(c.keys()), list(c.values()))
        ax.set_title("08g consistency tags (not causal proof)")
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "08g_state_vs_decision.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        "hidden activation → state abstraction → candidate legality germ → decision effect\n"
        "Arrows are schematic. Causal arrows require 08f targeted legality change.",
        fontsize=9,
    )
    fig.savefig(fig_dir / "08h_germ_schematic.png", dpi=140)
    plt.close(fig)


def render_report(decision, art, executed, cfg) -> str:
    b = art.get("b") or {}
    c = art.get("c_gate") or {}
    sim_ok = bool((art.get("sim") or {}).get("passed"))
    val_mass = (b.get("val") or {}).get("mean_legal_mass")
    test_mass = (b.get("test") or {}).get("mean_legal_mass")
    return f"""# 08 final scientific report — Experiment 0A

**Official 07d NO_GO, 07f A, 07g–07i B, 07j A, 07k C unchanged.**

**0A decision: {decision['letter']} — {decision['title']}**

**SECTION_5_3_EMPIRICAL_GATE = {decision['section_5_3']}**

**OBSTRUCTION_READY = NO**

{decision['missing']}

## 1. Executive conclusion

This experiment asks whether Othello-GPT uses a mathematically specified internal board state that can be decoded, transformed, and causally edited on held-out games. Decoding a board (Claim A / Li–Nanda) is treated as necessary groundwork, not success. A Section 5.3 germ requires targeted counterfactual legality changes that beat attribution and logit steering.

## 2. Relation to Section 5.3

The paper's \(s_c(v;x)=P_v^c\\phi_v(x)\) is replaced, empirically, by a state abstraction \(R_v(\\phi_v)\\to\\hat B\) and a candidate germ \(s_m=1[m\\in L(\\hat B)]\). This report does not define compatibility maps or obstruction.

## 3. Replication vs new claims

Li et al. (ICLR 2023) and Nanda already showed board decode and some activation edits. Anything here that only reproduces those facts is labeled replication. A new claim requires held-out legal-set specificity plus controls in `08f_causal_specificity.csv`.

## 4–5. Simulator, data, model

Simulator passed: {sim_ok}. Artifact: `08a_simulator_validation.json`.
Dataset: `08a_dataset_manifest.json`. Splits are by complete games.
Model: {MODEL_SPEC['hf_repo']} / {MODEL_SPEC['checkpoint_file']}. Gate 0: {b.get('gate0')}. Status: {b.get('status')}.
Val legal mass: {val_mass}. Test legal mass: {test_mass}.

## 6–8. Decode and abstraction

Gate 1: {c.get('gate1')}. Selected: {c.get('selected')}.
Tables: `08c_state_decode.csv`, `08c_probe_controls.csv`, `08d_state_abstraction_consistency.csv`.
Whole-board exact match is required to interpret square accuracy.

## 9–12. Causal interchange and edits

Full patch: `08e_full_interchange.csv`. State edits: `08f_state_edit_effects.csv`, `08f_causal_specificity.csv`.
Unexecuted stages are not treated as successes. Executed flags: {executed}.

## 13. Non-identifiability

Probe row-spaces, PCA, and full residuals can all write the same logits. Complementary projections are reported when 08f ran.

## 14–15. Failures and germs

`08g_state_vs_decision_failures.csv` uses consistency tags only.
`08h_candidate_germ_validation.csv` evaluates \(s_m\) vs simulator legality. Not an edge / obstruction.

## 16. Uncertainty and limits

Bootstrap-by-game is implemented in `src/toy_othello.bootstrap_by_game` and used when scripts report CIs. Positions from one game are dependent. Synthetic random-legal games are unseen trajectories, not a held-out operand class. Pass tokens are omitted in the training convention.

## 17–19. Classification

Decision {decision['letter']}. SECTION_5_3_EMPIRICAL_GATE={decision['section_5_3']}.
Obstruction is not ready: no validated compatibility relation exists.

## 20. Next step

If A: a later 5.4 experiment must define task-grounded maps between sites that preserve legal-set germs, not learned arbitrary transports.
If not A: the failure is recorded above (identification vs intervention vs model vs theory). Do not rebuild obstruction from a failed germ.

## Reproduce

```text
python run_08a_prepare_othello_data.py --config configs/toy_othello.json
python run_08b_othello_model_baseline.py --config configs/toy_othello.json
python run_08c_othello_state_probes.py --config configs/toy_othello.json
python run_08d_othello_state_abstraction.py --config configs/toy_othello.json
python run_08e_othello_state_interchange.py --config configs/toy_othello.json
python run_08f_othello_state_causal_edits.py --config configs/toy_othello.json
python run_08g_othello_failure_modes.py --config configs/toy_othello.json
python run_08h_othello_candidate_germs.py --config configs/toy_othello.json
python run_08i_othello_final_evaluation.py --config configs/toy_othello.json
```

On Colab, keep artifacts under the mounted Drive `toy_othello` directories and pass `--results-dir` / `--data-dir` / `--checkpoint`.
"""


if __name__ == "__main__":
    main()
