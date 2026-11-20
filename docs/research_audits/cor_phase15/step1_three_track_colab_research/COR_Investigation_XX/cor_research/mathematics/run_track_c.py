"""Track C runner (CPU only; independent of all training).  Usage: python -m cor_research.mathematics.run_track_c OUT.json
Computes every Track C mathematical check and applies the PREDECLARED classification logic (protocols/od_design/od_protocol.json).
The classification is a function of the computed checks only; it does not use any neural result."""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E
E.configure("cpu"); E.legacy_paths()
import numpy as np
import xvitask as T, genlib as GL, relpairs as RP
from cor_research.mathematics import od_original as OO, od_repaired as OR, t4_order as T4O


def t4_reference():
    rng = np.random.default_rng(5); d = GL.sample(RP.ENV1, 4000, rng); q = rng.integers(0, 6, 4000)
    Pc, Pb = GL.posterior_closed(RP.ENV1, d["X"], q), GL.posterior_brute(RP.ENV1, d["X"], q)
    return dict(closed_vs_brute_max_abs=float(np.abs(Pc - Pb).max()), bayes_accuracy=float((Pb.argmax(1) == d["Y"][np.arange(4000), q]).mean()))


def classify(R):
    """Predeclared logic (protocols/od_design/track_c_protocol.json), v2 after review B (C-MAJ-1..4):
    original O/D: source-level instantiation -> C4 iff the exact order-1 certificate (no order-1 additive rule reproduces the GATE Bayes
    decisions, Farkas-verified) holds in every condition; germ-level definition -> C5/underdetermined (Phase-1 Attack F: P = 0 gives
    O = D = 0; metric choice arbitrary).  Repaired O/D on GATE -> C2 iff R1 is exact and the factor-deletion destinations equal the
    independent marginalised-joint Bayes reference (M1 is NOT a C2 condition: it is a restricted, GATE-only C3 statement).
    Neural prediction: C5, REVIEWER-ADJUDICATED, with every candidate considered listed (it is not a designer-only enumeration)."""
    orig = R["original"]; rep = R["repaired_gate"]; t4 = R["t4_order"]; v = {}
    o1 = all((c["LP_max_margin_by_order"].get("1") or c["LP_max_margin_by_order"].get(1))["representable"] is False and
             (c["LP_max_margin_by_order"].get("1") or c["LP_max_margin_by_order"].get(1)).get("farkas_exact", {}).get("verified_exact", False) for c in rep.values())
    v["original_OD"] = dict(source_level_instantiation="C4" if (o1 and orig["theorems"]["T3"]["holds"]) else "UNRESOLVED",
                            germ_level_definition="C5 / underdetermined (Phase-1 Attack F; metric arbitrariness)")
    r1 = all(c["R1_max_err_signed_identity"] < 1e-9 for c in rep.values()); m2 = all(max(c["M2_deleted_h_destination_equals_bayes_without_factor"].values()) < 1e-9 for c in rep.values())
    m1 = all(c["M1_O_is_function_of_sorted_posterior"] for c in rep.values())
    v["repaired_OD_gate"] = "C2" if (r1 and m2) else "C1-candidate (requires a frozen neural test and a new review)"
    v["restricted_theorems"] = ["C3: " + t4["statement"]] + (["C3 (GATE only; false in general): the gluing gap O is a function of the Bayes posterior (O > 0 iff exact top-2 tie)"] if m1 else [])
    v["neural_prediction"] = ("C5 (reviewer-adjudicated)" if (r1 and m2) else "candidate exists: freeze a neural test")
    v["candidates_considered"] = {
        "factor-deletion (FD) destinations": "equal exact Bayes with the factor's source marginalised out (M2): no prediction beyond Bayes on the declared model",
        "O-tie sensitivity": "O is a function of the posterior on GATE only; a near-Bayes network is near a tie there by Bayes-closeness alone",
        "soft-gate expected-energy pooling": "the mean-field / log-linear vs arithmetic-mixture pooling distinction (standard; not COR-specific); not in the note; needs an "
                                             "unstated bridge assumption (a scalar internal gate estimate multiplying energies); GATE models only (review B §3.6)",
        "T4 latent-role expected energy": "geometric pooling over role hypotheses differs from Bayes in about 1% of decisions: calibration-level (generators/t4.equivalence_table)",
        "T4 additive complex": "not representable below order 18 at the log-odds level; the only non-trivial completion adds latent roles = standard factor-graph marginal inference"}
    v["reasons"] = dict(R1_exact=r1, FD_equals_marginalised_joint_bayes=m2, M1_gate_only=m1, order1_certificate_exact=o1, t4_order_lower_bound=t4["certified_order_lower_bound"])
    return v


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(E.RUN_ROOT, "track_c", "track_c_math.json")
    R = dict(t4_reference=t4_reference(), original=dict(theorems=OO.checks(), gate_source_level=OO.gate_source_level(T)), repaired_gate=OR.run(T),
             t4_order=T4O.run(GL, RP.ENV1))
    R["classification"] = classify(R); E.atomic_json(R, out); print(json.dumps(R["classification"], indent=1))
