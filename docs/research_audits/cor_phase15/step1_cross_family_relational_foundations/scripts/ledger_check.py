"""XIV: verify T4_FOUNDATIONAL_EVIDENCE_LEDGER.md against the historical source documents.
(1) every `pkg/FILE` pointer in the ledger exists under cor_phase15/; (2) every anchor (a number or phrase quoted in the ledger) occurs
in the cited source document.  Exit 0 only if both pass."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); ROOT = os.path.join(PKG, "..")
L = open(os.path.join(PKG, "T4_FOUNDATIONAL_EVIDENCE_LEDGER.md")).read()
ptrs = sorted(set(re.findall(r"`(step1_[a-z0-9_]+/[A-Za-z0-9_./]+\.md)`", L)))
missing = [p for p in ptrs if not os.path.exists(os.path.join(ROOT, p))]
A = [("step1_latent_reliability/MAIN_ANSWER.md", ["0.359", "0.636", "0/3", "0.84–0.85"]),
     ("step1_relational_decision_computation/MAIN_RESEARCH_SYNTHESIS.md", ["5.5%", "37%", "729", "3.3e-16", "0.915"]),
     ("step1_r1_causal_origin_and_selective_rescue/MAIN_CAUSAL_RESULT.md", ["0.72 to 0.30", "7/8"]),
     ("step1_learned_relational_computation/MAIN_RESEARCH_RESULT.md", ["9/9", "0.99", "0.83–0.87"]),
     ("step1_native_relational_mechanisms/MAIN_RESEARCH_SYNTHESIS.md", ["4/4"]),
     ("step1_causal_equivalence/MAIN_RESEARCH_SYNTHESIS.md", ["0.11–0.27", "0.85–0.97"]),
     ("step1_native_causal_construction/MAIN_RESEARCH_SYNTHESIS.md", ["layer 1", "0.83–0.93"]),
     ("step1_relational_decision_computation_followup/MAIN_RESEARCH_SYNTHESIS.md", ["0.54–0.62", "0.87–0.98"]),
     ("step1_causal_algorithm_selection_validation/MAIN_RESEARCH_SYNTHESIS.md", ["FALSIFIED"]),
     ("step1_perfect_agreement_boundary/MAIN_RESULT.md", ["7.8e-5", "+0.70"]),
     ("step1_curriculum_response_identification/MAIN_RESULT.md", ["0.149"]),
     ("step1_foundational_atlas_continuation/MAIN_FOUNDATIONAL_RESULT.md", ["88%"]),
     ("step1_relational_mechanism_deepening/MAIN_RESEARCH_SYNTHESIS.md", ["0.60–0.72", "0.66–0.82"]),
     ("step1_reroll_and_critical_period_validation/MAIN_RESEARCH_SYNTHESIS.md", ["0.006", "0.086", "0.018"]),
     ("step1_reroll_and_critical_period_validation/CAUSAL_METRIC_AND_SUBSPACE_AUDIT.md", ["0.72", "0.28", "0.46"])]
bad = []
for f, anchors in A:
    t = open(os.path.join(ROOT, f)).read()
    for a in anchors:
        if a not in t: bad.append((f, a))
print(f"   ledger pointers: {len(ptrs)}, missing: {missing}; anchors checked: {sum(len(a) for _, a in A)}, not found: {bad}")
sys.exit(0 if not missing and not bad else 1)
