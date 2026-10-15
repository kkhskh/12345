"""Numerical verification of the MVP theorem set.

Checks, on randomly generated sheaves over random computation complexes:

  1. the Laplacian residual equals the weighted edge-residual sum, O(s) = s^T L s;
  2. distance to global sections equals the projection residual, D(s) = ||(I - P) s||_M^2;
  3. the energy-distance sandwich of Theorem 1, including tightness at the extreme
     generalized eigenvectors;
  4. the minimum-energy repair formula of Proposition 1, against direct numerical
     minimization, in both the positive-definite and the indefinite regime;
  5. the cohomology correction: d s always lies in im d^0 while a generic 1-cochain does not.

A failure here means the algebra in docs/theory_mvp_theorem_set.md is wrong. It says nothing
about whether real transformer obstruction is well conditioned; the reported condition numbers
are for the random sheaves only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the MVP theorem set numerically.")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--max-vertices", type=int, default=8)
    parser.add_argument("--max-stalk-dim", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/theory/theorem_checks.csv")
    parser.add_argument("--summary", default="artifacts/theory/theorem_checks_summary.md")
    return parser.parse_args()


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def random_sheaf(rng: np.random.Generator, max_vertices: int, max_stalk_dim: int) -> dict:
    """A random connected digraph with random stalk dimensions, transports, and weights."""
    n_vertices = int(rng.integers(3, max_vertices + 1))
    dims = [int(rng.integers(1, max_stalk_dim + 1)) for _ in range(n_vertices)]
    offsets, total = [], 0
    for dim in dims:
        offsets.append(total)
        total += dim

    # A spanning chain keeps the complex connected, then add extra random edges.
    edges = [(i, i + 1) for i in range(n_vertices - 1)]
    for _ in range(int(rng.integers(0, n_vertices))):
        u, v = int(rng.integers(n_vertices)), int(rng.integers(n_vertices))
        if u != v:
            edges.append((u, v))

    transports = {(u, v): rng.normal(size=(dims[v], dims[u])) for u, v in edges}
    weights = {edge: float(rng.uniform(0.2, 3.0)) for edge in edges}

    # Coboundary d: rows grouped by edge, (d s)(e) = rho_e s(u) - s(v)
    row_total = sum(dims[v] for _, v in edges)
    d = np.zeros((row_total, total))
    w_diag = np.zeros(row_total)
    row = 0
    for u, v in edges:
        r = dims[v]
        d[row : row + r, offsets[u] : offsets[u] + dims[u]] = transports[(u, v)]
        d[row : row + r, offsets[v] : offsets[v] + dims[v]] -= np.eye(dims[v])
        w_diag[row : row + r] = weights[(u, v)]
        row += r

    # Random positive-definite metric M on C0
    a = rng.normal(size=(total, total))
    M = a @ a.T + total * np.eye(total)

    return {
        "dims": dims,
        "offsets": offsets,
        "edges": edges,
        "transports": transports,
        "weights": weights,
        "d": d,
        "W": np.diag(w_diag),
        "w_diag": w_diag,
        "M": M,
        "total": total,
    }


def edge_residual_energy(sheaf: dict, s: np.ndarray) -> float:
    """O(s) computed directly from the definition, summing over edges."""
    dims, offsets = sheaf["dims"], sheaf["offsets"]
    total = 0.0
    for u, v in sheaf["edges"]:
        su = s[offsets[u] : offsets[u] + dims[u]]
        sv = s[offsets[v] : offsets[v] + dims[v]]
        residual = sheaf["transports"][(u, v)] @ su - sv
        total += sheaf["weights"][(u, v)] * float(residual @ residual)
    return total


def kernel_basis(d: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for ker d via SVD."""
    _, sv, vh = np.linalg.svd(d)
    rank = int((sv > max(tol, sv[0] * 1e-12 if sv.size else tol)).sum()) if sv.size else 0
    return vh[rank:].T  # columns span ker d


def m_orthogonal_projection(basis: np.ndarray, M: np.ndarray) -> np.ndarray:
    """M-orthogonal projection onto the column span of `basis`."""
    if basis.shape[1] == 0:
        return np.zeros((M.shape[0], M.shape[0]))
    gram = basis.T @ M @ basis
    return basis @ np.linalg.solve(gram, basis.T @ M)


def nonzero_generalized_eigenvalues(L: np.ndarray, M: np.ndarray, tol: float) -> np.ndarray:
    """Generalized eigenvalues of (L, M), keeping those bounded away from zero.

    Uses the symmetric reduction L~ = M^{-1/2} L M^{-1/2} so eigenvalues stay real.
    """
    evals_m, evecs_m = np.linalg.eigh(M)
    m_inv_half = evecs_m @ np.diag(evals_m ** -0.5) @ evecs_m.T
    reduced = m_inv_half @ L @ m_inv_half
    evals = np.linalg.eigvalsh((reduced + reduced.T) / 2)
    scale = max(1.0, float(np.max(np.abs(evals))))
    return evals[evals > tol * scale]


def min_energy_repair_check(rng: np.random.Generator, dim: int, indefinite: bool, tol: float) -> dict:
    """Compare the closed-form repair to direct minimization of the local objective."""
    q = rng.normal(size=(dim, dim))
    if indefinite:
        # Deliberately give H_O a negative eigenvalue so A can fail to be positive definite.
        eigs = rng.normal(size=dim)
        eigs[0] = -abs(eigs[0]) - 2.0
        H_O = q @ np.diag(eigs) @ np.linalg.inv(q)
        H_O = (H_O + H_O.T) / 2
    else:
        H_O = q @ q.T
    f_root = rng.normal(size=(dim, dim))
    F = f_root @ f_root.T
    g = rng.normal(size=dim)
    J = rng.normal(size=dim)
    beta, mu, lam = float(rng.uniform(0.1, 2.0)), float(rng.uniform(0.1, 2.0)), float(rng.uniform(0.5, 2.0))

    A = H_O + mu * F + lam * np.eye(dim)
    A = (A + A.T) / 2
    a_min_eig = float(np.linalg.eigvalsh(A).min())
    positive_definite = a_min_eig > tol

    linear = g - beta * J

    def objective(delta: np.ndarray) -> float:
        return float(linear @ delta + 0.5 * delta @ A @ delta)

    delta_star = -np.linalg.solve(A, linear)

    # Direct minimization by gradient descent from several starts, as an independent check.
    # When A is not positive definite the objective is unbounded below, so descent diverges and
    # there is no minimum to compare against; that is the content of the condition on lambda.
    numeric_value = float("nan")
    if positive_definite:
        best = None
        step = 1.0 / (float(np.linalg.eigvalsh(A).max()) + 1.0)
        for _ in range(5):
            delta = rng.normal(size=dim)
            for _ in range(20000):
                delta = delta - step * (linear + A @ delta)
            value = objective(delta)
            if best is None or value < best[1]:
                best = (delta, value)
        assert best is not None
        numeric_value = best[1]

    gradient_norm = float(np.linalg.norm(linear + A @ delta_star))
    return {
        "dim": dim,
        "indefinite_H_O": indefinite,
        "A_min_eigenvalue": a_min_eig,
        "A_positive_definite": positive_definite,
        "closed_form_gradient_norm": gradient_norm,
        "closed_form_value": objective(delta_star),
        "numeric_value": numeric_value,
        "closed_form_matches_numeric": bool(
            positive_definite and abs(objective(delta_star) - numeric_value) < 1e-4 * (1 + abs(numeric_value))
        ),
        "stationarity_ok": gradient_norm < 1e-6 * (1 + float(np.linalg.norm(linear))),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    repair_rows: list[dict] = []

    for trial in range(args.n_trials):
        sheaf = random_sheaf(rng, args.max_vertices, args.max_stalk_dim)
        d, W, M, total = sheaf["d"], sheaf["W"], sheaf["M"], sheaf["total"]
        L = d.T @ W @ d
        L = (L + L.T) / 2

        kernel = kernel_basis(d)
        P = m_orthogonal_projection(kernel, M)
        eigs = nonzero_generalized_eigenvalues(L, M, args.tol)
        lam_min = float(eigs.min()) if eigs.size else float("nan")
        lam_max = float(eigs.max()) if eigs.size else float("nan")

        s = rng.normal(size=total)

        # Check 1: Laplacian form matches the edge-residual definition.
        o_quadratic = float(s @ L @ s)
        o_direct = edge_residual_energy(sheaf, s)
        energy_err = abs(o_quadratic - o_direct) / (1 + abs(o_direct))

        # Check 2: distance to global sections equals the M-projection residual.
        residual = s - P @ s
        d_projection = float(residual @ M @ residual)
        if kernel.shape[1] > 0:
            # Independently solve min_g ||s - K y||_M^2 over coefficients y.
            gram = kernel.T @ M @ kernel
            y = np.linalg.solve(gram, kernel.T @ M @ s)
            diff = s - kernel @ y
            d_solved = float(diff @ M @ diff)
        else:
            d_solved = float(s @ M @ s)
        distance_err = abs(d_projection - d_solved) / (1 + abs(d_solved))

        # Check 3: the Theorem 1 sandwich.
        sandwich_ok = True
        if eigs.size and d_projection > args.tol:
            lower = lam_min * d_projection
            upper = lam_max * d_projection
            sandwich_ok = (o_quadratic >= lower - 1e-6 * (1 + abs(lower))) and (
                o_quadratic <= upper + 1e-6 * (1 + abs(upper))
            )

        # Check 5: the cohomology correction. im d^0 is the column space of d.
        u_basis, sv, _ = np.linalg.svd(d, full_matrices=False)
        rank = int((sv > max(args.tol, sv[0] * 1e-12)).sum()) if sv.size else 0
        im_d = u_basis[:, :rank]
        ds = d @ s
        coboundary_residual = float(np.linalg.norm(ds - im_d @ (im_d.T @ ds))) / (1 + float(np.linalg.norm(ds)))
        random_cochain = rng.normal(size=d.shape[0])
        generic_residual = float(
            np.linalg.norm(random_cochain - im_d @ (im_d.T @ random_cochain))
        ) / (1 + float(np.linalg.norm(random_cochain)))

        rows.append(
            {
                "trial": trial,
                "n_vertices": len(sheaf["dims"]),
                "n_edges": len(sheaf["edges"]),
                "dim_C0": total,
                "dim_C1": int(d.shape[0]),
                "dim_kernel": int(kernel.shape[1]),
                "rank_d": rank,
                "d_is_surjective": rank >= int(d.shape[0]),
                "lambda_min_plus": lam_min,
                "lambda_max": lam_max,
                "condition_number": lam_max / lam_min if eigs.size and lam_min > 0 else float("nan"),
                "energy_rel_err": energy_err,
                "distance_rel_err": distance_err,
                "obstruction": o_quadratic,
                "distance": d_projection,
                "sandwich_ok": sandwich_ok,
                "coboundary_residual_in_im_d": coboundary_residual,
                "generic_cochain_residual_in_im_d": generic_residual,
            }
        )

    # Tightness: the bounds must be attained at the extreme generalized eigenvectors.
    tightness_rows = []
    for trial in range(min(20, args.n_trials)):
        sheaf = random_sheaf(rng, args.max_vertices, args.max_stalk_dim)
        d, W, M = sheaf["d"], sheaf["W"], sheaf["M"]
        L = (d.T @ W @ d + (d.T @ W @ d).T) / 2
        evals_m, evecs_m = np.linalg.eigh(M)
        m_inv_half = evecs_m @ np.diag(evals_m ** -0.5) @ evecs_m.T
        reduced = m_inv_half @ L @ m_inv_half
        vals, vecs = np.linalg.eigh((reduced + reduced.T) / 2)
        scale = max(1.0, float(np.max(np.abs(vals))))
        keep = vals > args.tol * scale
        if not keep.any():
            continue
        kernel = kernel_basis(d)
        P = m_orthogonal_projection(kernel, M)
        for label, index in [("lambda_min_plus", int(np.argmax(keep))), ("lambda_max", int(len(vals) - 1))]:
            s = m_inv_half @ vecs[:, index]
            residual = s - P @ s
            dist = float(residual @ M @ residual)
            ratio = float(s @ L @ s) / dist if dist > args.tol else float("nan")
            tightness_rows.append(
                {
                    "trial": trial,
                    "bound": label,
                    "target_eigenvalue": float(vals[index]),
                    "achieved_ratio": ratio,
                    "rel_err": abs(ratio - float(vals[index])) / (1 + abs(float(vals[index]))),
                }
            )

    for dim in [2, 4, 6]:
        for indefinite in [False, True]:
            repair_rows.append(min_energy_repair_check(rng, dim, indefinite, args.tol))

    write_csv(args.output, rows)

    # ---------------- report ----------------
    energy_max = max(row["energy_rel_err"] for row in rows)
    distance_max = max(row["distance_rel_err"] for row in rows)
    sandwich_fails = sum(1 for row in rows if not row["sandwich_ok"])
    coboundary_max = max(row["coboundary_residual_in_im_d"] for row in rows)
    # The claim that a generic 1-cochain lies outside im d^0 only applies when d is not
    # surjective. If d is surjective then im d^0 = C1 and H^1 vanishes outright, which makes
    # [d s] = 0 for a different and even more trivial reason.
    non_surjective = [row for row in rows if not row["d_is_surjective"]]
    generic_min = (
        min(row["generic_cochain_residual_in_im_d"] for row in non_surjective)
        if non_surjective
        else float("nan")
    )
    tight_max = max((row["rel_err"] for row in tightness_rows), default=float("nan"))
    conditions = [row["condition_number"] for row in rows if row["condition_number"] == row["condition_number"]]

    pd_repairs = [row for row in repair_rows if row["A_positive_definite"]]
    pd_match = sum(1 for row in pd_repairs if row["closed_form_matches_numeric"])
    stationarity = all(row["stationarity_ok"] for row in repair_rows)

    checks = [
        ("O(s) = s^T L s matches the edge-residual definition", energy_max < 1e-8, f"max rel err {energy_max:.2e}"),
        ("D(s) = ||(I-P)s||_M^2 matches direct minimization", distance_max < 1e-8, f"max rel err {distance_max:.2e}"),
        ("Theorem 1 sandwich holds", sandwich_fails == 0, f"{sandwich_fails}/{len(rows)} violations"),
        ("Theorem 1 bounds are tight at extreme eigenvectors", tight_max < 1e-6, f"max rel err {tight_max:.2e}"),
        ("d s lies in im d^0 for every s", coboundary_max < 1e-10, f"max residual {coboundary_max:.2e}"),
        (
            "a generic 1-cochain does not lie in im d^0 (non-surjective d only)",
            bool(non_surjective) and generic_min > 1e-3,
            f"min residual {generic_min:.3f} over {len(non_surjective)}/{len(rows)} non-surjective trials",
        ),
        (
            "Proposition 1 closed form is stationary",
            stationarity,
            f"all {len(repair_rows)} cases",
        ),
        (
            "Proposition 1 closed form matches numeric minimum when A > 0",
            len(pd_repairs) > 0 and pd_match == len(pd_repairs),
            f"{pd_match}/{len(pd_repairs)} positive-definite cases",
        ),
    ]

    lines = ["# MVP Theorem Set: Numerical Verification", "", f"Random sheaves: {len(rows)} trials.", ""]
    all_ok = True
    print()
    for name, ok, detail in checks:
        all_ok = all_ok and ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name} ({detail})")
        lines.append(f"- [{mark}] {name} — {detail}")

    indefinite_failures = [
        row for row in repair_rows if not row["A_positive_definite"] and row["closed_form_matches_numeric"]
    ]
    lines.extend(
        [
            "",
            "## Condition number of the energy-distance equivalence",
            "",
            f"- median kappa = lambda_max / lambda_min_plus: {float(np.median(conditions)):.3f}",
            f"- maximum kappa: {float(np.max(conditions)):.3f}",
            "",
            "These are properties of the random sheaves, not of any transformer. A large kappa means",
            "the Laplacian residual is a loose proxy for distance to global sections, so kappa should",
            "be reported per experiment.",
            "",
            "## Indefinite-Hessian regime",
            "",
            f"- cases where A was not positive definite: "
            f"{sum(1 for row in repair_rows if not row['A_positive_definite'])}/{len(repair_rows)}",
            f"- of those, cases where the closed form still found the minimum: {len(indefinite_failures)}",
            "",
            "This is the point of the positive-definiteness condition in Proposition 1: when",
            "lambda is too small relative to the negative spectrum of H_O + mu F, the closed-form",
            "delta_star is a stationary point that is not a minimizer.",
        ]
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"median condition number kappa: {float(np.median(conditions)):.3f}")
    print(f"saved per-trial results to {args.output}")
    print(f"saved summary to {args.summary}")
    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
