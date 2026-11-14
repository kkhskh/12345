"""Workstream A (supplement): exact checks of the corrected proofs in MATHEMATICAL_AUDIT_AND_RESULTS.md.
E1  I2-VS, any (orthogonal or oblique) projection: the three on-support pairs with base e=(1,0,0) and sources (0,1,0), (0,0,1), (1,1,0)
    give Delta-m vectors spanning R^3; VS forces P Delta-m = (1^T Delta-m / 2) 1, hence P = (1/2) 1 1^T on the m-block, which is not idempotent.
E2  U-block argument: pairs with Delta-e = 0 and Delta-U != 0 force P_{mU} = 0 and P_{UU} = 0 (VS keeps U and m when S is unchanged).
E3  VS lands I2 outside its input image: e' = (S_s - S_b)/2 + e_b is non-integer when S_s - S_b is odd.
E4  Characterization (Proposition A2): random implementations z = a(V) + b(rest) (direct sum) are interchange-exact under the corresponding
    oblique projection; adding a cross term c(V, rest) breaks it (best projection residual > 0).  Also: a LINEAR invertible re-mixing T z keeps
    exactness (P -> T P T^-1); a NONLINEAR mixing does not.
Writes logs/audit_x1_extra.json"""
import json, os, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); out = {}
one = np.ones(3)
m = lambda e: e.sum() * one - e
eb = np.array([1., 0, 0]); srcs = [np.array(v, float) for v in ((0, 1, 0), (0, 0, 1), (1, 1, 0))]
D = np.stack([m(es) - m(eb) for es in srcs]); T = np.stack([(es.sum() - eb.sum()) * one for es in srcs])     # rows: Delta-m, required Delta-m'
P = np.linalg.solve(D, T).T                                                                                    # P D^T = T^T
out["E1_det_DeltaM"] = float(np.linalg.det(D)); out["E1_forced_P"] = P.round(12).tolist(); out["E1_P_idempotent"] = bool(np.allclose(P @ P, P))
out["E1_P_equals_half_11T"] = bool(np.allclose(P, 0.5 * np.outer(one, one))); out["E1_eig"] = np.linalg.eigvals(P).real.round(12).tolist()
# check P=(1/2)11^T realizes VS on all on-support pairs (S>=1)
E = [np.array(v, float) for v in itertools.product((0, 1), repeat=3) if sum(v) >= 1]
out["E1_half11T_realizes_VS_all_pairs"] = bool(all(np.allclose(m(b) + 0.5 * np.outer(one, one) @ (m(s) - m(b)), s.sum() * one - b) for b in E for s in E))
out["E2_note"] = "pairs with e_s = e_b, U_s != U_b: VS target = base, so P[:, U] (U - U_b) = 0 for all U -> P_mU = P_UU = 0; and U' = U_b for all pairs -> P_Um = 0"
odd = [(b, s) for b in E for s in E if (s.sum() - b.sum()) % 2 == 1]; out["E3_frac_pairs_VS_off_image"] = len(odd) / len(E) ** 2
# E4 characterization
rng = np.random.default_rng(0); nV, nR, n = 4, 5, 400
V = rng.standard_normal((n, 2)); Rr = rng.standard_normal((n, 3)); Wa = rng.standard_normal((2, nV)); Wb = rng.standard_normal((3, nR))
a = np.tanh(V @ Wa); b = np.tanh(Rr @ Wb)
z_sep = np.concatenate([a, b], 1); Pobl = np.diag([1.] * nV + [0.] * nR)
def phi(Vv, Rv, cross=0.0):
    A_ = np.tanh(Vv @ Wa); B_ = np.tanh(Rv @ Wb); z = np.concatenate([A_, B_], 1)
    if cross: z = z + cross * np.concatenate([np.tanh(Vv @ Wa) * np.tanh(Rv @ Wb)[:, :nV], np.zeros((len(Vv), nR))], 1)
    return z
i, j = rng.integers(0, n, 300), rng.integers(0, n, 300)
def best_proj_residual(Z, Zs, Zt, k=nV, iters=4000):
    """least-squares best LINEAR map M with z' = z_b + M (z_s - z_b) (a superset of projections); residual relative to target change."""
    dZ = Zs - Z; dT = Zt - Z; Mx, *_ = np.linalg.lstsq(dZ, dT, rcond=None); return float(np.linalg.norm(dZ @ Mx - dT) / np.linalg.norm(dT))
for nm, cross in (("separable", 0.0), ("cross_term", 0.5)):
    Zb, Zs, Zt = phi(V[i], Rr[i], cross), phi(V[j], Rr[j], cross), phi(V[j], Rr[i], cross)
    out[f"E4_{nm}_oblique_projection_err"] = float(np.abs(Zb + (Zs - Zb) @ Pobl.T - Zt).max())
    out[f"E4_{nm}_best_linear_map_rel_residual"] = best_proj_residual(Zb, Zs, Zt)
Tm = rng.standard_normal((nV + nR, nV + nR)); Zb, Zs, Zt = (phi(V[i], Rr[i]) @ Tm.T, phi(V[j], Rr[j]) @ Tm.T, phi(V[j], Rr[i]) @ Tm.T)
Pt = Tm @ Pobl @ np.linalg.inv(Tm); out["E4_linear_remix_conjugated_projection_err"] = float(np.abs(Zb + (Zs - Zb) @ Pt.T - Zt).max())
nl = lambda Z: np.tanh(Z @ Tm.T); Zb, Zs, Zt = nl(phi(V[i], Rr[i])), nl(phi(V[j], Rr[j])), nl(phi(V[j], Rr[i]))
out["E4_nonlinear_remix_best_linear_map_rel_residual"] = best_proj_residual(Zb, Zs, Zt)
json.dump(out, open(os.path.join(HERE, "..", "logs", "audit_x1_extra.json"), "w"), indent=1); print(json.dumps(out, indent=1))
