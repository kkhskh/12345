"""R4 check 1: what the code's I_agree / D_sum criterion flags on synthetic unit functions,
versus a proper 'non-relational' class g(a+b)+f1(a)+f2(b) fitted by backfitting."""
import sys; sys.path.insert(0, "../../src")
import numpy as np
from learned_analysis import grid, dsum, additive_fit, r2
from scipy.special import erf
A, B, g = grid(); n = len(g); S = A + B
def code_crit(y):
    ds, means, idx = dsum(y, S); addf = additive_fit(y.reshape(n, n)).ravel(); sumf = means[idx]
    base = r2(np.column_stack([addf, sumf]), y)
    full = r2(np.column_stack([addf, sumf, A*B, np.sign(A*B)]), y)
    return ds, full - base
def backfit_resid_share(y, it=200):
    """1 - R^2 of best g(a+b) + f1(a) + f2(b) (alternating projections on the grid)."""
    yc = y - y.mean(); fit_s = np.zeros_like(y); fit_a = np.zeros_like(y)
    for _ in range(it):
        r = yc - fit_s; Z = r.reshape(n, n); fit_a = additive_fit(Z).ravel() - Z.mean()
        r = yc - fit_a; _, m, idx = dsum(r, S); fit_s = m[idx]
    res = yc - fit_a - fit_s
    return res.var() / y.var()
gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))
cases = {
 "ab": A*B,
 "(a+b)^2 - a^2 - b^2  (=2ab)": S**2 - A**2 - B**2,
 "sign(ab)": np.sign(A*B),
 "|a+b| + |a| + |b|  (sum-fn + additive)": np.abs(S) + np.abs(A) + np.abs(B),
 "gelu(a+b) + gelu(a)  (sum-fn + additive)": gelu(S) + gelu(A),
 "sign(ab)*sign(a+b)": np.sign(A*B) * np.sign(S),
 "ab*(a-b)  (odd interaction)": A*B*(A - B),
 "sign(ab)*(a-b)": np.sign(A*B) * (A - B),
 "1[ab>0]*sign(a)  (T3-type, = (sgn a+sgn b)/2)": (A*B > 0) * np.sign(A),
 "gelu(a-2b)  (ridge unit)": gelu(A - 2*B),
 "gelu(2a+b-1)": gelu(2*A + B - 1),
}
print(f"{'function':52s} D_sum  I_agree(code)  flagged  nonrel-resid(backfit)")
for k, y in cases.items():
    ds, ia = code_crit(y); br = backfit_resid_share(y)
    print(f"{k:52s} {ds:5.3f}  {ia:6.3f}        {str(ds>=.2 and ia>=.2):5s}    {br:6.3f}")
