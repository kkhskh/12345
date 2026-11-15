"""Statistics helpers for Investigation XII (imported by score_xii.py).
Episode-level: nonparametric bootstrap over episodes WITHIN one model (B = 2000, percentile 95% CI, seed fixed).
Ratio statistics (TRUE/CTRL designs are independent episode samples): resample each design independently.
Paired designs (same episodes, two conditions): resample episode indices jointly.
Parent-level: the independent unit is the pretrained parent.  Reported: per-parent estimates, mean, min/max, and a bootstrap-over-parents
95% CI of the mean (with n = 4-5 parents this interval is wide and is NOT a population-level inference; it is reported for transparency)."""
import numpy as np
B = 2000
def ci_mean(x, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed); idx = r.integers(0, len(x), (B, len(x))); m = x[idx].mean(1)
    return dict(est=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(len(x)))
def ci_ratio(x, y, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float); r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (B, len(x)))].mean(1) / y[r.integers(0, len(y), (B, len(y)))].mean(1)
    return dict(est=float(x.mean() / y.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))
def ci_frac(b, seed=0): return ci_mean(np.asarray(b, float), seed)
def parent_level(vals, seed=0):
    v = np.asarray(list(vals), float); r = np.random.default_rng(seed); m = v[r.integers(0, len(v), (B, len(v)))].mean(1)
    return dict(per_parent=v.tolist(), mean=float(v.mean()), min=float(v.min()), max=float(v.max()), boot_lo=float(np.percentile(m, 2.5)), boot_hi=float(np.percentile(m, 97.5)), n_parents=int(len(v)))
