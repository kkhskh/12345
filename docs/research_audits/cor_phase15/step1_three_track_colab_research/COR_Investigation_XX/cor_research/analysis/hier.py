"""Hierarchical statistics with the pretrained PARENT as the independent unit and fine-tuning pairs nested within parents.
  hier_mean    grand mean = mean of parent means (each parent weighted equally), two-stage bootstrap CI (resample parents, then
               pairs within each resampled parent), 10 000 replicates, fixed seed (reproducible)
  var_comp     one-way random-effects variance components (unbalanced method of moments): between-parent tau^2 and within-parent
               sigma^2, ICC = tau^2 / (tau^2 + sigma^2)
  prop         per-parent proportions and the parent-level pooled proportion with the same two-stage bootstrap
Pairs from the same parent are NEVER treated as independent replicates of the parent-level effect."""
import numpy as np


def _clean(d): return {p: np.asarray([v for v in vs if v is not None and np.isfinite(v)], float) for p, vs in d.items() if len([v for v in vs if v is not None and np.isfinite(v)])}


def hier_mean(d, B=10000, seed=0, stat=np.mean):
    d = _clean(d); ps = sorted(d)
    if not ps: return dict(n_parents=0, n_pairs=0, mean=None, ci95=None)
    pm = np.array([stat(d[p]) for p in ps]); rng = np.random.default_rng(seed); bs = np.empty(B)
    for b in range(B):
        sel = rng.integers(0, len(ps), len(ps)); bs[b] = np.mean([stat(d[ps[i]][rng.integers(0, len(d[ps[i]]), len(d[ps[i]]))]) for i in sel])
    return dict(n_parents=len(ps), n_pairs=int(sum(len(d[p]) for p in ps)), mean=float(pm.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                parent_means={p: float(m) for p, m in zip(ps, pm)}, sd_between_parent_means=float(pm.std(ddof=1)) if len(ps) > 1 else None)


def var_comp(d):
    d = _clean(d); ps = sorted(d); k = len(ps)
    if k < 2: return dict(n_parents=k, tau2=None, sigma2=None, icc=None)
    n = np.array([len(d[p]) for p in ps]); N = n.sum(); gm = np.concatenate([d[p] for p in ps]).mean(); m = np.array([d[p].mean() for p in ps])
    ssb = float((n * (m - gm) ** 2).sum()); ssw = float(sum(((d[p] - d[p].mean()) ** 2).sum() for p in ps))
    msb = ssb / (k - 1); msw = ssw / (N - k) if N > k else float("nan"); n0 = (N - (n ** 2).sum() / N) / (k - 1)
    tau2 = max((msb - msw) / n0, 0.0) if np.isfinite(msw) else None
    return dict(n_parents=k, n_pairs=int(N), tau2=tau2, sigma2=msw if np.isfinite(msw) else None, icc=(tau2 / (tau2 + msw) if tau2 is not None and np.isfinite(msw) and tau2 + msw > 0 else None))


def prop(d, B=10000, seed=0):
    r = hier_mean({p: [float(x) for x in vs] for p, vs in d.items()}, B, seed)
    r["per_parent"] = {p: dict(k=int(sum(vs)), n=len(vs)) for p, vs in d.items() if len(vs)}; return r


def sign_consistency(parent_means, tol_exceptions=1):
    v = [x for x in parent_means.values() if x is not None]
    if not v: return dict(consistent=False, n=0)
    pos = sum(x > 0 for x in v); neg = sum(x < 0 for x in v); maj = 1 if pos >= neg else -1
    return dict(consistent=bool(len(v) - max(pos, neg) <= tol_exceptions), sign=maj, n_pos=pos, n_neg=neg, n=len(v))
