"""EXPLORATORY: is each binary target sign-representable by a source-additive margin f(a) + g(b)?
Exact LP feasibility on a grid (axes excluded): find f_i, g_j with y_ij (f_i + g_j) >= 1.
Also T5b: logit_k = f_k(zA) + g_k(zB) + h_k(p) is checked by an explicit construction (doc)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantics import target_binary

def feasible(task, p=0.0, n=24):
    g = np.concatenate([-np.linspace(3, 0.05, n // 2), np.linspace(0.05, 3, n // 2)])
    A, B = np.meshgrid(g, g, indexing="ij")
    y, _ = target_binary(task, A.ravel(), B.ravel(), np.full(A.size, p))
    rows = []
    for idx, (i, j) in enumerate([(i, j) for i in range(n) for j in range(n)]):
        r = np.zeros(2 * n); r[i] = -y[idx]; r[n + j] = -y[idx]; rows.append(r)
    res = linprog(np.zeros(2 * n), A_ub=np.array(rows), b_ub=-np.ones(len(rows)), bounds=[(None, None)] * (2 * n), method="highs")
    return res.status == 0

def main():
    out = {}
    for task, p in [("T1", 0), ("T2", 0), ("T3", 1), ("T3", -1), ("T4", 0)]:
        for n in (12, 24, 48):
            out[f"{task} p={p} n={n}"] = bool(feasible(task, p, n))
            print(f"{task} p={p} grid {n}x{n}: additive sign-representable: {out[f'{task} p={p} n={n}']}")
    Path(__file__).resolve().parents[1].joinpath("logs", "additive_lp.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
