"""Independent check of censored per-question likelihood factors by enumerating the generator at one question (G=3, C=2)."""
import itertools, numpy as np
K, A = 3, 0.8
def p_vals(vals, y, g, H, beta):
    # vals: 6 values. reliable g: indep alpha; colluders H: with prob beta shared copy c ~ U(K) all 4 sources = c; else indep uniform.
    p = 1.0
    for s in (0, 1):
        p *= A if vals[2*g+s] == y else (1-A)/(K-1)
    cs = [vals[2*h+s] for h in H for s in (0, 1)]
    p *= beta * (1/K) * all(c == cs[0] for c in cs) + (1-beta) * K**-4
    return p
pr = A**2 + (1-A)**2/(K-1)
maxerr = 0
for beta in (1.0, 0.6):
    for g in range(3):
        H = tuple(h for h in range(3) if h != g)
        # q' marginal over Y uniform, censored to self-agreement bits
        tab = {}
        for vals in itertools.product(range(K), repeat=6):
            bits = tuple(vals[2*h] == vals[2*h+1] for h in range(3))
            tab[bits] = tab.get(bits, 0) + np.mean([p_vals(vals, y, g, H, beta) for y in range(K)])
        for bits, v in tab.items():
            a = bits[g]; script = (pr if a else 1-pr)
            b0, b1 = bits[H[0]], bits[H[1]]
            ind = (1/K if b0 else 1-1/K) * (1/K if b1 else 1-1/K)
            script *= beta * (b0 and b1) + (1-beta) * ind
            maxerr = max(maxerr, abs(v - script))
        # at q, joint values: script lik_q joint
        for vals in itertools.product(range(K), repeat=6):
            for y in range(K):
                p1 = lambda v: A if v == y else (1-A)/(K-1)
                cs = [vals[2*h+s] for h in H for s in (0, 1)]
                s = p1(vals[2*g]) * p1(vals[2*g+1]) * (beta * all(c == cs[0] for c in cs) / K + (1-beta) / K**4)
                maxerr = max(maxerr, abs(s - p_vals(vals, y, g, H, beta)))
print("max |enumerated - script factor| =", maxerr)
# sum-to-one check of bit table
