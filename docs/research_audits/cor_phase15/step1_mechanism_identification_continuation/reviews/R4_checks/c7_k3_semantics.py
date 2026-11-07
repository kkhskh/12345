"""R4 check 7: K=3 table in SOURCE_SEMANTICS §2, and observation-level vs latent-level agreement."""
import numpy as np
K, n = 3, 50000; rng = np.random.default_rng(21)
# replicate sample_k but keep Y_s and mu (sample_k discards them)
Y = rng.integers(0, K, n)
def src():
    hit = rng.random(n) < 0.6; Ys = np.where(hit, Y, rng.integers(0, K, n)); mu = rng.uniform(0, 3, n)[:, None]
    return mu*np.eye(K)[Ys] + rng.normal(0, 1, (n, K)), Ys
zA, YA = src(); zB, YB = src()
ag = zA.argmax(1) == zB.argmax(1); dot = (zA*zB).sum(1)
c = lambda z: z - z.mean(1, keepdims=True); cdot = (c(zA)*c(zB)).sum(1)
print("P(dot<0|agree)=%.3f P(dot>0|disagree)=%.3f acc sign dot=%.3f acc centred=%.3f" % (
    (dot[ag] < 0).mean(), (dot[~ag] > 0).mean(), ((dot > 0) == ag).mean(), ((cdot > 0) == ag).mean()))
print("P(argmax z_s = Y_s)=%.3f ; P(observed agree != latent claim agree [Y_A=Y_B])=%.3f" % (
    (zA.argmax(1) == YA).mean(), (ag != (YA == YB)).mean()))
