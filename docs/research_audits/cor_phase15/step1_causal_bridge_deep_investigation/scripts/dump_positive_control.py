"""Print the exact parameter matrices of the positive control and its forward pass."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cbmath import forward, margin, positive_control, layernorm
np.set_printoptions(precision=10, suppress=True, linewidth=140)
m, (e1, e2, e3), embed = positive_control(final_ln=True)
L = m.layers[0]
print("eps =", m.eps)
print("e1 =", e1); print("e2 =", e2); print("e3 =", e3)
print("W_Q = W_K = 0 (4x4)")
print("W_V = I_4");
print("W_O = 3*sqrt(1/4+eps)*I_4, scalar =", L.WO[0][0, 0])
print("W_in (4x4) columns = sqrt(1/2+eps)*[e1+e2, -(e1+e2), e1-e2, -(e1-e2)]:\n", L.Win)
print("W_out (4x4) rows = [e3, e3, -e3, -e3]:\n", L.Wout)
print("biases = 0; LN gains 1, biases 0; final LN on; W_U columns [e3, -e3]:\n", m.WU)
for a in (-1.0, 1.0):
    for b in (-1.0, 1.0):
        tr = forward(m, embed(a, b), 2)
        print(f"\na={a:+.0f} b={b:+.0f}")
        print("  tokens X:\n", embed(a, b))
        print("  attention row at read:", tr.pattern[0][0][2])
        print("  read residual after attention (mid):", tr.mid[0][2], " = a e1 + b e2:", np.allclose(tr.mid[0][2], a*e1+b*e2))
        print("  MLP preactivations:", tr.mlp_pre[0][2])
        print("  MLP write on e3 (q):", tr.mlp_out[0][2] @ e3)
        print("  logits:", tr.logits, " margin:", margin(tr))
