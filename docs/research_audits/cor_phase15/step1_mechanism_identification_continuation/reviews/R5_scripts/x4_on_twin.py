"""R5: apply the X4 criterion (equal 'blind' margin with block-1 MLP ablated, opposite decisions) to the
relation-free twin of C' (src/relation_free_equivalent.py), in which no unit reads two sources."""
import sys, copy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2] / "src"; sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath
from relation_free_equivalent import relation_free
from cprime import closed_form
mR, emb = relation_free()
mB = copy.deepcopy(mR); mB.layers[0].mlp_on = False
full = lambda a, b: cbmath.margin(cbmath.forward(mR, emb(a, b), 3))
blind = lambda a, b: cbmath.margin(cbmath.forward(mB, emb(a, b), 3))
for base, conf in [((1, 1), (3, -0.238)), ((1.5, 0.5), (3, -0.2)), ((0.6, 0.6), (2, -0.634)), ((2, 2), (3, -2.5))]:
    print(f"twin  base {base}: m_blind={blind(*base):+.4f} m={full(*base):+.4f} C'm={closed_form(*base)['m']:+.4f} | "
          f"conf {conf}: m_blind={blind(*conf):+.4f} m={full(*conf):+.4f} C'm={closed_form(*conf)['m']:+.4f}")
rng = np.random.default_rng(0); vals = [blind(*rng.uniform(-3, 3, 2)) for _ in range(200)]
print("twin m_blind range over 200 random inputs:", min(vals), max(vals))
