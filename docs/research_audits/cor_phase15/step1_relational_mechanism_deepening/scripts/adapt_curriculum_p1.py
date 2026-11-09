"""EXPLORATORY (decided after adapt_s0 failed Q3-E1/E2; labelled post hoc): ADAPT fine-tuned from the atlas T1 s0 checkpoint
(curriculum, as the atlas used for T4), 3000 steps, same optimiser/schedule.  Output checkpoints/adapt_fromT1p1_s1.{npz,json}."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import train3 as TR, ckpt as C
p0 = C.load(os.path.join(HERE, "..", "..", "step1_latent_reliability", "checkpoints", "main", "T1_M_K3_s1.npz"))
out = os.path.join(HERE, "..", "checkpoints"); lf = open(os.path.join(out, "adapt_fromT1p1_s1.log"), "w")
p, h, dt = TR.train("adapt", 1, 3000, log=lambda s: (lf.write(s + "\n"), lf.flush()), p0=p0, rng_offset=777)
C.save(os.path.join(out, "adapt_fromT1p1_s1.npz"), p); json.dump(dict(hist=h, seconds=dt), open(os.path.join(out, "adapt_fromT1p1_s1.json"), "w"))
