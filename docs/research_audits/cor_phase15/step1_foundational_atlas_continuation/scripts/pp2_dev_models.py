"""Model registry shared by the PP2 scripts."""
M = {f"tf_s{s}": ("tf", f"checkpoints/posthoc_curriculum/T4_M_K3_s{s}_fromT1.npz") for s in [0, 1, 2]}
M.update({f"tf_s{s}": ("tf", f"checkpoints/replication/T4_M_K3_s{s}_fromT1.npz") for s in [10, 11, 12, 13]})
M.update({f"ds_s{s}": ("ds", f"checkpoints/main/T4_M_K3_s{s}_deepsets.npz") for s in [0, 1, 2]})
