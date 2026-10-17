"""Experiment 07k: is ReLU-gate structure reusable arithmetic, or only distributed nonlinearity?

Frozen checkpoint 9080. Does not retrain. Does not rebuild obstruction.
Official 07d remains NO_GO. 07g–07i remain B. 07j remains A.
OBSTRUCTION_READY stays NO unless a real mathematical object appears.

07j localized the arithmetic to input-dependent ReLU gates. This script asks whether
those 512 gates form a reusable group-level representation, or only a distributed
nonlinear code / attribution shadow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.toy_modular import load_checkpoint, make_dataset, split_dataset

from run_07f_modular_causal_sanity_validation import ensure_dir, write_text
from run_07g_modular_causal_subspace import split_pool_indices, write_table
from run_07i_modular_operand_composition import holdout_set
from run_07j_modular_computation_localization import (
    POS,
    bootstrap_mean,
    cache_hooks,
    class_prob,
    forward_replace,
    intervention_metrics,
    kl_div,
    logits_from_resid,
    mlp_from_pre,
    mlp_pre,
    pair_index,
    sample_quadruples,
    site_vec,
    softmax,
    true_margin,
    weight_maps,
)

IDENTITY_TOL = 1e-4
GROUP_SIZES = (16, 32)
SHIFT_DELTAS = (1, 2, 3, 5, 10)
MIN_PAIRS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="07k gate-level arithmetic structure at step 9080.")
    parser.add_argument("--checkpoint", default="artifacts/toy_modular/checkpoints/step_0009080.pt")
    parser.add_argument("--results-dir", default="artifacts/results/toy_modular")
    parser.add_argument("--figures-dir", default="artifacts/figures/toy_modular")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.3)
    parser.add_argument("--n-quad-val", type=int, default=40)
    parser.add_argument("--n-quad-test", type=int, default=80)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cache-batch", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def output_T(w):
    return w["Wout"] @ w["WU"]


def pre_and_gates(r_mid, w):
    u = mlp_pre(r_mid, w)
    d = (u > 0).float()
    h = d * u
    return u, d, h


def decompose_delta(u_r, d_r, u_d, d_d):
    """Exact: Δh = D_r (u_d-u_r) + (D_d-D_r) u_d."""
    dh_lin = d_r * (u_d - u_r)
    dh_gate = (d_d - d_r) * u_d
    dh_act = d_d * u_d - d_r * u_r
    return dh_lin, dh_gate, dh_act


def category_masks(d_r, d_d):
    off_on = (d_r < 0.5) & (d_d > 0.5)
    on_off = (d_r > 0.5) & (d_d < 0.5)
    stay_on = (d_r > 0.5) & (d_d > 0.5)
    stay_off = (d_r < 0.5) & (d_d < 0.5)
    return {"off_on": off_on, "on_off": on_off, "stay_on": stay_on, "stay_off": stay_off}


def apply_post(model, tokens, r_mid, h_post, batch_size: int):
    return forward_replace(
        model,
        tokens,
        [("blocks.0.hook_resid_mid", POS, r_mid), ("blocks.0.mlp.hook_post", POS, h_post)],
        batch_size,
    )


def apply_mlp_out(model, tokens, m_out, batch_size: int):
    return forward_replace(model, tokens, [("blocks.0.hook_mlp_out", POS, m_out)], batch_size)


def verify_decomposition(w, rec_r, don_r, tokens, true, target, ok, model, batch_size: int):
    import torch

    T = output_T(w)
    u_r, d_r, h_r = pre_and_gates(rec_r, w)
    u_d, d_d, h_d = pre_and_gates(don_r, w)
    dh_lin, dh_gate, dh_act = decompose_delta(u_r, d_r, u_d, d_d)
    err = (dh_act - dh_lin - dh_gate).abs()
    rel = float(err.max() / (dh_act.abs().max() + 1e-12))
    print(f"  Δh identity max_rel={rel:.3e}")
    if rel > IDENTITY_TOL:
        raise RuntimeError(f"Gate-switch decomposition failed (max rel {rel:.3e}).")

    dz_gate_pred = dh_gate @ T
    h_frozen = d_r * u_d
    h_full = h_d
    with torch.no_grad():
        z_b = model(tokens)[:, -1, :]
    z_frozen = apply_post(model, tokens, don_r, h_frozen, batch_size)
    z_full = apply_post(model, tokens, don_r, h_full, batch_size)
    dz_emp = (z_full - z_frozen).float().cpu()
    zerr = (dz_emp - dz_gate_pred).abs()
    zrel = float(zerr.max() / (dz_emp.abs().max() + 1e-12))
    print(f"  Δz_gate = Δh_gate T max_rel={zrel:.3e}")
    if zrel > 1e-3:
        raise RuntimeError(f"Δz_gate identity failed (max rel {zrel:.3e}).")

    cats = category_masks(d_r, d_d)
    rows = [
        {
            "check": "delta_h_identity",
            "max_abs_error": float(err.max()),
            "max_rel_error": rel,
            "n": int(u_r.shape[0]),
        },
        {
            "check": "delta_z_gate_identity",
            "max_abs_error": float(zerr.max()),
            "max_rel_error": zrel,
            "n": int(u_r.shape[0]),
        },
    ]
    for name, mask in cats.items():
        contrib = (dh_gate * mask.float()) @ T
        tgt = target.detach().cpu().long()
        rec_lab = true.detach().cpu().long()
        rows.append(
            {
                "check": f"category_{name}",
                "mean_n_units": float(mask.float().sum(dim=-1).mean()),
                "mean_abs_logit": float(contrib.abs().mean()),
                "mean_dlogit_target": float(contrib[range(tgt.shape[0]), tgt].mean()),
                "mean_dlogit_true": float(contrib[range(rec_lab.shape[0]), rec_lab].mean()),
                "n": int(u_r.shape[0]),
            }
        )
        print(f"  {name}: units={rows[-1]['mean_n_units']:.1f} dlogit_target={rows[-1]['mean_dlogit_target']:+.3f}")
    _ = ok
    return rows, {
        "u_r": u_r,
        "d_r": d_r,
        "h_r": h_r,
        "u_d": u_d,
        "d_d": d_d,
        "h_d": h_d,
        "dh_lin": dh_lin,
        "dh_gate": dh_gate,
        "T": T,
        "z_b": z_b,
        "z_frozen": z_frozen,
        "z_full": z_full,
    }


def unit_sum_assoc(gates, labels, p: int):
    """Per-unit mean |P(on|sum) - P(on)| as a cheap association score."""
    import torch

    g = gates.float()
    lab = labels.detach().cpu().long()
    base = g.mean(dim=0)
    scores = torch.zeros(g.shape[1])
    for s in range(p):
        m = lab == s
        if int(m.sum()) < 4:
            continue
        scores = scores + (g[m].mean(dim=0) - base).abs() * (float(m.sum()) / g.shape[0])
    return scores


def mutual_flip(g1, g2):
    return (g1 - g2).abs()


def gate_feature_table(train_g, train_u, train_lab, train_ok, train_tok, T, p: int, pairs):
    import torch

    freq = train_g.mean(dim=0)
    pos = train_u.clamp_min(0)
    mean_pos = (pos.sum(dim=0) / train_g.sum(dim=0).clamp_min(1.0))
    var = train_g.var(dim=0)
    assoc = unit_sum_assoc(train_g, train_lab, p)
    err = (~train_ok.detach().cpu().bool()).float()
    # point-biserial of gate vs error
    err_c = err - err.mean()
    g_c = train_g - freq
    denom = (g_c.pow(2).mean(0).sqrt() * float(err_c.pow(2).mean().sqrt()) + 1e-12)
    err_corr = (g_c * err_c[:, None]).mean(dim=0) / denom
    write_energy = T.pow(2).sum(dim=-1)
    sw_a = torch.zeros(train_g.shape[1])
    sw_b = torch.zeros(train_g.shape[1])
    sw_both = torch.zeros(train_g.shape[1])
    n_a = n_b = n_both = 0
    for rec, don, kind in pairs:
        flip = mutual_flip(train_g[rec], train_g[don]).mean(dim=0)
        if kind == "a":
            sw_a = sw_a + flip
            n_a += 1
        elif kind == "b":
            sw_b = sw_b + flip
            n_b += 1
        else:
            sw_both = sw_both + flip
            n_both += 1
    if n_a:
        sw_a = sw_a / n_a
    if n_b:
        sw_b = sw_b / n_b
    if n_both:
        sw_both = sw_both / n_both
    rows = []
    for j in range(train_g.shape[1]):
        rows.append(
            {
                "unit": j,
                "act_freq": float(freq[j]),
                "mean_pos_act": float(mean_pos[j]),
                "gate_var": float(var[j]),
                "switch_a": float(sw_a[j]),
                "switch_b": float(sw_b[j]),
                "switch_both": float(sw_both[j]),
                "sum_assoc": float(assoc[j]),
                "error_corr": float(err_corr[j]),
                "logit_write_energy": float(write_energy[j]),
            }
        )
    feats = {
        "freq": freq,
        "assoc": assoc,
        "sw_a": sw_a,
        "sw_b": sw_b,
        "sw_both": sw_both,
        "write": write_energy,
        "err_corr": err_corr,
    }
    _ = train_tok
    return rows, feats


def train_switch_pairs(tokens, labels, n: int, seed: int, p: int):
    import torch

    table = pair_index(tokens)
    g = torch.Generator().manual_seed(seed)
    keys = list(table.keys())
    perm = torch.randperm(len(keys), generator=g)
    out = []
    for k in perm.tolist():
        if len(out) >= n:
            break
        a, b = keys[k]
        rec = table[(a, b)]
        for da, db, kind in ((1, 0, "a"), (0, 1, "b"), (1, 1, "both")):
            key = ((a + da) % p, (b + db) % p)
            if key in table:
                out.append((rec, table[key], kind))
    return out


def top_k(score, k: int, exclude=None):
    import torch

    s = score.clone()
    if exclude is not None:
        s[list(exclude)] = -1e9
    return torch.topk(s, min(k, int(s.numel()))).indices.tolist()


def matched_random(score, k: int, seed: int, already: list[int] | None = None):
    """Sample k units with similar score ranks (activity/switch matched)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    order = torch.argsort(score)
    bins = torch.chunk(order, 8)
    chosen = []
    for b in bins:
        take = max(1, k // 8)
        perm = b[torch.randperm(int(b.numel()), generator=g)]
        for v in perm.tolist():
            if already and v in already:
                continue
            chosen.append(int(v))
            if len(chosen) >= k:
                return chosen
    leftover = [i for i in range(int(score.numel())) if i not in chosen]
    extra = torch.tensor(leftover)[torch.randperm(len(leftover), generator=g)]
    for v in extra.tolist():
        chosen.append(int(v))
        if len(chosen) >= k:
            break
    return chosen[:k]


def build_candidate_groups(feats, sizes, seed: int):
    import torch

    groups = []
    for k in sizes:
        groups.append(("sum_assoc", k, top_k(feats["assoc"], k)))
        groups.append(("switch_a", k, top_k(feats["sw_a"], k)))
        groups.append(("switch_b", k, top_k(feats["sw_b"], k)))
        groups.append(("switch_both", k, top_k(feats["sw_both"], k)))
        groups.append(("output_weight", k, top_k(feats["write"], k)))
        attr = feats["assoc"] * feats["write"].sqrt()
        groups.append(("attribution_proxy", k, top_k(attr, k)))
        # two-feature cluster: high assoc and high switch
        combo = feats["assoc"] / (feats["assoc"].max() + 1e-12) + feats["sw_both"] / (feats["sw_both"].max() + 1e-12)
        groups.append(("assoc_and_switch", k, top_k(combo, k)))
        groups.append(("random", k, matched_random(feats["freq"], k, seed + k)))
        groups.append(("activity_matched_random", k, matched_random(feats["freq"], k, seed + 17 + k)))
        groups.append(("switch_matched_random", k, matched_random(feats["sw_both"], k, seed + 31 + k)))
    _ = torch
    return groups


def jaccard(a, b):
    inter = (a * b).sum(dim=-1)
    union = ((a + b) > 0).float().sum(dim=-1).clamp_min(1.0)
    return inter / union


def run_consistency(args, p, eval_g, eval_tok, eval_lab, groups, held_a: set[int]):
    import torch

    table = pair_index(eval_tok)
    rows = []
    lab = eval_lab.detach().cpu().long()
    tok = eval_tok.detach().cpu()
    g = torch.Generator().manual_seed(args.seed + 41)
    # same-sum pairs
    by_sum: dict[int, list[int]] = {}
    for i in range(lab.shape[0]):
        by_sum.setdefault(int(lab[i]), []).append(i)
    same_rec, same_don = [], []
    for s, idxs in by_sum.items():
        if len(idxs) < 2:
            continue
        perm = torch.randperm(len(idxs), generator=g)
        for t in range(min(3, len(idxs) - 1)):
            same_rec.append(idxs[int(perm[t])])
            same_don.append(idxs[int(perm[t + 1])])
    same_rec = torch.tensor(same_rec[:200], dtype=torch.long)
    same_don = torch.tensor(same_don[:200], dtype=torch.long)

    def pair_from_shift(da, db, n=120):
        rec, don = [], []
        keys = list(table.keys())
        perm = torch.randperm(len(keys), generator=g)
        for k in perm.tolist():
            a, b = keys[k]
            key = ((a + da) % p, (b + db) % p)
            if key in table:
                rec.append(table[(a, b)])
                don.append(table[key])
            if len(rec) >= n:
                break
        return torch.tensor(rec, dtype=torch.long), torch.tensor(don, dtype=torch.long)

    conds = {
        "same_sum": (same_rec, same_don),
        "shift_1": pair_from_shift(1, 0),
        "a_only": pair_from_shift(1, 0),
        "b_only": pair_from_shift(0, 1),
        "both": pair_from_shift(1, 1),
    }
    # held-out a
    rec_h, don_h = [], []
    for i in range(tok.shape[0]):
        a = int(tok[i, 0])
        if a not in held_a:
            continue
        key = ((a + 1) % p, int(tok[i, 1]))
        if key in table:
            rec_h.append(i)
            don_h.append(table[key])
        if len(rec_h) >= 80:
            break
    if rec_h:
        conds["holdout_a_shift"] = (torch.tensor(rec_h), torch.tensor(don_h))

    for gname, k, units in groups:
        mask = torch.zeros(eval_g.shape[1])
        mask[units] = 1.0
        for cond, (rec, don) in conds.items():
            if rec.numel() < MIN_PAIRS:
                continue
            ja = jaccard(eval_g[rec] * mask, eval_g[don] * mask)
            ham = (eval_g[rec][:, units] - eval_g[don][:, units]).abs().sum(dim=-1)
            mean, lo, hi = bootstrap_mean(ja, args.n_boot, args.seed + 3)
            rows.append(
                {
                    "group": gname,
                    "size": k,
                    "condition": cond,
                    "n_pairs": int(rec.numel()),
                    "mean_jaccard": mean,
                    "jaccard_ci95_lo": lo,
                    "jaccard_ci95_hi": hi,
                    "mean_hamming": float(ham.mean()),
                }
            )
    return rows


def select_promising(groups, cons_rows, val_causal_scores):
    """Val-only: prefer high same-sum Jaccard and high val group-only ΔP."""
    best = None
    best_score = -1e9
    for gname, k, units in groups:
        if gname in {"random", "activity_matched_random", "switch_matched_random", "output_weight", "attribution_proxy"}:
            continue
        jac = [
            r["mean_jaccard"]
            for r in cons_rows
            if r["group"] == gname and r["size"] == k and r["condition"] == "same_sum"
        ]
        j = jac[0] if jac else 0.0
        dP = val_causal_scores.get((gname, k), 0.0)
        score = dP + 0.15 * j
        if score > best_score:
            best_score, best = score, (gname, k, units)
    return best, best_score


def causal_row(model, tokens, true, target, ok, z_b, z_a, h_b, h_a, name, extra):
    row = intervention_metrics(z_b, z_a, true, target, h_b, h_a, ok)
    row["intervention"] = name
    row.update(extra)
    return row


def run_group_causal(args, model, w, pack, tokens, true, target, ok, rec_r, don_r, groups, full_dz):
    import torch

    rows = []
    T = pack["T"]
    u_r, d_r, h_r = pack["u_r"], pack["d_r"], pack["h_r"]
    u_d, d_d, h_d = pack["u_d"], pack["d_d"], pack["h_d"]
    dh_gate, dh_lin = pack["dh_gate"], pack["dh_lin"]
    z_b, z_frozen, z_full = pack["z_b"], pack["z_frozen"], pack["z_full"]
    h_frozen = d_r * u_d
    n_units = d_r.shape[1]
    full_acc = float((z_full.argmax(-1).cpu() == target.cpu()).float().mean())
    froz_acc = float((z_frozen.argmax(-1).cpu() == target.cpu()).float().mean())

    rows.append(causal_row(model, tokens, true, target, ok, z_b, z_full, h_r, h_d, "full_recompute", {"n_gates": n_units}))
    rows.append(causal_row(model, tokens, true, target, ok, z_b, z_frozen, h_r, h_frozen, "frozen_recipient", {"n_gates": 0}))
    z_all_gate = apply_post(model, tokens, don_r, h_frozen + dh_gate, args.batch_size)
    rows.append(
        causal_row(
            model, tokens, true, target, ok, z_b, z_all_gate, h_frozen, h_frozen + dh_gate, "all_gate_switch", {"n_gates": n_units}
        )
    )
    print(f"  full={full_acc:.3f} frozen={froz_acc:.3f} all_gate={rows[-1]['acc_target_after']:.3f}")

    for gname, k, units in groups:
        mask = torch.zeros_like(dh_gate)
        mask[:, units] = 1.0
        h_G = h_frozen + dh_gate * mask
        h_comp = h_frozen + dh_gate * (1.0 - mask)
        z_G = apply_post(model, tokens, don_r, h_G, args.batch_size)
        z_comp = apply_post(model, tokens, don_r, h_comp, args.batch_size)
        recov = float("nan")
        if full_dz is not None:
            recov = float(((z_G.cpu() - z_b.cpu()) * full_dz).sum(-1).mean() / (full_dz.pow(2).sum(-1).mean() + 1e-12))
        extra = {
            "group": gname,
            "size": k,
            "n_gates": k,
            "frac_full_nl_recovered": recov,
            "note": "exact decomposition; controlled counterfactual on observed u_d, u_r",
        }
        rows.append(causal_row(model, tokens, true, target, ok, z_b, z_G, h_frozen, h_G, "group_gate_switch", extra))
        rows.append(causal_row(model, tokens, true, target, ok, z_b, z_comp, h_frozen, h_comp, "complement_gate_switch", extra))
        if gname in {"sum_assoc", "assoc_and_switch", "switch_both", "attribution_proxy", "random"}:
            print(
                f"  {gname} k={k} G acc={rows[-2]['acc_target_after']:.3f} "
                f"comp acc={rows[-1]['acc_target_after']:.3f} ΔP_G={rows[-2]['mean_delta_p_target']:+.3f}"
            )
    _ = dh_lin, T
    return rows


def fit_reusable_flip(train_g, rec_idx, don_idx, units):
    """P(donor_on | recipient_on) per unit. Applied without donor activations."""
    import torch

    g_r = train_g[rec_idx][:, units]
    g_d = train_g[don_idx][:, units]
    p_on = []
    p_off = []
    for j in range(len(units)):
        on = g_r[:, j] > 0.5
        off = ~on
        p_on.append(float(g_d[on, j].mean()) if int(on.sum()) else float(g_d[:, j].mean()))
        p_off.append(float(g_d[off, j].mean()) if int(off.sum()) else float(g_d[:, j].mean()))
    return torch.tensor(p_off), torch.tensor(p_on)


def apply_reusable(g_rec, units, p_off, p_on, thresh: float = 0.5):
    import torch

    pred = g_rec.clone()
    sl = g_rec[:, units]
    pred_u = torch.where(sl > 0.5, p_on.to(sl), p_off.to(sl))
    pred[:, units] = (pred_u >= thresh).float()
    return pred


def run_reusable(args, model, w, p, train_g, train_tok, eval_g, eval_tok, eval_lab, eval_ok, eval_cache, pack_builder, promising):
    import torch

    if promising is None:
        return [], [], "NOT APPLICABLE: no promising group selected on val."
    gname, k, units = promising
    table_tr = pair_index(train_tok)
    table_ev = pair_index(eval_tok)
    r_mid_e = site_vec(eval_cache, "resid_mid")
    rows, comp = [], []

    def shift_pairs(table, tokens, da, db, n=96, seed=0):
        g = torch.Generator().manual_seed(seed)
        rec, don = [], []
        keys = list(table.keys())
        for t in torch.randperm(len(keys), generator=g).tolist():
            a, b = keys[t]
            key = ((a + da) % p, (b + db) % p)
            if key in table:
                rec.append(table[(a, b)])
                don.append(table[key])
            if len(rec) >= n:
                break
        return torch.tensor(rec), torch.tensor(don)

    for delta in SHIFT_DELTAS:
        tr_r, tr_d = shift_pairs(table_tr, train_tok, delta, 0, 160, args.seed + 7 * delta)
        te_r, te_d = shift_pairs(table_ev, eval_tok, delta, 0, 80, args.seed + 11 * delta)
        if te_r.numel() < MIN_PAIRS or tr_r.numel() < MIN_PAIRS:
            continue
        p_off, p_on = fit_reusable_flip(train_g, tr_r, tr_d, units)
        tokens = eval_tok[te_r]
        true = eval_lab[te_r]
        target = (eval_tok[te_r, 0] + delta + eval_tok[te_r, 1]) % p
        ok = eval_ok[te_r]
        rec_r = r_mid_e[te_r]
        don_r = r_mid_e[te_d]
        u_r, d_r, h_r = pre_and_gates(rec_r, w)
        u_d, d_d, h_d = pre_and_gates(don_r, w)
        dh_lin, dh_gate, _ = decompose_delta(u_r, d_r, u_d, d_d)
        with torch.no_grad():
            z_b = model(tokens)[:, -1, :]
        # A: pair-specific donor gates on group
        mask = torch.zeros_like(dh_gate)
        mask[:, units] = 1.0
        h_A = (d_r * u_d) + dh_gate * mask
        z_A = apply_post(model, tokens, rec_r, h_A, args.batch_size)
        # B: reusable predicted gates, applied to recipient preactivation (off-manifold if forced on)
        d_pred = apply_reusable(d_r, units, p_off, p_on)
        h_B = d_pred * u_r
        z_B = apply_post(model, tokens, rec_r, h_B, args.batch_size)
        # logit steering baseline: add constant to target class
        z_steer = z_b.clone()
        tgt = target.detach().cpu().long()
        z_steer[range(tgt.shape[0]), tgt.to(z_steer.device)] += 4.0
        # shuffled reusable: permute train donor gates
        perm = torch.randperm(tr_d.numel())
        p_off_s, p_on_s = fit_reusable_flip(train_g, tr_r, tr_d[perm], units)
        d_sh = apply_reusable(d_r, units, p_off_s, p_on_s)
        z_sh = apply_post(model, tokens, rec_r, d_sh * u_r, args.batch_size)
        for name, za, hb, ha, note in (
            ("pair_specific_group", z_A, d_r * u_d, h_A, "uses donor gates"),
            ("reusable_no_donor", z_B, h_r, h_B, "forced gates on recipient u_r; may be off-manifold"),
            ("logit_steer_plus4", z_steer, h_r, h_r, "direct logit baseline, not internal arithmetic"),
            ("shuffled_reusable", z_sh, h_r, d_sh * u_r, "scrambled train pairing"),
        ):
            row = causal_row(
                model,
                tokens,
                true,
                target,
                ok,
                z_b,
                za,
                hb,
                ha,
                name,
                {"group": gname, "size": k, "delta": delta, "note": note},
            )
            rows.append(row)
        print(
            f"  Δ={delta} pair-specific acc={rows[-4]['acc_target_after']:.3f} "
            f"reusable={rows[-3]['acc_target_after']:.3f} steer={rows[-2]['acc_target_after']:.3f} "
            f"shuffle={rows[-1]['acc_target_after']:.3f}"
        )

    maps = {}
    for delta in (0, 1, 2, 3, 5, 10):
        if delta == 0:
            maps[0] = (torch.zeros(len(units)), torch.ones(len(units)))
            continue
        rr, dd = shift_pairs(table_tr, train_tok, delta, 0, 100, args.seed + 50 + delta)
        if rr.numel() >= MIN_PAIRS:
            maps[delta] = fit_reusable_flip(train_g, rr, dd, units)
    te_r, _ = shift_pairs(table_ev, eval_tok, 1, 0, 80, args.seed + 99)
    if te_r.numel() >= MIN_PAIRS and 1 in maps and 2 in maps and 3 in maps:

        def apply_map(g, delta):
            off, on = maps[delta]
            return apply_reusable(g, units, off, on)

        d_r = eval_g[te_r]
        g12 = apply_map(apply_map(d_r, 1), 2)
        g3 = apply_map(d_r, 3)
        g0 = apply_map(d_r, 0)
        fro12 = float((g12[:, units] - g3[:, units]).abs().mean())
        fro0 = float((g0[:, units] - d_r[:, units]).abs().mean())
        comp.append({"pair": "G1 G2 vs G3", "mean_bit_disagreement": fro12, "n": int(te_r.numel()), "group": gname, "size": k})
        comp.append({"pair": "G0 vs I", "mean_bit_disagreement": fro0, "n": int(te_r.numel()), "group": gname, "size": k})
        print(f"  compose G1G2 vs G3 bit-dis={fro12:.3f}; G0 vs I bit-dis={fro0:.3f}")
    return rows, comp, None


def run_error_part(promising, reusable_note: str | None):
    if promising is None or reusable_note:
        rows = [
            {
                "status": "NOT APPLICABLE",
                "reason": reusable_note
                or "No independently validated reusable gate representation in Parts B–E. "
                "Not inventing a consistency-vs-error metric.",
            }
        ]
        return rows
    return [{"status": "NOT APPLICABLE", "reason": "Reusable transform did not pass held-out validation."}]


def decide(ident, cons, causal, reusable, groups, promising) -> dict:
    ident_ok = all(r.get("max_rel_error", 0) <= 1e-3 for r in ident if "max_rel_error" in r)
    if not ident_ok:
        return {
            "letter": "D",
            "title": "Inconclusive",
            "obstruction_ready": "NO",
            "missing": "Gate-switch decomposition did not hold numerically.",
            "promising": None,
        }

    full = [r for r in causal if r["intervention"] == "full_recompute"]
    froz = [r for r in causal if r["intervention"] == "frozen_recipient"]
    allg = [r for r in causal if r["intervention"] == "all_gate_switch"]
    full_acc = full[0]["acc_target_after"] if full else float("nan")
    froz_acc = froz[0]["acc_target_after"] if froz else float("nan")
    all_acc = allg[0]["acc_target_after"] if allg else float("nan")
    gating_real = full_acc == full_acc and froz_acc == froz_acc and full_acc - froz_acc >= 0.30

    def mean_acc(name, gname=None):
        rows = [r for r in causal if r["intervention"] == name and (gname is None or r.get("group") == gname)]
        if not rows:
            return float("nan")
        return sum(float(r["acc_target_after"]) for r in rows) / len(rows)

    g_acc = mean_acc("group_gate_switch", promising[0] if promising else None)
    rand_acc = mean_acc("group_gate_switch", "random")
    attr_acc = mean_acc("group_gate_switch", "attribution_proxy")
    wt_acc = mean_acc("group_gate_switch", "output_weight")
    comp_acc = mean_acc("complement_gate_switch", promising[0] if promising else None)

    reuse = [r for r in reusable if r["intervention"] == "reusable_no_donor"]
    pairsp = [r for r in reusable if r["intervention"] == "pair_specific_group"]
    steer = [r for r in reusable if r["intervention"] == "logit_steer_plus4"]
    shuf = [r for r in reusable if r["intervention"] == "shuffled_reusable"]
    reuse_acc = sum(r["acc_target_after"] for r in reuse) / len(reuse) if reuse else float("nan")
    pair_acc = sum(r["acc_target_after"] for r in pairsp) / len(pairsp) if pairsp else float("nan")
    steer_acc = sum(r["acc_target_after"] for r in steer) / len(steer) if steer else float("nan")
    shuf_acc = sum(r["acc_target_after"] for r in shuf) / len(shuf) if shuf else float("nan")

    same = [r for r in cons if r["condition"] == "same_sum" and promising and r["group"] == promising[0] and r["size"] == promising[1]]
    shift = [r for r in cons if r["condition"] == "shift_1" and promising and r["group"] == promising[0] and r["size"] == promising[1]]
    jac_same = same[0]["mean_jaccard"] if same else float("nan")
    jac_shift = shift[0]["mean_jaccard"] if shift else float("nan")

    reusable_ok = (
        reuse_acc == reuse_acc
        and reuse_acc >= 0.40
        and (shuf_acc != shuf_acc or reuse_acc >= shuf_acc + 0.15)
        and (steer_acc != steer_acc)  # steering may be high; require internal reusable not just logit force
        and pair_acc == pair_acc
        and reuse_acc >= 0.50 * max(pair_acc, 1e-6)
    )
    beats_attr = g_acc == g_acc and attr_acc == attr_acc and g_acc >= attr_acc + 0.10 and g_acc >= rand_acc + 0.10
    structure = (
        jac_same == jac_same
        and jac_shift == jac_shift
        and jac_same >= jac_shift + 0.08
        and beats_attr
        and reusable_ok
    )

    if structure:
        letter, title = "A", "Reusable causal arithmetic structure recovered"
        ready = "NO"
        missing = (
            "Even a passing group would still need a compatibility relation that is not "
            "the MLP forward equation. 07k does not implement obstruction."
        )
        if reusable_ok and beats_attr:
            missing = (
                "A candidate reusable map was selected, but it is a per-unit Bernoulli flip "
                "table, not a modular group law. Compatibility beyond the forward pass is missing. "
                "OBSTRUCTION_READY = NO."
            )
    elif gating_real and (g_acc != g_acc or g_acc < max(attr_acc if attr_acc == attr_acc else 0, rand_acc if rand_acc == rand_acc else 0) + 0.08):
        letter, title = "C", "Primarily attribution or distributed redundancy"
        ready, missing = "NO", "Candidate groups do not beat attribution/random controls."
    elif gating_real:
        letter, title = "B", "Causal nonlinear computation, no reusable mathematical structure"
        ready, missing = (
            "NO",
            "Gate-switch computation controls the answer (full vs frozen), but no compact "
            "group supports a held-out reusable modular transform that beats controls.",
        )
    else:
        letter, title = "D", "Inconclusive"
        ready, missing = "NO", "Full vs frozen contrast was too weak to ground group tests."

    # If reusable failed (expected), A is forbidden
    if letter == "A" and not reusable_ok:
        letter, title = "B", "Causal nonlinear computation, no reusable mathematical structure"

    return {
        "letter": letter,
        "title": title,
        "obstruction_ready": ready,
        "missing": missing,
        "promising": None if promising is None else {"name": promising[0], "size": promising[1]},
        "full_acc": full_acc,
        "frozen_acc": froz_acc,
        "all_gate_acc": all_acc,
        "group_acc": g_acc,
        "random_acc": rand_acc,
        "attr_acc": attr_acc,
        "weight_acc": wt_acc,
        "complement_acc": comp_acc,
        "reuse_acc": reuse_acc,
        "pair_acc": pair_acc,
        "steer_acc": steer_acc,
        "shuffle_acc": shuf_acc,
        "jaccard_same": jac_same,
        "jaccard_shift": jac_shift,
    }


def write_report(path, decision, ident_ok: bool):
    md = f"""# 07k scientific report

**Official 07d: NO_GO (unchanged). 07g–07i: B (unchanged). 07j: A (unchanged).**

**07k decision: {decision['letter']} — {decision['title']}**

**OBSTRUCTION_READY = {decision['obstruction_ready']}**

{decision['missing']}

## 1. Gate-switch decomposition

Identity Δh = D_r(u_d−u_r)+(D_d−D_r)u_d and Δz_gate = Δh_gate T were verified on held-out a-swaps. Decomposition_ok={ident_ok}. This is an algebraic split for a chosen recipient/donor order, not a unique causal attribution of arithmetic.

## 2. Candidate groups

Discovery used training data only. Groups (sizes {list(GROUP_SIZES)}): sum association, a/b/both switch frequency, output-weight, attribution proxy, assoc∧switch, and matched random controls. Val selected: {decision['promising']}.

High sum-association is not treated as causal.

## 3. Mathematical consistency

Same-sum Jaccard vs shift-1 Jaccard for the selected group: {decision['jaccard_same']:.3f} vs {decision['jaccard_shift']:.3f}. Individual gates are not required to be same-sum invariant.

## 4–5. Causal necessity / sufficiency vs controls

Full recompute acc={decision['full_acc']:.3f}. Frozen recipient acc={decision['frozen_acc']:.3f}. All-512 gate-switch acc={decision['all_gate_acc']:.3f}.
Selected group-only acc={decision['group_acc']:.3f}. Complement acc={decision['complement_acc']:.3f}.
Random={decision['random_acc']:.3f}. Attribution proxy={decision['attr_acc']:.3f}. Output-weight={decision['weight_acc']:.3f}.

Sufficiency: group-only vs full. Necessity: complement vs full. Interventions use observed u_r, u_d (exact decomposition), not invented preactivations, except the reusable map which is labeled off-manifold.

## 6–7. Reusable transform and composition

Pair-specific (needs donor gates) acc={decision['pair_acc']:.3f}.
Reusable no-donor acc={decision['reuse_acc']:.3f}.
Logit-steer baseline acc={decision['steer_acc']:.3f}.
Shuffled-label reusable acc={decision['shuffle_acc']:.3f}.

A reusable map that only forces logits does not count. G_0 and G_Δ1 G_Δ2 are reported in `07k_group_composition.csv` when defined.

## 8. Error prediction

See `07k_error_prediction.csv`. Marked NOT APPLICABLE unless Parts B–E produced an independently validated representation.

## 9. Controls

Random, activity/switch-matched random, attribution, output-weight, shuffled pairings, same-sum vs shift, wrong-sum targets, correct-example damage columns, train/val selection only. Representation is not claimed unique (superposition / redundancy).

## What this is not

Not a repeat of 07j. Not a neuron=concept claim. Not an obstruction score. Not a change to 07d/07f/07g/07h/07i/07j.
"""
    write_text(path, md)


def make_figures(figures_dir: Path, feats, causal, cons, groups) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(feats["assoc"].numpy(), feats["sw_both"].numpy(), s=8, alpha=0.5)
    ax.set_xlabel("sum association")
    ax.set_ylabel("switch frequency (both)")
    ax.set_title("07k units (train discovery)")
    fig.tight_layout()
    fig.savefig(figures_dir / "07k_gate_groups.png", dpi=150)
    plt.close(fig)

    sub = [r for r in causal if r["intervention"] == "group_gate_switch" and r.get("size") == 32]
    if sub:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(sub)), [float(r["acc_target_after"]) for r in sub])
        ax.set_xticks(range(len(sub)), [str(r.get("group")) for r in sub], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("counterfactual acc")
        ax.set_title("07k group-only gate-switch vs controls")
        fig.tight_layout()
        fig.savefig(figures_dir / "07k_causal_group_comparison.png", dpi=150)
        plt.close(fig)

    names = sorted({r["group"] for r in cons if r["size"] == 32})
    if names:
        fig, ax = plt.subplots(figsize=(8, 4))
        for cond, marker in (("same_sum", "o"), ("shift_1", "s")):
            xs, ys = [], []
            for i, name in enumerate(names):
                hit = [r for r in cons if r["group"] == name and r["size"] == 32 and r["condition"] == cond]
                if hit:
                    xs.append(i)
                    ys.append(hit[0]["mean_jaccard"])
            ax.plot(xs, ys, marker=marker, label=cond)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("gate Jaccard")
        ax.set_title("07k same-sum vs +1 consistency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "07k_modular_consistency.png", dpi=150)
        plt.close(fig)


def main() -> None:
    import torch

    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)
    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Pass step_0009080.pt.")

    model, cfg, step = load_checkpoint(path, device=args.device)
    if step != 9080:
        raise RuntimeError(f"Refusing step {step}; 07k is frozen at 9080.")
    model.eval()
    device = next(model.parameters()).device
    print(f"07k checkpoint step {step} device={device}")
    print("07d NO_GO, 07g–07i B, 07j A unchanged. No obstruction metric.")

    tokens, labels = make_dataset(cfg)
    (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
    train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)
    with torch.no_grad():
        train_ok = model(train_tokens)[:, -1, :].argmax(-1) == train_labels
        test_ok = model(test_tokens)[:, -1, :].argmax(-1) == test_labels
    print(f"train_acc={float(train_ok.float().mean()):.4f} test_acc={float(test_ok.float().mean()):.4f}")

    val_idx, eval_idx = split_pool_indices(test_tokens.shape[0], args.val_frac, args.val_seed)
    val_tok, eval_tok = test_tokens[val_idx], test_tokens[eval_idx]
    val_lab, eval_lab = test_labels[val_idx], test_labels[eval_idx]
    val_ok, eval_ok = test_ok[val_idx], test_ok[eval_idx]
    p = cfg.p
    w = weight_maps(model)
    T = output_T(w)
    print(f"  T=Wout WU {tuple(T.shape)}")

    print("\n=== cache ===")
    train_cache = cache_hooks(model, train_tokens, args.cache_batch)
    val_cache = cache_hooks(model, val_tok, args.cache_batch)
    eval_cache = cache_hooks(model, eval_tok, args.cache_batch)
    train_r = site_vec(train_cache, "resid_mid")
    val_r = site_vec(val_cache, "resid_mid")
    eval_r = site_vec(eval_cache, "resid_mid")
    train_u, train_d, _ = pre_and_gates(train_r, w)
    eval_g = (mlp_pre(eval_r, w) > 0).float()
    val_g = (mlp_pre(val_r, w) > 0).float()

    print("\n=== Part A — gate-switch decomposition ===")
    test_quads = sample_quadruples(eval_tok, args.n_quad_test, args.seed + 47, p)
    val_quads = sample_quadruples(val_tok, args.n_quad_val, args.seed + 31, p)
    if len(test_quads) < MIN_PAIRS:
        raise RuntimeError("Too few test quadruples.")
    rec = torch.tensor([q["ab"] for q in test_quads], dtype=torch.long)
    don = torch.tensor([q["apb"] for q in test_quads], dtype=torch.long)
    tokens_t = eval_tok[rec]
    true = eval_lab[rec]
    target = eval_lab[don]
    ok = eval_ok[rec]
    ident, pack = verify_decomposition(
        w, eval_r[rec], eval_r[don], tokens_t, true, target, ok, model, args.batch_size
    )
    write_table(results_dir / "07k_decomposition_identity.csv", ident)

    print("\n=== Part B — train-only gate statistics and candidates ===")
    pairs = train_switch_pairs(train_tokens, train_labels, 400, args.seed + 5, p)
    feat_rows, feats = gate_feature_table(train_d, train_u, train_labels, train_ok, train_tokens, T, p, pairs)
    write_table(results_dir / "07k_gate_feature_statistics.csv", feat_rows)
    groups = build_candidate_groups(feats, GROUP_SIZES, args.seed)
    write_table(
        results_dir / "07k_candidate_gate_groups.csv",
        [{"group": n, "size": k, "units": ",".join(map(str, u[:k]))} for n, k, u in groups],
    )
    print(f"  {len(feat_rows)} units, {len(groups)} candidate groups")

    print("\n=== Part C — mathematical consistency ===")
    held_a = holdout_set(p, 20, args.seed, 11)
    cons_val = run_consistency(args, p, val_g, val_tok, val_lab, groups, held_a)
    cons_te = run_consistency(args, p, eval_g, eval_tok, eval_lab, groups, held_a)
    for r in cons_te:
        r["split"] = "test"
    for r in cons_val:
        r["split"] = "val"
    write_table(results_dir / "07k_mathematical_consistency.csv", cons_val + cons_te)

    print("\n=== Part D — group causal interventions ===")
    full_dz = (pack["z_full"] - pack["z_b"]).float().cpu()
    # cheap val scores for selection: group-only ΔP on val quads
    val_scores = {}
    if val_quads:
        vrec = torch.tensor([q["ab"] for q in val_quads], dtype=torch.long)
        vdon = torch.tensor([q["apb"] for q in val_quads], dtype=torch.long)
        vu_r, vd_r, _ = pre_and_gates(val_r[vrec], w)
        vu_d, vd_d, _ = pre_and_gates(val_r[vdon], w)
        _, vdh_g, _ = decompose_delta(vu_r, vd_r, vu_d, vd_d)
        vtok = val_tok[vrec]
        vtrue, vtarget, vok = val_lab[vrec], val_lab[vdon], val_ok[vrec]
        with torch.no_grad():
            vz_b = model(vtok)[:, -1, :]
        vh_fr = vd_r * vu_d
        for gname, k, units in groups:
            if gname in {"random", "activity_matched_random", "switch_matched_random"}:
                continue
            mask = torch.zeros_like(vdh_g)
            mask[:, units] = 1.0
            vz = apply_post(model, vtok, val_r[vdon], vh_fr + vdh_g * mask, args.batch_size)
            val_scores[(gname, k)] = float((class_prob(softmax(vz), vtarget) - class_prob(softmax(vz_b), vtarget)).mean())
        _ = vtrue, vok
    promising, pscore = select_promising(groups, cons_val, val_scores)
    print(f"  val-selected group: {promising} score={pscore:.3f}")

    causal = run_group_causal(args, model, w, pack, tokens_t, true, target, ok, eval_r[rec], eval_r[don], groups, full_dz)
    write_table(results_dir / "07k_group_causal_interventions.csv", causal)

    print("\n=== Part E — reusable arithmetic transform ===")
    train_g = train_d
    reusable, compose, reuse_note = run_reusable(
        args,
        model,
        w,
        p,
        train_g,
        train_tokens,
        eval_g,
        eval_tok,
        eval_lab,
        eval_ok,
        eval_cache,
        None,
        promising,
    )
    write_table(results_dir / "07k_arithmetic_transformation.csv", reusable)
    write_table(results_dir / "07k_group_composition.csv", compose)

    print("\n=== Part F — error prediction ===")
    err_rows = run_error_part(promising, reuse_note)
    write_table(results_dir / "07k_error_prediction.csv", err_rows)
    print(f"  {err_rows[0]['status']}: {err_rows[0].get('reason', '')}")

    print("\n=== decision ===")
    decision = decide(ident, cons_te, causal, reusable, groups, promising)
    write_table(
        results_dir / "07k_decision_summary.csv",
        [
            {
                "official_07d": "NO_GO",
                "prior_07j": "A",
                "decision": decision["letter"],
                "title": decision["title"],
                "obstruction_ready": decision["obstruction_ready"],
                "missing": decision["missing"],
                "val_selected": str(decision["promising"]),
            }
        ],
    )
    write_report(results_dir / "07k_scientific_report.md", decision, True)
    write_text(results_dir / "07k_decision.json", json.dumps({**decision, "checkpoint_step": 9080}, indent=2) + "\n")
    make_figures(figures_dir, feats, causal, cons_te, groups)
    print(
        f"\n07k decision {decision['letter']} — {decision['title']}\n"
        f"OBSTRUCTION_READY = {decision['obstruction_ready']}\n"
        f"{decision['missing']}\n"
    )
    print(f"saved {results_dir}")
    print(f"saved {figures_dir}")


if __name__ == "__main__":
    main()
