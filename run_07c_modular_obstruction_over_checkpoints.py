"""Experiment 0B step 3: track latent obstruction, abstraction strength, and accuracy over training.

This is the monitoring stage. The central hypothesis from the updated plan is
A_cyc(t) up  =>  O_y(t) down  =>  m_c(t) up, i.e. representational emergence precedes or
accompanies behavioural emergence. Learned transports are compared against identity and random
transports so that any obstruction signal cannot be attributed to the edge structure alone.
"""

from __future__ import annotations

import argparse
import csv

from src.factual_obstruction_eval import auc_scores
from src.toy_modular import (
    abstraction_strength,
    cache_site_activations,
    candidate_margin,
    chain_edges,
    decode_sum_from_latent,
    final_logits,
    fit_latent_germs,
    fit_transports,
    germ_values,
    hardest_incorrect,
    identity_transports,
    key_frequencies,
    latent_obstruction,
    latent_targets,
    load_checkpoint,
    make_dataset,
    make_toy_sites,
    random_transports,
    split_dataset,
)

from run_07b_modular_latent_representation import select_checkpoints, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track latent obstruction across training checkpoints.")
    parser.add_argument("--checkpoint-dir", default="artifacts/toy_modular/checkpoints")
    parser.add_argument("--output", default="artifacts/toy_modular/obstruction_over_checkpoints.csv")
    parser.add_argument("--output-examples", default="artifacts/toy_modular/latent_obstruction_examples.csv")
    parser.add_argument("--n-freqs", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 means use all")
    parser.add_argument("--max-example-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    import torch
    from tqdm.auto import tqdm

    args = parse_args()
    rows: list[dict] = []
    example_rows: list[dict] = []
    paths = select_checkpoints(args.checkpoint_dir, args.max_checkpoints)
    final_path = paths[-1]

    for path in tqdm(paths, desc="checkpoints"):
        model, cfg, step = load_checkpoint(path, device=args.device)
        tokens, labels = make_dataset(cfg)
        (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
        device = next(model.parameters()).device
        train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
        test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)

        freqs = key_frequencies(model, cfg.p, top_k=args.n_freqs)
        a_cyc = abstraction_strength(model, cfg.p, top_k=args.n_freqs)
        train_latent = latent_targets(train_tokens, cfg.p, freqs).cpu()
        latent_dim = train_latent.shape[1]

        sites = make_toy_sites(cfg)
        edges = chain_edges(sites)
        train_acts = cache_site_activations(model, train_tokens, sites)
        test_acts = cache_site_activations(model, test_tokens, sites)

        germs = fit_latent_germs(train_acts, train_latent, sites, args.ridge)
        germ_train = germ_values(train_acts, germs, sites)
        germ_test = germ_values(test_acts, germs, sites)

        transport_sets = {
            "learned": fit_transports(germ_train, edges, args.ridge),
            "identity": identity_transports(edges, latent_dim),
            "random": random_transports(edges, latent_dim, args.seed),
        }

        with torch.no_grad():
            test_logits = final_logits(model, test_tokens)
            train_logits = final_logits(model, train_tokens)
        test_pred = test_logits.argmax(dim=-1)
        test_correct = (test_pred == test_labels).cpu()
        test_margin = candidate_margin(
            test_logits, test_labels, hardest_incorrect(test_logits, test_labels)
        ).cpu()

        row = {
            "step": step,
            "abstraction_strength": a_cyc,
            "key_frequencies": " ".join(str(k) for k in freqs),
            "train_acc": float((train_logits.argmax(dim=-1) == train_labels).float().mean()),
            "test_acc": float(test_correct.float().mean()),
            "mean_test_margin": float(test_margin.mean()),
            "n_test": int(test_tokens.shape[0]),
        }

        # Out-of-sample decodability of the latent state at the final residual site.
        readout = germ_test[sites[-1].site_id]
        true_test_sum = ((test_tokens[:, 0] + test_tokens[:, 1]) % cfg.p).cpu()
        decoded = decode_sum_from_latent(readout, cfg.p, freqs)
        row["decode_acc_test_resid_post"] = float((decoded == true_test_sum).float().mean())

        for name, transports in transport_sets.items():
            O_train = latent_obstruction(germ_train, transports, edges)
            O_test = latent_obstruction(germ_test, transports, edges)
            o_test = O_test["normalized"]
            correct_vals = o_test[test_correct].tolist()
            wrong_vals = o_test[~test_correct].tolist()
            row[f"O_train_{name}"] = float(O_train["normalized"].mean())
            row[f"O_test_{name}"] = float(o_test.mean())
            row[f"O_test_correct_{name}"] = mean(correct_vals)
            row[f"O_test_incorrect_{name}"] = mean(wrong_vals)
            # Does high obstruction flag the model's own errors?
            labels_err = [0 if c else 1 for c in test_correct.tolist()]
            row[f"auc_O_detects_error_{name}"] = (
                auc_scores(labels_err, o_test.tolist()) if 0 < sum(labels_err) < len(labels_err) else float("nan")
            )
        rows.append(row)
        write_csv(args.output, rows)

        if path == final_path:
            o_learned = latent_obstruction(germ_test, transport_sets["learned"], edges)["normalized"]
            limit = min(args.max_example_rows, test_tokens.shape[0])
            for i in range(limit):
                example_rows.append(
                    {
                        "step": step,
                        "a": int(test_tokens[i, 0]),
                        "b": int(test_tokens[i, 1]),
                        "true_sum": int(true_test_sum[i]),
                        "predicted": int(test_pred[i]),
                        "correct": bool(test_correct[i]),
                        "margin": float(test_margin[i]),
                        "obstruction_normalized": float(o_learned[i]),
                        "decoded_latent_sum": int(decoded[i]),
                        "latent_state_correct": bool(decoded[i] == true_test_sum[i]),
                    }
                )

    write_csv(args.output, rows)
    write_csv(args.output_examples, example_rows)

    print("\nstep      A_cyc   test_acc  O_learned  O_identity  O_random  AUC_err(learned)")
    for row in rows:
        print(
            f"{row['step']:>8}  {row['abstraction_strength']:.4f}  {row['test_acc']:.4f}    "
            f"{row['O_test_learned']:.5f}    {row['O_test_identity']:.5f}   "
            f"{row['O_test_random']:.5f}   {row['auc_O_detects_error_learned']:.4f}"
        )
    print(f"\nsaved checkpoint metrics to {args.output}")
    print(f"saved final-checkpoint example rows to {args.output_examples}")


if __name__ == "__main__":
    main()
