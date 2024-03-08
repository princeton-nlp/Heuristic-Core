import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in-path", 
        "-i", 
        default="data/eval/mnli-acc_vs_sparsity.json",
        help="Accuracy of pruned subnetworks as a dict of seed -> sparsity -> acc"
    )
    parser.add_argument(
        "--out-path", 
        "-o", 
        default="data/plots/mnli-acc_vs_sparsity.pdf",
        help="File to save the plot to"
    )
    parser.add_argument("--id-cases", "-ic", nargs="+", default=[
        "validation_matched",
        "validation_mismatched",
    ])
    parser.add_argument("--ood-cases", "-oc", nargs="+", default=[
        "lexical_overlap_ln_preposition",
        "lexical_overlap_ln_subject-object_swap",
        "constituent_cn_embedded_under_if",
        "constituent_cn_embedded_under_verb",
    ])
    parser.add_argument(
        "--seed-min", 
        "-sn", 
        default=42, 
        type=int
    )
    parser.add_argument(
        "--seed-max",
        "-sN",
        default=44,
        type=int
    )
    parser.add_argument(
        "--add-legend",
        "-l",
        action="store_true",
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    accuracies = json.load(open(args.in_path))

    plt.clf()
    sns.set_theme()

    seeds = list(range(args.seed_min, args.seed_max + 1))
    palette = iter(sns.color_palette("tab10"))
    
    df_json_id = {
        "Seed": [],
        "Sparsity (%)": [],
        "Accuracy (%)": [],
    }
    df_json_ood = {
        "Seed": [],
        "Sparsity (%)": [],
        "Accuracy (%)": [],
    }
    
    sparsities = list(accuracies[str(seeds[0])].keys())
    sparsities = sorted([int(sparsity) for sparsity in sparsities])

    for seed in seeds:
        seed = str(seed)

        for sparsity in accuracies[seed].keys():
            sparsity_int = int(sparsity)

            id_avg = 100.0*np.mean(np.array([accuracies[seed][sparsity][case] for case in args.id_cases])).item()
            ood_avg = 100.0*np.mean(np.array([accuracies[seed][sparsity][case] for case in args.ood_cases])).item()
            
            df_json_id["Seed"].append(seed)
            df_json_id["Sparsity (%)"].append(sparsity_int)
            df_json_id["Accuracy (%)"].append(id_avg)
            
            df_json_ood["Seed"].append(seed)
            df_json_ood["Sparsity (%)"].append(sparsity_int)
            df_json_ood["Accuracy (%)"].append(ood_avg)

    df_id = pd.DataFrame(df_json_id)
    df_ood = pd.DataFrame(df_json_ood)
    
    fig, ax = plt.subplots(figsize=(6.4, 3))
    next(palette)   # We just do this to keep a consistent color scheme with the other plots in the paper

    sns.lineplot(df_id, x="Sparsity (%)", y="Accuracy (%)", label=f"ID", linewidth=2.5, color=next(palette), marker="v", ax=ax)
    ax = sns.lineplot(df_ood, x="Sparsity (%)", y="Accuracy (%)", label=f"OOD", linewidth=2.5, color=next(palette), marker="^", ax=ax)
    
    ax.set_ylim(0, 100)
    
    ax.set_xticks(sparsities[::2])
    ax.set_xticklabels(sparsities[::2], fontsize=12)

    ax.set_xlabel("Sparsity (%)", fontsize=15)
    ax.set_ylabel("Avg. Accuracy (%)", fontsize=15)

    if args.add_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(0.3, 1.16), ncol=6, prop={'size': 12}, frameon=False, markerscale=2)
    else:
        plt.legend([], [], frameon=False)

    ax.figure.savefig(args.out_path, bbox_inches="tight")

if __name__ == "__main__":
    main()