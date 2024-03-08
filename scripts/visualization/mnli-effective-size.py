"""
Plot the effective size of the pruned model vs steps, for MNLI - both ID and OOD
"""

import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--orig-accuracy-path", 
        "-io", 
        default="data/eval/mnli-acc_over_ft.json", help="Accuracy of original model over fine-tuning steps"
    )
    parser.add_argument(
        "--pruned-accuracy-path", 
        "-ip", 
        default="data/eval/mnli-sparsity-grid-acc.json",
        help="Accuracy of pruned subnetworks as a dict of seed -> step -> sparsity -> case -> acc"
    )
    parser.add_argument(
        "--out-path", 
        "-o", 
        default="data/plots/mnli-effective-size.pdf",
        help="File to save the plot to"
    )
    parser.add_argument(
        "--margin", 
        "-m", 
        type=float, 
        default=0.03,
        help="Margin for considering a subnetwork as faithful to the original model"
    )
    parser.add_argument(
        "--show-acc", 
        "-sa", 
        action="store_true",
        help="Show the accuracy of the original model"
    )
    parser.add_argument(
        "--domain", 
        "-d", 
        default="both",
        help="id, ood or both"
    )
    parser.add_argument(
        "--cases", 
        "-c", 
        nargs="+", 
        default=[
            "validation_matched", # ID
            "validation_mismatched",  # ID
            "lexical_overlap_ln_preposition", # OOD
            "lexical_overlap_ln_subject-object_swap", # OOD
            "constituent_cn_embedded_under_if", # OOD
            "constituent_cn_embedded_under_verb", # OOD
        ],
        help="Cases to include in the plot"
    )
    parser.add_argument(
        "--add-legend", 
        "-l", 
        action="store_true", 
        help="Add a legend to the plot"
    )
    parser.add_argument("--lower-y", "-ly", type=float, default=22, help="Lower bound for y-axis")
    parser.add_argument("--upper-y", "-uy", type=float, default=101, help="Upper bound for y-axis")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    orig_accuracies = json.load(open(args.orig_accuracy_path, "r"))
    pruned_accuracies = json.load(open(args.pruned_accuracy_path, "r"))

    steps = list(orig_accuracies.keys())
    steps = sorted([int(s) for s in steps])

    if args.domain == "id":
        args.cases = [c for c in args.cases if c.startswith("validation")]
    elif args.domain == "ood":
        args.cases = [c for c in args.cases if not c.startswith("validation")]

    plt.clf()
    sns.set_theme()

    seeds = list(pruned_accuracies.keys())

    fig, ax = plt.subplots(figsize=(6.4, 3))
    palette = iter(sns.color_palette("tab10"))

    orig_model_avg_acc_id = []
    orig_model_avg_acc_ood = []

    if args.show_acc:
        ax2 = ax.twinx()
        for step in steps:
            step_str = str(step)
            orig_id_avg = np.mean([orig_accuracies[step_str][case] for case in args.cases if case.startswith("validation")]).item()
            orig_ood_avg = np.mean([orig_accuracies[step_str][case] for case in args.cases if not case.startswith("validation")]).item()
            orig_model_avg_acc_id.append(orig_id_avg*100)
            orig_model_avg_acc_ood.append(orig_ood_avg*100)

        sns.lineplot(x=steps, y=orig_model_avg_acc_id, label="ID", ax=ax2, color='k', alpha=1, linewidth=1, linestyle='--')
        sns.lineplot(x=steps, y=orig_model_avg_acc_ood, label="OOD", ax=ax2, color='blue', alpha=1, linewidth=1, linestyle='--')

    df_json = {
        "Seed" : [],
        "Steps" : [],
        "Effective Size (%)" : [],
    }
    
    df_json_id = {
        "Seed" : [],
        "Steps" : [],
        "Effective Size (%)" : [],
    }
    
    df_json_ood = {
        "Seed" : [],
        "Steps" : [],
        "Effective Size (%)" : [],
    }

    for seed in seeds:
        sparsities = []
        effective_size = []
        for step in steps:
            step_str = str(step)
            orig_acc_values = [orig_accuracies[step_str][case] for case in args.cases]
            orig_acc_mean = np.mean(np.array(orig_acc_values)).item()

            step_sparsities = list(pruned_accuracies[seed][step_str].keys())
            step_sparsities =  sorted([int(s) for s in step_sparsities], reverse=True)

            min_sparsity = None

            for sparsity in step_sparsities:
                sparsity_str = str(sparsity)
                pruned_acc_values = [pruned_accuracies[seed][step_str][sparsity_str][case] for case in args.cases]
                pruned_acc_mean = np.mean(np.array(pruned_acc_values)).item()
                
                if pruned_acc_mean > orig_acc_mean - args.margin:
                    min_sparsity = sparsity
                    break
        
            if min_sparsity is None:
                min_sparsity = 0
            
            df_json["Seed"].append(seed)
            df_json["Steps"].append(step)
            df_json["Effective Size (%)"].append(100-min_sparsity)
            
            min_sparsity_id = 0
            orig_acc_values = [orig_accuracies[step_str][case] for case in args.cases if case.startswith("validation")]
            orig_acc_mean = np.mean(np.array(orig_acc_values)).item()
            
            for sparsity in step_sparsities:
                sparsity_str = str(sparsity)
                pruned_acc_values = [pruned_accuracies[seed][step_str][sparsity_str][case] for case in args.cases if case.startswith("validation")]
                pruned_acc_mean = np.mean(np.array(pruned_acc_values)).item()
                
                if pruned_acc_mean > orig_acc_mean - args.margin:
                    min_sparsity_id = sparsity
                    break
                
            df_json_id["Seed"].append(seed)
            df_json_id["Steps"].append(step)
            df_json_id["Effective Size (%)"].append(100-min_sparsity_id)
            
            min_sparsity_ood = 0
            orig_acc_values = [orig_accuracies[step_str][case] for case in args.cases if not case.startswith("validation")]
            orig_acc_mean = np.mean(np.array(orig_acc_values)).item()
            
            for sparsity in step_sparsities:
                sparsity_str = str(sparsity)
                pruned_acc_values = [pruned_accuracies[seed][step_str][sparsity_str][case] for case in args.cases if not case.startswith("validation")]
                pruned_acc_mean = np.mean(np.array(pruned_acc_values)).item()
                
                if pruned_acc_mean > orig_acc_mean - args.margin:
                    min_sparsity_ood = sparsity
                    break
                
            df_json_ood["Seed"].append(seed)
            df_json_ood["Steps"].append(step)
            df_json_ood["Effective Size (%)"].append(100-min_sparsity_ood)            
    
    df = pd.DataFrame(df_json)
    df_id = pd.DataFrame(df_json_id)
    df_ood = pd.DataFrame(df_json_ood)
    
    sns.lineplot(df, x="Steps", y="Effective Size (%)", ax=ax, color=next(palette), linewidth=2.5, label=f"ID + OOD", marker="o")
    sns.lineplot(df_id, x="Steps", y="Effective Size (%)", ax=ax, color=next(palette), linewidth=2.5, label=f"ID", marker="v")
    ax = sns.lineplot(df_ood, x="Steps", y="Effective Size (%)", ax=ax, color=next(palette), linewidth=2.5, label=f"OOD", marker="^")

    ax.set_xticks(steps[::2])
    ax.set_xticklabels(steps[::2], fontsize=12)
    ax.set_xlabel("Steps", fontsize=15)
    ax.set_ylabel("Effective Size (%)", fontsize=15)
    
    if args.add_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.14, 1.16), ncol=3, prop={'size': 12}, frameon=False, markerscale=2)
    
    ax.set_ylim(args.lower_y, args.upper_y)
        
    if args.show_acc:
        ax2.set_ylabel("Accuracy (%)", fontsize=15)
        ax2.legend("")
        ax2.set_ylim(args.lower_y, args.upper_y)

    if not args.add_legend:
        plt.legend([],[], frameon=False)

    ax.figure.savefig(args.out_path, bbox_inches="tight")

if __name__ == "__main__":
    main()