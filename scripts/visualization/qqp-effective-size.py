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
        "--orig-accuracy-path", 
        "-io", 
        default="data/eval/qqp-acc_over_ft.json"
    )
    parser.add_argument(
        "--pruned-accuracy-path", 
        "-ip", 
        default="data/eval/qqp-sparsity-grid-acc.json"
    )
    parser.add_argument(
        "--out-path", 
        "-o", 
        default="data/plots/qqp-effective-size.pdf"
    )
    parser.add_argument(
        "--margin", 
        "-m", 
        type=float, 
        default=0.03
    )
    parser.add_argument(
        "--show-acc", 
        "-sa", 
        action="store_true"
    )
    parser.add_argument(
        "--cases", 
        "-c", 
        nargs="+", 
        default=[
            "qqp", # ID
            "paws-qqp_0" # OOD
        ]
    )
    parser.add_argument(
        "--add-legend", 
        "-l", 
        action="store_true", 
        help="Add a legend to the plot"
    )
    parser.add_argument("--lower-y", "-ly", type=float, default=30, help="Lower bound for y-axis")
    parser.add_argument("--upper-y", "-uy", type=float, default=101, help="Upper bound for y-axis")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    pruned_accuracies = json.load(open(args.pruned_accuracy_path, "r"))
    orig_accuracies = json.load(open(args.orig_accuracy_path, "r"))

    sample_seed = list(pruned_accuracies.keys())[0]
    steps = list(pruned_accuracies[sample_seed].keys())
    steps = sorted([int(s) for s in steps])

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
            orig_id_avg = orig_accuracies[step_str]["qqp"]
            orig_ood_avg = orig_accuracies[step_str]["paws-qqp_0"]
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
                if "paws-qqp_0" not in pruned_accuracies[seed][step_str][sparsity_str]:
                    continue
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
            orig_acc_value = orig_accuracies[step_str]["qqp"]
            
            for sparsity in step_sparsities:
                sparsity_str = str(sparsity)
                if "qqp" not in pruned_accuracies[seed][step_str][sparsity_str]:
                    continue
                pruned_acc_value = pruned_accuracies[seed][step_str][sparsity_str]["qqp"]
                if pruned_acc_value > orig_acc_value - args.margin:
                    min_sparsity_id = sparsity
                    break
            
            df_json_id["Seed"].append(seed)
            df_json_id["Steps"].append(step)
            df_json_id["Effective Size (%)"].append(100-min_sparsity_id)
            
            min_sparsity_ood = 0
            orig_acc_value = orig_accuracies[step_str]["paws-qqp_0"]
            
            for sparsity in step_sparsities:
                sparsity_str = str(sparsity)
                if "paws-qqp_0" not in pruned_accuracies[seed][step_str][sparsity_str]:
                    continue
                pruned_acc_value = pruned_accuracies[seed][step_str][sparsity_str]["paws-qqp_0"]
                if pruned_acc_value > orig_acc_value - args.margin:
                    min_sparsity_ood = sparsity
                    break
                
            df_json_ood["Seed"].append(seed)
            df_json_ood["Steps"].append(step)
            df_json_ood["Effective Size (%)"].append(100-min_sparsity_ood)
        
    df = pd.DataFrame(df_json)
    df_id = pd.DataFrame(df_json_id)
    df_ood = pd.DataFrame(df_json_ood)

    fig, ax = plt.subplots(figsize=(6.4, 3))

    sns.lineplot(df, x="Steps", y="Effective Size (%)", ax=ax, color=next(palette), linewidth=2.5, label=f"ID + OOD", marker="o")
    sns.lineplot(df_id, x="Steps", y="Effective Size (%)", ax=ax, color=next(palette), linewidth=2.5, label=f"ID", marker="v")
    ax = sns.lineplot(df_ood, x="Steps", y="Effective Size (%)", ax=ax, color=next(palette), linewidth=2.5, label=f"OOD", marker="^")

    ax.set_xticks(steps[::2])
    ax.set_xticklabels(steps[::2], fontsize=12)
    ax.set_xlabel("Steps", fontsize=15)
    ax.set_ylabel("Effective Size (%)", fontsize=15)
    ax.set_ylabel("")

    ax.set_ylim(args.lower_y, args.upper_y)
    if args.add_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.14, 1.16), ncol=3, prop={'size': 12}, frameon=False, markerscale=2)
    else:
        plt.legend([],[], frameon=False)

    if args.show_acc:
        ax2.set_ylabel("Accuracy (%)", fontsize=15)
        ax2.set_ylim(args.lower_y, args.upper_y)

    ax.figure.savefig(args.out_path, bbox_inches="tight")

if __name__ == "__main__":
    main()