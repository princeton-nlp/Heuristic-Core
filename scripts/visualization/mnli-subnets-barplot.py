import os
import json
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BIG_NAME_MAP = {
    "mnli-matched": "MNLI (matched)",
    "mnli-mismatched": "MNLI (mismatched)",
    "lexical_overlap_ln_subject-object_swap": "SO-Swap [LO]",
    "lexical_overlap_ln_preposition": "Prep [LO]",
    "constituent_cn_embedded_under_if": "Embed-If [C]",
    "constituent_cn_embedded_under_verb": "Embed-Verb [C]",
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--in-path", "-i", default="data/eval/mnli-sp50.json", help="File with the accuracies")
    parser.add_argument("--out-path", "-o", default="data/plots/mnli-sp50-bar.pdf", help="File to save the plot to")
    parser.add_argument(
        "--cases", 
        "-c", 
        nargs="+",
        default=[
            "mnli-matched",
            "mnli-mismatched",
            "lexical_overlap_ln_subject-object_swap",
            "lexical_overlap_ln_preposition",
            "constituent_cn_embedded_under_if",
            "constituent_cn_embedded_under_verb",
        ],
        help="Cases to include in the plot"
    )
    parser.add_argument("--add-legend", "-l", action="store_true", help="Add a legend to the plot")
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    name_map = {k: BIG_NAME_MAP[k] for k in args.cases}
    
    acc = json.load(open(args.in_path))
    seeds = acc["seed"]
    seeds = ["Full\nModel"] + ["#"+str(i) for i in range(1,len(seeds))] # The first one is the full model

    full_model = {}

    df_json = {
        "Seed": [],
        "Accuracy (%)": [],
        "Subcase": [],
    } 
    for k in name_map:
        full_model[name_map[k]] = acc[k][0]
        
        for i in range(len(seeds)):
            df_json["Seed"].append(seeds[i])
            df_json["Subcase"].append(name_map[k])
            df_json["Accuracy (%)"].append(100.0*acc[k][i])


    df_df = pd.DataFrame(df_json)

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(10, 3))
    palette = sns.color_palette("tab10")

    ax = sns.barplot(df_df, x="Seed", y="Accuracy (%)", hue="Subcase", palette=palette, ax=ax)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.axvline(0.45, color="black", linestyle="--", linewidth=1)

    if not args.add_legend:
        plt.legend([],[], frameon=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, fontsize=8, frameon=False)

    ax.figure.savefig(args.out_path, bbox_inches="tight")

if __name__ == '__main__':
    main()