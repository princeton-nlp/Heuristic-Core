import sys
import json
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

BIG_NAME_MAP = {
    "mnli-matched": "MNLI\n(matched)",
    "mnli-mismatched": "MNLI\n(mis.)",
    "lexical_overlap_ln_subject-object_swap": "SO-Swap\n[LO]",
    "lexical_overlap_ln_preposition": "Prep\n[LO]",
    "constituent_cn_embedded_under_if": "Embed-\nIf [C]",
    "constituent_cn_embedded_under_verb": "Embed-\nVerb [C]",
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--in-path", "-i", default="data/eval/mnli-sp50.json", help="File with the accuracies")
    parser.add_argument("--out-path", "-o", default="data/plots/mnli-sp50-vplot.pdf", help="File to save the plot to")
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
    parser.add_argument("--linepos", "-p", default=1.51, type=float, help="Position of horizontal line")
    
    args = parser.parse_args()
    
    return args

def main(): 
    args = parse_args()
    
    df_dict = json.load(open(args.in_path))
    df_dict.pop("sparsity")
    df_dict.pop("seed")

    case_map = {k: BIG_NAME_MAP[k] for k in args.cases}

    df_dict = {v: df_dict[k] for k, v in case_map.items()}

    orig = {}

    order = case_map.values()

    for k in df_dict.keys():
        orig[k] = df_dict[k][0]
        df_dict[k] = df_dict[k][1:]

    df_intermediate = {
        "Subcase": [],
        "Accuracy (%)": []
    }

    for k, v in df_dict.items():
        for e in v:
            df_intermediate["Subcase"].append(k)
            df_intermediate["Accuracy (%)"].append(100.0*e)

    df_df = pd.DataFrame(df_intermediate)

    sns.set_theme()

    ax = sns.violinplot(data=df_df, x="Subcase", y="Accuracy (%)", palette="Set3", cut=0, inner="point", order=order)

    xlabels = ax.get_xticklabels()

    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)", fontsize=16)

    ls = ax.get_xticklabels()
    ax.set_xticklabels(ls, fontsize=14)

    for i, l in enumerate(ls):
        start = i/len(ls)
        end = (i+1)/len(ls)
        ax.axhline(y=100.0*orig[l._text], xmin=start, xmax=end, color='k')

    ax.axvline(x=args.linepos, color='k', linestyle='--')
    ax.set_ylim(-1, 100)

    ax.annotate("In domain", (0.04, 1.03), xycoords='axes fraction', fontsize=18)
    ax.annotate("Out of domain", (0.47, 1.03), xycoords='axes fraction', fontsize=18)

    ax.figure.savefig(args.out_path, bbox_inches='tight')
    
if __name__ == '__main__':
    main()