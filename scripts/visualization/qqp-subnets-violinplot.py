import sys
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse

BIG_NAME_MAP = {
    "qqp": "QQP",
    "paws-qqp": "PAWS-QQP",
    "paws-qqp_0": "PAWS-QQP\n(only adversarial)",
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--in-path", "-i", default="data/eval/qqp-sp30.json", help="File with the accuracies")
    parser.add_argument("--out-path", "-o", default="data/plots/qqp-sp30-vplot.pdf", help="File to save the plot to")
    parser.add_argument(
        "--cases", 
        "-c", 
        nargs="+",
        default=[
            "qqp",
            "paws-qqp_0",
        ],
        help="Cases to include in the plot"
    )
    parser.add_argument("--add-legend", "-l", action="store_true", help="Add a legend to the plot")
    parser.add_argument("--linepos", "-p", default=0.5, type=float, help="Position of horizontal line")
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    df_dict = json.load(open(args.in_path))
    df_intermediate = {
        "Task": [],
        "Accuracy (%)": []
    }

    full_model = {}

    for case in args.cases:
        name = BIG_NAME_MAP[case]
        full_model[name] = 100.0*df_dict[case][0]
        for i in range(1, len(df_dict[case])):
            df_intermediate["Task"].append(name)
            df_intermediate["Accuracy (%)"].append(100.0*df_dict[case][i])

    df_df = pd.DataFrame(df_intermediate)

    sns.set_theme()
    sns.set(rc={'figure.figsize':(3.3,5)})

    tasks = [BIG_NAME_MAP[case] for case in args.cases]
    
    ax = sns.violinplot(data=df_df, x="Task", y="Accuracy (%)", palette="Set3", cut=0, inner="point", order=tasks)
    ls = ax.get_xticklabels()
    
    for i, l in enumerate(ls):
        start = i/len(ls)
        end = (i+1)/len(ls)
        ax.axhline(y=full_model[l._text], xmin=start, xmax=end, color='k')

    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")

    ax.set_xticklabels(tasks, fontsize=13)
    ax.set_ylim(-1, 100)

    ax.axvline(x=args.linepos, color='k', linestyle='--')

    ax.annotate("In Domain", (0.05, 1.03), xycoords='axes fraction', fontsize=14)
    ax.annotate("Out of Domain", (0.5, 1.03), xycoords='axes fraction', fontsize=13)

    ax.figure.savefig(args.out_path, bbox_inches='tight')
    
if __name__ == '__main__':
    main()