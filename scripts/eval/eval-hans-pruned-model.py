"""
Evaluate a single pruned model on HANS (all cases).
Please note that you are expected to have figured out a threshold by running the MNLI script first,
so we do not support a threshold sweep here.
The output is printed sorted by accuracy of the pruned model, in descending order.
We also support forcefully ablating or including heads/MLPs here.
Format: mlp.2 or a4.h3, etc. (all indices are 0-indexed)
"""

import os
import json
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_from_disk, load_dataset

from transformers import AutoTokenizer, BertForSequenceClassification

import sys
sys.path.append(os.path.join(os.getcwd(), "src", "pruning"))
from modeling_pert import PertForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", "-m", default="models/ft/MNLI-61360")
    parser.add_argument("--avg-activation-path", "-a", default="data/activations/mnli.pkl")
    parser.add_argument("--hans-path", "-hp", default="data/datasets/hans-cases")
    parser.add_argument("--max-num-examples", "-n", type=int, default=1000000)
    parser.add_argument("--out-path", "-o", default=None)
    parser.add_argument("--device", "-d", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--threshold-for-deterministic", "-t", type=float, default=None)
    parser.add_argument("--force-include", "-f", nargs="+", default=[])
    parser.add_argument("--force-exclude", "-e", nargs="+", default=[])
    parser.add_argument("--original", "-og", action="store_true")       # If the model is not pruned
    parser.add_argument("--suffix", "-s", default=None)
    parser.add_argument("--print-in-alphabetical-order", "-p", action="store_true")

    args = parser.parse_args()
    
    if args.avg_activation_path == "None":
        args.avg_activation_path = None

    return args

def load_avg_activations(model, avg_activation_path, device):
    avg_activations = pickle.load(open(avg_activation_path, "rb"))
    for n, m in model.named_modules():
        if n in avg_activations:
            m.set_avg_activation(torch.from_numpy(avg_activations[n]).to(device))

def load_hans(hans_path, case):
    if os.path.exists(hans_path):
        if case == "full":
            case = "og"
        hans = load_from_disk(os.path.join(hans_path, case.replace("/", "-")))
    else:
        assert case == "full", "Cannot use remote for cases"
        hans = load_dataset("hans")
    return hans

@torch.no_grad()
def force(names, model, device, include=True):
    """
    Forcefully includes or removes specified heads/MLPs from the model, by setting the log alphas to 10 or -10.
    """
    for name in names:
        if ".mlp" in name:
            a = int(name[1:name.find(".")])
            model.bert.encoder.layer[a].output.log_alpha_mlp[0] = 10 if include else -10
        else:
            dot = name.find(".")
            h = int(name[dot+2:])
            a = int(name[1:dot])
            model.bert.encoder.layer[a].attention.self.log_alpha_heads[h] = 10 if include else -10


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if args.original:
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.device)
    else:
        model = PertForSequenceClassification.from_pretrained(args.model_name_or_path, device_map=args.device)
        if args.threshold_for_deterministic is not None:
            model.set_threshold_for_deterministic(args.threshold_for_deterministic)

    if len(args.force_include) > 0:
        force(args.force_include, model, args.device, include=True)
    if len(args.force_exclude) > 0:
        force(args.force_exclude, model, args.device, include=False)

    if args.avg_activation_path is not None:
        load_avg_activations(model, args.avg_activation_path, args.device)
    
    cases = os.listdir(args.hans_path)
    cases = [c for c in cases if c != "og"]

    accuracies = {}

    for case in tqdm(cases):
        hans = load_hans(args.hans_path, case)
        num_examples = min(len(hans["validation"]), args.max_num_examples)
        hans = hans["validation"].select(range(num_examples))

        preds = []
        logits = []
        outputs_ = []
        for i in range(0, num_examples, args.batch_size):
            batch = hans.select(range(i, min(i + args.batch_size, num_examples)))
            inputs = tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
            preds.extend(outputs.logits.argmax(dim=-1).tolist())

            if args.out_path is not None:
                for j in range(outputs.logits.shape[0]):
                    logits.append(outputs.logits[j].tolist())

        accuracy = 0
        for i in range(len(preds)):
            output = {
                "label": hans[i]["label"],
                "pred": preds[i],
                "logits": logits[i] if args.out_path is not None else None,
            }
            preds[i] = 0 if preds[i] == 0 else 1
            output["pred_2"] = preds[i]    # Discount the neutral class, as HANS does not have it
            accuracy += 1 if preds[i] == hans[i]["label"] else 0

            if args.out_path is not None:
                outputs_.append(output)
        accuracy /= len(preds)

        accuracies[case] = accuracy
        if args.out_path is not None:
            fname = f"{case}"
            if args.suffix is not None:
                fname += "-" + args.suffix
            fname += ".json"
            json.dump(outputs_, open(os.path.join(args.out_path, fname), "w+"), indent=4)
    
    if args.print_in_alphabetical_order:
        cases = sorted(cases)
    else:
        cases = sorted(cases, key=lambda x: accuracies[x], reverse=True)
    for case in cases:
        print(f"{case}: {accuracies[case]}")

if __name__ == '__main__':
    main()