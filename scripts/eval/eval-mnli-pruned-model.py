"""
Evaluate a single pruned model on MNLI (validation matched or mismatched).
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
from modeling_pert import PertForSequenceClassification, get_mask

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", "-m", default="models/ft/MNLI-61360")
    parser.add_argument("--avg-activation-path", "-a", default="data/activations/mnli.pkl")
    parser.add_argument("--glue-path", "-gp", default="glue")
    parser.add_argument("--split", "-s", default="validation_matched")
    parser.add_argument("--max-num-examples", "-n", type=int, default=1000000)
    parser.add_argument("--out-path", "-o", default=None)       # Only pass in if not doing a step-size sweep
    parser.add_argument("--device", "-d", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--threshold-for-deterministic", "-t", type=float, default=None)
    parser.add_argument("--step-size", "-ss", type=float, default=None)
    parser.add_argument("--original", "-og", action="store_true")
    parser.add_argument("--force-include", "-f", nargs="+", default=[])
    parser.add_argument("--force-exclude", "-e", nargs="+", default=[])

    args = parser.parse_args()
    
    if args.avg_activation_path == "None":
        args.avg_activation_path = None
        
    if args.step_size is not None and args.out_path is not None:
        raise ValueError(
            "Only last output is saved to file when doing a step-size sweep. " + 
            "Please remove --out-path or set --step-size to None."
        )

    return args

def load_avg_activations(model, avg_activation_path, device):
    avg_activations = pickle.load(open(avg_activation_path, "rb"))
    for n, m in model.named_modules():
        if n in avg_activations:
            m.set_avg_activation(torch.from_numpy(avg_activations[n]).to(device))

def load_mnli(glue_path):
    if os.path.exists(glue_path):
        mnli = load_from_disk(os.path.join(glue_path, "MNLI"))
    else:
        mnli = load_dataset("glue", "mnli")
    return mnli

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
        model_cls = PertForSequenceClassification
        model = model_cls.from_pretrained(args.model_name_or_path).to(args.device)

    if len(args.force_include) > 0:
        force(args.force_include, model, args.device, include=True)
    if len(args.force_exclude) > 0:
        force(args.force_exclude, model, args.device, include=False)

    if args.avg_activation_path is not None:
        load_avg_activations(model, args.avg_activation_path, args.device)
        
    if args.threshold_for_deterministic is not None:
        thresholds = [args.threshold_for_deterministic]
    elif args.step_size is not None:
        thresholds = np.arange(0, 1, args.step_size)
    else:
        thresholds = [None]
    
    mnli = load_mnli(args.glue_path)
    num_examples = min(len(mnli[args.split]), args.max_num_examples)
    mnli = mnli[args.split].select(range(num_examples))

    for threshold in thresholds:
        print(f"Threshold: {threshold}")
        if threshold is not None:
            model.set_threshold_for_deterministic(threshold)
            
            nx = 0
            cx = 0
            for n, p in model.named_parameters():
                if "log_alpha" in n:
                    mask = get_mask(p, threshold_for_deterministic=threshold)
                    nx += mask.numel()
                    cx += mask.sum().item()
            print(f"Sparsity: {1 - (cx / nx)}") 

        preds = []
        logits = []
        for i in tqdm(range(0, num_examples, args.batch_size)):
            batch = mnli.select(range(i, min(i + args.batch_size, num_examples)))
            inputs = tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True, return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model(**inputs)
            preds.extend(outputs.logits.argmax(dim=-1).tolist())
            for j in range(outputs.logits.shape[0]):
                logits.append(outputs.logits[j].tolist())

        accuracy = 0
        if args.out_path is not None:
            out_data = []
            for i in range(len(preds)):
                e = mnli[i]
                e["pred"] = preds[i]
                e["logits"] = logits[i]
                accuracy += 1 if e["pred"] == e["label"] else 0
                out_data.append(e)
            with open(args.out_path, "w+") as f:
                json.dump(out_data, f, indent=4)
        else:
            for i in range(len(preds)):
                accuracy += 1 if preds[i] == mnli[i]["label"] else 0
        accuracy /= len(preds)

        print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    main()