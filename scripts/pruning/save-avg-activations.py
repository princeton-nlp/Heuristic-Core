"""
Save average activations of the model on a dataset
The activations are averaged per-token-position, therefore we cannot use a batch size larger than 1,
as the length of the sequences in the batch may vary (padding would sway the average).
"""

import os
import json
import argparse
from tqdm import tqdm
import pickle

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.join(os.getcwd(), "src", "pruning"))
from modeling_pert import PertForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", "-d", default="mnli")
    parser.add_argument("--key1", "-k1", default="premise")
    parser.add_argument("--key2", "-k2", default="hypothesis")
    parser.add_argument("--initialize-from", "-i", default="models/ft/MNLI-61360/")
    parser.add_argument("--max-num-examples", "-n", default=100000, type=int)
    parser.add_argument("--device", "-g", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out-path", "-o", default="data/activations/mnli.pkl")

    args = parser.parse_args()

    if args.key2.lower() == "none":
        args.key2 = None

    return args

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.initialize_from)
    model = PertForSequenceClassification.from_pretrained(args.initialize_from).to(args.device)

    data = load_dataset("glue", args.dataset)["train"]
    data = data.select(range(min(args.max_num_examples, len(data))))

    model.reset_read_avg_activation()

    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data.select(range(i, min(i + args.batch_size, len(data))))
        if args.key2 is None:
            batch = tokenizer(batch[args.key1], padding=True, truncation=True, return_tensors="pt").to(args.device)
        else:
            batch = tokenizer(batch[args.key1], batch[args.key2], padding=True, truncation=True, return_tensors="pt").to(args.device)
        _ = model(**batch)

    avg_activations = {}
    for n, m in model.named_modules():
        if hasattr(m, "get_avg_activation"):
            avg_activations[n] = m.get_avg_activation().cpu().numpy()

    pickle.dump(avg_activations, open(args.out_path, "wb+"))

if __name__ == '__main__':
    main()