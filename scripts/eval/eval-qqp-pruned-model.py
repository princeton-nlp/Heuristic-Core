"""
Evaluate a pruned QQP model, either as it stands, or evaluating it after remving 
a specified set of attention heads from it, one at a time.
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

    parser.add_argument("--model_name_or_path", "-m", default="models/ft/QQP-40000")
    parser.add_argument("--paws-path", "-p", default="data/datasets/paws-qqp")
    parser.add_argument("--avg-activation-path", "-a", default="data/activations/qqp.pkl")
    parser.add_argument("--out-path", "-o", default=None)
    parser.add_argument("--device", "-d", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--original", "-og", action="store_true")
    parser.add_argument("--try-removing-edges", "-re", nargs="+", default=[])

    args = parser.parse_args()

    return args

def load_avg_activations(model, avg_activation_path, device):
    avg_activations = pickle.load(open(avg_activation_path, "rb"))
    for n, m in model.named_modules():
        if n in avg_activations:
            m.set_avg_activation(torch.from_numpy(avg_activations[n]).to(device))

@torch.no_grad()
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = PertForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.device)

    if args.avg_activation_path is not None and not args.original:
        load_avg_activations(model, args.avg_activation_path, args.device)
    
    qqp = load_dataset("glue", "qqp")["validation"]
    paws_qqp = load_from_disk(args.paws_path)

    if args.try_removing_edges == ["all"]:
        args.try_removing_edges = [f"{i}.{j}" for i in range(12) for j in range(12)] + ["None"]
    elif args.try_removing_edges == []:
        args.try_removing_edges = ["None"]
    
    ablated = {}
    
    for e in tqdm(args.try_removing_edges):
        if e in ablated:
            continue
        
        if e != "None":
            layer, head = e.split(".")
            layer = int(layer)
            head = int(head)
            
            saved = model.bert.encoder.layer[layer].attention.self.log_alpha_heads[head].item()
            model.bert.encoder.layer[layer].attention.self.log_alpha_heads[head] = -10
        
        preds = []
        for i in tqdm(range(0, len(qqp), args.batch_size)):
            batch = qqp.select(range(i, min(i + args.batch_size, len(qqp))))
            inputs = tokenizer(batch["question1"], batch["question2"], padding=True, truncation=True, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
            preds.extend(outputs.logits.argmax(dim=-1).tolist())
            
        qqp_accuracy = (torch.LongTensor(preds) == torch.LongTensor(qqp["label"])).float().mean().item()
        
        preds = []
        for i in tqdm(range(0, len(paws_qqp), args.batch_size)):
            batch = paws_qqp.select(range(i, min(i + args.batch_size, len(paws_qqp))))
            inputs = tokenizer(batch["sentence1"], batch["sentence2"], padding=True, truncation=True, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
            preds.extend(outputs.logits.argmax(dim=-1).tolist())
        
        paws_qqp_accuracy = (torch.LongTensor(preds) == torch.LongTensor(paws_qqp["label"])).float().mean().item()
        paws_qqp_0_accuracy = (
            (torch.LongTensor(paws_qqp["label"]) == 0).float() * 
            (torch.LongTensor(paws_qqp["label"]) == torch.LongTensor(preds)).float()
        ).sum().item() / (torch.LongTensor(paws_qqp["label"]) == 0).int().sum().item()
        
        ablated[e] = {
            "qqp": qqp_accuracy,
            "paws-qqp": paws_qqp_accuracy,
            "paws-qqp_0": paws_qqp_0_accuracy
        }
        
        if e != "None":
            print(f"{e}: QQP = {qqp_accuracy}, PAWS-QQP = {paws_qqp_accuracy}, PAWS-QQP_0 (adv) = {paws_qqp_0_accuracy}")
            model.bert.encoder.layer[layer].attention.self.log_alpha_heads[head] = saved
        else:
            print(f"QQP = {qqp_accuracy}, PAWS-QQP = {paws_qqp_accuracy}, PAWS-QQP_0 (adv) = {paws_qqp_0_accuracy}")

    if args.out_path is not None:
        json.dump(ablated, open(args.out_path, "w+"), indent=4)

if __name__ == '__main__':
    main()