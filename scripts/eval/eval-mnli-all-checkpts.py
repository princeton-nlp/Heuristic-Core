import os
import json
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_from_disk, load_dataset

from transformers import AutoTokenizer, BertForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", "-m", default="models/ft/MNLI-61360/")
    parser.add_argument("--hans-path", "-hp", default="data/datasets/hans-cases")
    parser.add_argument("--out-path", "-o", default="models/ft/MNLI-61360/eval_all.json")
    parser.add_argument("--device", "-d", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--step-size", "-s", type=int, default=500)

    args = parser.parse_args()

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


def main():
    args = parse_args()
    
    cases = os.listdir(args.hans_path)
    cases = [c for c in cases if c != "og"]

    accuracies = {}
    
    data_map = {}
    tasks = []
    
    print("Loading data...")
    mnli = load_dataset("glue", "mnli")
    for split in ["validation_matched", "validation_mismatched"]:
        data_map["mnli-"+split] = mnli[split]
        tasks.append("mnli-"+split)
    for case in cases:
        hans = load_hans(args.hans_path, case)
        data_map[case] = hans["validation"]
        tasks.append(case)
    
    if os.path.exists(args.out_path):
        perf_map = json.load(open(args.out_path, "r"))
    else:
        perf_map = {
            task: [] for task in tasks
        }
        perf_map["steps"] = []
        
        cur = args.step_size
        while os.path.exists(os.path.join(args.model_dir, f"checkpoint-{cur}")):
            perf_map["steps"].append(cur)
            cur += args.step_size

    cnt = 0
    for step in tqdm(perf_map["steps"]):
        cnt += 1
        if cnt <= len(perf_map["mnli-validation_matched"]):
            continue
        model_path = os.path.join(args.model_dir, f"checkpoint-{step}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path).to(args.device)
        
        for i in tqdm(range(len(tasks))):
            task = tasks[i]
            data = data_map[task]
            
            preds = []
            for i in range(0, len(data), args.batch_size):
                batch = data.select(range(i, min(i + args.batch_size, len(data))))
                inputs = tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True, return_tensors="pt").to(args.device)
                outputs = model(**inputs)
                preds.extend(outputs.logits.argmax(dim=-1).tolist())

            accuracy = 0
            for i in range(len(preds)):
                if "mnli" not in task:
                    preds[i] = 0 if preds[i] == 0 else 1
                accuracy += 1 if preds[i] == data[i]["label"] else 0
            accuracy /= len(preds)
            perf_map[task].append(accuracy)
        
    json.dump(perf_map, open(args.out_path, "w+"), indent=4)

if __name__ == '__main__':
    main()