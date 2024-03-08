"""
For a single pruned model, print out the heads and MLPs that are included.
"""

import os
import argparse

import sys
sys.path.append("/n/fs/nlp-ab4197/printer/pert")
from modeling_pert import PertForSequenceClassification, get_mask

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", default="ft-stash/ft-result-mnli-seed42mean-50")
    parser.add_argument("--threshold", "-t", default=0.5, type=float)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    model = PertForSequenceClassification.from_pretrained(args.model)
    model.set_threshold_for_deterministic(args.threshold)

    layers = []

    for n, m in model.named_modules():
        n = n[len("bert.encoder.layer."):]
        n = n[:n.find(".")].strip()
        if hasattr(m, 'log_alpha_heads'):
            mask = get_mask(m.log_alpha_heads, args.threshold).tolist()
            for i in range(len(mask)):
                if mask[i] == 1:
                    layers.append("a{}.h{}".format(n, i))
        elif hasattr(m, 'log_alpha_mlp'):
            mask = get_mask(m.log_alpha_mlp, args.threshold).detach().item()
            if mask == 1:
                layers.append("a{}.mlp".format(n))
    
    layers = sorted(layers)
    print(" ".join(layers))

if __name__ == "__main__":
    main()