#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAMT — Noise Injection + Robustness Evaluation
Evaluates model stability under different noise types:
    - Typos
    - Keyboard noise
    - Random character insertions
    - Casing noise
    - Slang replacements

Outputs:
    - robustness_metrics.json
    - robustness_samples.csv
"""

import torch
import random
import json
import csv
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


# ----------------------------------------------------------
# Noise Functions
# ----------------------------------------------------------

def typo_noise(text, prob=0.15):
    s = list(text)
    for i in range(len(s)):
        if random.random() < prob:
            s[i] = ""  # delete character
    return "".join(s)


QWERTY_MAP = {
    "q": "w", "w": "e", "e": "r", "r": "t",
    "t": "y", "y": "u", "u": "i", "i": "o", "o": "p",
    "a": "s", "s": "d", "d": "f", "f": "g", "g": "h",
    "h": "j", "j": "k", "k": "l",
    "z": "x", "x": "c", "c": "v", "v": "b", "b": "n", "n": "m"
}

def keyboard_noise(text, prob=0.1):
    s = list(text.lower())
    for i in range(len(s)):
        if s[i] in QWERTY_MAP and random.random() < prob:
            s[i] = QWERTY_MAP[s[i]]
    return "".join(s)


def random_char_injection(text, prob=0.1):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_text = ""
    for ch in text:
        new_text += ch
        if random.random() < prob:
            new_text += random.choice(alphabet)
    return new_text


def casing_noise(text, prob=0.5):
    return "".join([ch.upper() if random.random() < prob else ch.lower() for ch in text])


SLANG_MAP = {
    "you": "u",
    "are": "r",
    "before": "b4",
    "people": "ppl",
    "going to": "gonna",
    "want to": "wanna",
    "got to": "gotta",
}

def slang_noise(text):
    for k, v in SLANG_MAP.items():
        text = text.replace(k, v)
    return text


NOISE_TYPES = {
    "clean": lambda x: x,
    "typo": typo_noise,
    "keyboard": keyboard_noise,
    "random_char": random_char_injection,
    "casing": casing_noise,
    "slang": slang_noise
}


# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------

def evaluate(model, tokenizer, dataset, noise_fn):

    gold_intents = []
    pred_intents = []

    gold_arg_type = []
    pred_arg_type = []

    gold_arg_value = []
    pred_arg_value = []

    samples = []

    for item in tqdm(dataset, desc=f"Evaluating [{noise_fn.__name__}]"):
        text_clean = item["text"]
        text_noisy = noise_fn(text_clean)

        encoded = tokenizer(text_noisy, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_logits=True
            )

        intent_pred = outputs["intent_logits"].argmax(dim=1).item()
        arg_type_pred = outputs["arg_type_logits"].argmax(dim=1).item()
        arg_val_pred = outputs["arg_value_logits"].argmax(dim=1).item()

        intent_gold = model.label_map_intent[item["intent"]]
        arg_type_gold = model.label_map_arg_type[item["arg_type"]]
        arg_val_gold = model.label_map_arg_value[item["arg_value"]]

        gold_intents.append(intent_gold)
        pred_intents.append(intent_pred)

        gold_arg_type.append(arg_type_gold)
        pred_arg_type.append(arg_type_pred)

        gold_arg_value.append(arg_val_gold)
        pred_arg_value.append(arg_val_pred)

        samples.append({
            "clean": text_clean,
            "noisy": text_noisy,
            "intent_pred": intent_pred,
            "intent_gold": intent_gold,
            "arg_type_pred": arg_type_pred,
            "arg_type_gold": arg_type_gold,
            "arg_value_pred": arg_val_pred,
            "arg_value_gold": arg_val_gold,
        })

    metrics = {
        "intent_acc": accuracy_score(gold_intents, pred_intents),
        "arg_type_f1": f1_score(gold_arg_type, pred_arg_type, average="macro"),
        "arg_value_f1": f1_score(gold_arg_value, pred_arg_value, average="macro"),
    }

    return metrics, samples


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="JSONL dataset for robustness evaluation")
    parser.add_argument("--output_prefix", type=str, default="robustness")

    args = parser.parse_args()

    # Load model
    from damt_dependency_aware_multitask_transformer import DependencyAwareMultiTaskTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DependencyAwareMultiTaskTransformer.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model.backbone_name)
    model.to(device)

    # Load dataset
    data = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    all_metrics = {}
    all_samples = []

    # Loop over noise types
    for name, fn in NOISE_TYPES.items():
        metrics, samples = evaluate(model, tokenizer, data, fn)
        all_metrics[name] = metrics

        # Store labeled samples
        for s in samples:
            s["noise_type"] = name
            all_samples.append(s)

    # Save metrics
    with open(f"{args.output_prefix}_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[Saved] {args.output_prefix}_metrics.json")

    # Save sample predictions
    with open(f"{args.output_prefix}_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_samples[0].keys())
        writer.writeheader()
        writer.writerows(all_samples)
    print(f"[Saved] {args.output_prefix}_samples.csv")


if __name__ == "__main__":
    main()
