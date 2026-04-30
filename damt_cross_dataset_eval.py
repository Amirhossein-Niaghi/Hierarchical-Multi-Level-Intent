#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAMT — Cross Dataset Evaluation Script
Evaluates a model trained on Dataset A on Dataset B,
reporting generalization metrics as used in the article.

Outputs:
    - intent_accuracy.json
    - argument_f1.json
    - detailed_predictions.csv
"""

import torch
import json
import csv
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


# ----------------------------------------------------------
# Evaluation Logic
# ----------------------------------------------------------

def evaluate_model(model, tokenizer, dataset):
    """
    dataset is a list of items:
        {
            "text": "...",
            "intent": "BookFlight",
            "arg_type": "location",
            "arg_value": "Tehran"
        }
    """

    gold_intents = []
    pred_intents = []

    gold_arg_type = []
    pred_arg_type = []

    gold_arg_value = []
    pred_arg_value = []

    rows = []

    for item in tqdm(dataset, desc="Evaluating"):
        text = item["text"]

        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_logits=True
            )

        # Preds
        intent_pred = outputs["intent_logits"].argmax(dim=1).item()
        arg_type_pred = outputs["arg_type_logits"].argmax(dim=1).item()
        arg_val_pred = outputs["arg_value_logits"].argmax(dim=1).item()

        # Golds (convert string labels to index)
        # We assume model has stored label mappings
        intent_gold = model.label_map_intent[item["intent"]]
        arg_type_gold = model.label_map_arg_type[item["arg_type"]]
        arg_val_gold = model.label_map_arg_value[item["arg_value"]]

        # Append
        pred_intents.append(intent_pred)
        gold_intents.append(intent_gold)

        pred_arg_type.append(arg_type_pred)
        gold_arg_type.append(arg_type_gold)

        pred_arg_value.append(arg_val_pred)
        gold_arg_value.append(arg_val_gold)

        rows.append({
            "text": text,
            "intent_pred": intent_pred,
            "intent_gold": intent_gold,
            "arg_type_pred": arg_type_pred,
            "arg_type_gold": arg_type_gold,
            "arg_value_pred": arg_val_pred,
            "arg_value_gold": arg_val_gold,
        })

    # Aggregate metrics
    metrics = {
        "intent_accuracy": accuracy_score(gold_intents, pred_intents),
        "arg_type_f1_macro": f1_score(gold_arg_type, pred_arg_type, average="macro"),
        "arg_value_f1_macro": f1_score(gold_arg_value, pred_arg_value, average="macro"),
    }

    return metrics, rows


# ----------------------------------------------------------
# Dataset Loader (JSONL)
# ----------------------------------------------------------

def load_jsonl(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to JSONL target evaluation dataset.")
    parser.add_argument("--output_prefix", type=str, default="cross_eval")

    args = parser.parse_args()

    # Load model and tokenizer
    from damt_dependency_aware_multitask_transformer import DependencyAwareMultiTaskTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DependencyAwareMultiTaskTransformer.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model.backbone_name)
    model.to(device)

    dataset = load_jsonl(args.dataset)

    metrics, rows = evaluate_model(model, tokenizer, dataset)

    # Save metrics
    with open(f"{args.output_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] {args.output_prefix}_metrics.json")

    # Save predictions
    with open(f"{args.output_prefix}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {args.output_prefix}_predictions.csv")


if __name__ == "__main__":
    main()
