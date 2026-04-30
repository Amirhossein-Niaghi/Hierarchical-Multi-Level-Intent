#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAMT — Confusion Matrix Generator
This script loads a trained model checkpoint and generates confusion matrices
for all tasks described in the paper:
    - Intent Classification
    - Argument Type Prediction
    - Argument Value Prediction
    - Argument Pair Prediction (optional, used in some tables)

Completely compatible with previous generated codes:
    damt_dependency_aware_multitask_transformer.py
    damt_ablation_experiments.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# ----------------------------------------------------------
# Utility: Plot Confusion Matrix
# ----------------------------------------------------------

def plot_confusion_matrix(cm, labels, title, output_file):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[Saved] {output_file}")


# ----------------------------------------------------------
# Evaluation Loop
# ----------------------------------------------------------

def evaluate_and_confusion(model, dataloader, task_name, label_map, device):
    """Run model on dataloader and compute CM."""
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_logits=True        # must be implemented in main model code
            )

            # extract relevant logits
            logits = outputs[f"{task_name}_logits"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch[task_name].cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    return cm


# ----------------------------------------------------------
# Loading Model + Dataset
# ----------------------------------------------------------

def load_checkpoint(model_class, checkpoint_path, device):
    model = model_class.from_pretrained(checkpoint_path)
    model.to(device)
    return model


# ----------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained DAMT model checkpoint.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to preprocessed evaluation dataset (PyTorch).")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    # Import model class from main DAMT implementation
    from damt_dependency_aware_multitask_transformer import DependencyAwareMultiTaskTransformer
    from dataset_loader import IntentArgDataset  # same loader used in previous sections

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    eval_set = IntentArgDataset(args.dataset)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = load_checkpoint(DependencyAwareMultiTaskTransformer,
                            args.checkpoint, device)

    # Label names (should match dataset)
    intent_labels = eval_set.intent_labels
    argtype_labels = eval_set.argtype_labels
    argvalue_labels = eval_set.argvalue_labels

    # -------------------------
    # Intent Confusion Matrix
    # -------------------------
    cm_int = evaluate_and_confusion(model, eval_loader,
                                    task_name="intent",
                                    label_map=intent_labels,
                                    device=device)

    plot_confusion_matrix(
        cm_int,
        labels=intent_labels,
        title="Confusion Matrix — Intent Classification",
        output_file="confusion_matrix_intent.png"
    )

    # -------------------------
    # Argument Type
    # -------------------------
    cm_type = evaluate_and_confusion(model, eval_loader,
                                     task_name="arg_type",
                                     label_map=argtype_labels,
                                     device=device)

    plot_confusion_matrix(
        cm_type,
        labels=argtype_labels,
        title="Confusion Matrix — Argument Type Prediction",
        output_file="confusion_matrix_argtype.png"
    )

    # -------------------------
    # Argument Value
    # -------------------------
    cm_value = evaluate_and_confusion(model, eval_loader,
                                      task_name="arg_value",
                                      label_map=argvalue_labels,
                                      device=device)

    plot_confusion_matrix(
        cm_value,
        labels=argvalue_labels,
        title="Confusion Matrix — Argument Value Prediction",
        output_file="confusion_matrix_argvalue.png"
    )

    print("\nAll confusion matrices saved successfully.\n")


if __name__ == "__main__":
    main()
