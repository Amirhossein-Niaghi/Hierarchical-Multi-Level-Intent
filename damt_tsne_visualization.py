#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAMT — t‑SNE Latent Space Visualization
This script:
    - loads a trained DAMT (or other models from previous stages)
    - extracts latent representations (CLS embeddings)
    - applies t‑SNE
    - visualizes clusters for Intent classes

Fully compatible with:
    damt_dependency_aware_multitask_transformer.py
    damt_ablation_experiments.py
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------
# Extract Latent Representations (CLS Embeddings)
# ----------------------------------------------------------

def extract_latent_representations(model, dataloader, device):
    embeddings = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # forward pass requesting CLS embedding extraction
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden=True       # Must be supported in model (done in earlier steps)
            )

            cls_rep = outputs["cls_hidden"]    # shape: [batch, hidden_size]
            embeddings.append(cls_rep.cpu().numpy())
            labels.extend(batch["intent"].cpu().numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels


# ----------------------------------------------------------
# Run t-SNE and Plot
# ----------------------------------------------------------

def run_tsne_and_plot(embeddings, labels, label_names, output_file):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1500,
        metric="euclidean",
        init="pca",
        random_state=42
    )

    reduced = tsne.fit_transform(embeddings)

    df_x = reduced[:, 0]
    df_y = reduced[:, 1]

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=df_x,
        y=df_y,
        hue=[label_names[l] for l in labels],
        palette="tab10",
        s=35,
        alpha=0.8
    )

    plt.title("t‑SNE Projection of Latent Space (Intent Clusters)")
    plt.xlabel("t‑SNE Dimension 1")
    plt.ylabel("t‑SNE Dimension 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[Saved] {output_file}")

    # Also save CSV for reproducibility
    np.savetxt("tsne_points.csv",
               np.column_stack([df_x, df_y, labels]),
               delimiter=",",
               fmt="%.6f",
               header="x,y,intent_label",
               comments="")
    print("[Saved] tsne_points.csv")



# ----------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained DAMT model checkpoint.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to evaluation dataset (PyTorch format).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="tsne_intent.png")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import model + dataset
    from damt_dependency_aware_multitask_transformer import DependencyAwareMultiTaskTransformer
    from dataset_loader import IntentArgDataset

    # Load dataset
    eval_set = IntentArgDataset(args.dataset)
    eval_loader = DataLoader(eval_set,
                             batch_size=args.batch_size,
                             shuffle=False)

    # Load model
    model = DependencyAwareMultiTaskTransformer.from_pretrained(args.checkpoint)
    model = model.to(device)

    # Extract latent embeddings
    embeddings, labels = extract_latent_representations(model, eval_loader, device)

    # Run t‑SNE
    run_tsne_and_plot(
        embeddings=embeddings,
        labels=labels,
        label_names=eval_set.intent_labels,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
