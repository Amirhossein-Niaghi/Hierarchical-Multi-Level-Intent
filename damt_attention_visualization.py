#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAMT — Attention Weight Visualization
Generates attention heatmaps for each layer of the Transformer,
aggregated over attention heads.

Aligns with the methodology described in the paper for qualitative analysis.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from transformers import AutoTokenizer


# ----------------------------------------------------------
# Function: Extract Attention Weights from DAMT
# ----------------------------------------------------------

def extract_attention(model, input_ids, attention_mask):
    """
    Returns attention maps from all layers and all heads.
    Expected output shape: [num_layers, num_heads, seq_len, seq_len]
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True     # model must support this
        )

    # outputs.attentions is a list of len=num_layers
    # each: [batch, num_heads, seq_len, seq_len]
    attentions = outputs.attentions
    return attentions


# ----------------------------------------------------------
# Aggregate Attention Across Heads
# ----------------------------------------------------------

def aggregate_heads(attentions_layer):
    """
    attentions_layer shape: [num_heads, seq_len, seq_len]
    Returns: aggregated attention [seq_len, seq_len]
    """
    return attentions_layer.mean(axis=0)


# ----------------------------------------------------------
# Plot Heatmap
# ----------------------------------------------------------

def plot_attention_heatmap(att_matrix, tokens, output_file, title="Attention Heatmap"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        att_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        square=True,
        cbar=True
    )
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[Saved] {output_file}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="DAMT model checkpoint path.")
    parser.add_argument("--text", type=str, required=True,
                        help="A single input text to visualize attention for.")
    parser.add_argument("--layer", type=int, default=0,
                        help="Which Transformer layer to visualize (0-indexed).")
    parser.add_argument("--output", type=str, default="attention_heatmap.png")

    args = parser.parse_args()

    # load model
    from damt_dependency_aware_multitask_transformer import DependencyAwareMultiTaskTransformer

    model = DependencyAwareMultiTaskTransformer.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model.backbone_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # tokenize
    encoded = tokenizer(args.text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # extract raw attention weights
    att_all_layers = extract_attention(model, input_ids, attention_mask)

    # choose layer
    att_layer = att_all_layers[args.layer].squeeze(0)  # -> [num_heads, seq_len, seq_len]

    # aggregate heads
    att_agg = aggregate_heads(att_layer)

    # save JSON raw weights
    json_path = "attention_raw.json"
    with open(json_path, "w") as f:
        json.dump(att_agg.tolist(), f, indent=2)
    print(f"[Saved] {json_path}")

    # plot
    plot_attention_heatmap(
        att_matrix=att_agg,
        tokens=tokens,
        output_file=args.output,
        title=f"Attention Heatmap (Layer {args.layer})"
    )


if __name__ == "__main__":
    main()
