#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAMT — Integrated Gradients Attribution
Computes token-level attributions for model predictions.
Aligns with qualitative interpretability section of the article.
"""

import torch
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import json


# ----------------------------------------------------------
# Compute Integrated Gradients
# ----------------------------------------------------------

def integrated_gradients(model, input_ids, attention_mask, baseline_ids, steps=50, target_logit_index=None):
    """
    Implements Integrated Gradients:
        IG = (x - x_baseline) * integral_0^1 grad(F(x_baseline + α*(x-x_baseline))) dα

    Parameters:
        model: DAMT model
        input_ids: Tensor [1, seq_len]
        baseline_ids: Tensor [1, seq_len] (Usually all [PAD] or [MASK])
        steps: number of interpolation steps
        target_logit_index: index of class to compute gradient w.r.t.
    """

    device = input_ids.device

    # fetch embeddings
    embedding_layer = model.backbone.get_input_embeddings()

    input_embed = embedding_layer(input_ids)              # [1, seq_len, dim]
    baseline_embed = embedding_layer(baseline_ids)        # [1, seq_len, dim]

    delta = input_embed - baseline_embed

    accumulated_grads = torch.zeros_like(input_embed)

    for alpha in torch.linspace(0, 1, steps).to(device):
        embed = baseline_embed + alpha * delta
        embed.requires_grad_(True)

        outputs = model.forward_embeddings(
            embedded_inputs=embed,
            attention_mask=attention_mask,
            return_all_logits=True
        )

        # For intent attribution by default
        logits = outputs["intent_logits"][0]

        if target_logit_index is None:
            target_logit_index = torch.argmax(logits).item()

        target_logit = logits[target_logit_index]

        model.zero_grad()
        target_logit.backward(retain_graph=True)

        grads = embed.grad.clone()
        accumulated_grads += grads

    avg_grads = accumulated_grads / steps
    integrated_grads = delta * avg_grads

    # L2 norm across embedding dimensions for token-level score
    token_importance = integrated_grads.norm(dim=2).squeeze(0)

    return token_importance.cpu().numpy(), target_logit_index


# ----------------------------------------------------------
# Visualize Attribution (Bar + HTML-like)
# ----------------------------------------------------------

def render_colored_tokens(tokens, scores, output_html="attribution.html"):
    """
    Creates a simple HTML file with tokens colored based on attribution score.
    """
    scores = np.maximum(scores, 0)
    scores = scores / (scores.max() + 1e-10)

    def color_value(v):
        r = int(255 * v)
        return f"rgb({r}, 0, 0)"

    html = "<html><body><p style='font-size:18px;'>"
    for token, s in zip(tokens, scores):
        html += f"<span style='background-color:{color_value(s)}; padding:3px; margin:2px;'>{token}</span>"
    html += "</p></body></html>"

    with open(output_html, "w") as f:
        f.write(html)

    print(f"[Saved] {output_html}")


def plot_attribution_bar(tokens, scores, output_png="attribution_bar.png"):
    plt.figure(figsize=(14, 6))
    sns.barplot(x=tokens, y=scores, palette="Reds")
    plt.xticks(rotation=85)
    plt.title("Integrated Gradients Token Attribution")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[Saved] {output_png}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output_prefix", type=str, default="ig")

    args = parser.parse_args()

    from damt_dependency_aware_multitask_transformer import DependencyAwareMultiTaskTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    model = DependencyAwareMultiTaskTransformer.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model.backbone_name)
    model.to(device)

    encoded = tokenizer(args.text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Baseline: all PAD tokens (common in IG NLP work)
    baseline_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    scores, target_class = integrated_gradients(
        model,
        input_ids,
        attention_mask,
        baseline_ids,
        steps=args.steps,
        target_logit_index=None
    )

    # Save raw attributions
    with open(f"{args.output_prefix}_raw.json", "w") as f:
        json.dump({
            "tokens": tokens,
            "scores": scores.tolist(),
            "predicted_class": int(target_class)
        }, f, indent=2)
    print(f"[Saved] {args.output_prefix}_raw.json")

    # Output visualizations
    render_colored_tokens(tokens, scores, output_html=f"{args.output_prefix}_tokens.html")
    plot_attribution_bar(tokens, scores, output_png=f"{args.output_prefix}_bar.png")


if __name__ == "__main__":
    main()
