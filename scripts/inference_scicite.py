"""
inference_scicite.py
====================
Batch and single-sample inference script for isKeyCitation classification.
Uses the canonical SciCiteKeyModel from model_utils (768→256→LayerNorm→GELU→2).

Usage:
    python scripts/inference_scicite.py --context "..." --section "Methods"
    python scripts/inference_scicite.py --model_path best_model.pt --context "..." --abstract "..."
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import SciCiteKeyModel, load_model as _load_model, clean_section_name
from model_utils import MODEL_NAME, MAX_LEN


def load_model(model_path: str, device: torch.device) -> SciCiteKeyModel:
    """Loads the canonical trained model from a checkpoint file."""
    return _load_model(model_path, device)


def predict_single(model: SciCiteKeyModel, tokenizer, context: str,
                   section_name: str, cited_abstract: str = "",
                   device: str = "cpu", max_len: int = MAX_LEN):
    """
    Performs inference on a single example.

    Args:
        model:          Loaded SciCiteKeyModel
        tokenizer:      AutoTokenizer (scibert_scivocab_uncased)
        context:        The citation context text
        section_name:   The section where citation appears
        cited_abstract: Abstract of the cited paper (optional)
        device:         'cpu' or 'cuda'
        max_len:        Max sequence length

    Returns:
        (prediction: int, confidence: float)  — 0=Not Key, 1=Key
    """
    section = clean_section_name(section_name)
    text_a  = f"Section: {section}. {context}"
    text_b  = cited_abstract

    enc = tokenizer(
        text_a, text_b,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1)
        _, pred = torch.max(logits, dim=1)

    return pred.item(), float(probs[0][1].item())


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained SciCite model")
    parser.add_argument("--model_path", type=str, default="best_model.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--context",  type=str,
                        default="This method improves upon previous work by using a transformer.",
                        help="Citation context text")
    parser.add_argument("--section",  type=str, default="Methods",
                        help="Section name")
    parser.add_argument("--abstract", type=str, default="",
                        help="Cited paper abstract (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    try:
        model = load_model(args.model_path, device)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model_path}' not found.")
        return

    print("-" * 40)
    print(f"Context : {args.context[:100]}")
    print(f"Section : {args.section}")
    print("-" * 40)

    prediction, confidence = predict_single(
        model, tokenizer, args.context, args.section, args.abstract, device
    )

    label_map = {0: "Perfunctory (Not Key)", 1: "Non-Perfunctory (Key)"}
    print(f"Prediction  : {prediction} — {label_map[prediction]}")
    print(f"Confidence  : {confidence:.4f}")


if __name__ == "__main__":
    main()
