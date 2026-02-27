import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import argparse

# --- Model Definition (Must match training script) ---
MODEL_NAME = "allenai/scibert_scivocab_uncased"

def clean_section_name(text):
    if not isinstance(text, str):
        return "Other"
    
    t = text.lower()
    if "intro" in t or "background" in t:
        return "Introduction"
    elif "method" in t:
        return "Methods"
    elif "result" in t:
        return "Results"
    elif "discuss" in t or "conclus" in t:
        return "Discussion"
    else:
        return "Other"

class SciCiteKeyModel(nn.Module):
    def __init__(self):
        super(SciCiteKeyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        
        # LayerNorm for activation stability
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        
        # Single Head for Binary Classification
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        output = self.layer_norm(output)  # Normalize before classification
        logits = self.classifier(output)
        return logits

# --- Inference Functions ---

def load_model(model_path, device):
    """Loads the trained model from the specified path."""
    print(f"Loading model from {model_path}...")
    model = SciCiteKeyModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_single(model, tokenizer, context, section_name, cited_abstract="", device="cpu", max_len=512):
    """
    Performs inference on a single example.
    
    Args:
        model: Loaded SciCiteKeyModel
        tokenizer: AutoTokenizer
        context: The citation context text
        section_name: The section where citation appears
        cited_abstract: Abstract of the cited paper (optional)
        device: 'cpu' or 'cuda'
        max_len: Max sequence length
        
    Returns:
        prediction (0 or 1), probability (of class 1)
    """
    section = clean_section_name(section_name)
    
    # Input construction matching training script:
    # text_a = "Section: {section}. {context}"
    # text_b = abstract
    text_a = f"Section: {section}. {context}"
    text_b = cited_abstract
    
    encoding = tokenizer(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        _, pred = torch.max(logits, dim=1)
        
    return pred.item(), probs[0][1].item()

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained SciCite model")
    parser.add_argument("--model_path", type=str, default="best_model.pt", help="Path to the trained model file")
    parser.add_argument("--context", type=str, default="This method improves upon previous work by using a transformer.", help="Citation context")
    parser.add_argument("--section", type=str, default="Methods", help="Section name")
    parser.add_argument("--abstract", type=str, default="", help="Cited paper abstract")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load Model
    try:
        model = load_model(args.model_path, device)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model_path}' not found.")
        return

    # Run Prediction
    print("-" * 30)
    print(f"Context: {args.context}")
    print(f"Section: {args.section}")
    print("-" * 30)
    
    prediction, probability = predict_single(model, tokenizer, args.context, args.section, args.abstract, device)
    
    label_map = {0: "Not Key Citation", 1: "Key Citation"}
    print(f"Prediction: {prediction} ({label_map[prediction]})")
    print(f"Confidence (Key): {probability:.4f}")

if __name__ == "__main__":
    main()
