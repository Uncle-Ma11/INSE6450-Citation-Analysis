import os
import json
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.optim as optim

MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LEN = 512
BATCH_SIZE = 16

class SciCiteKeyModel(nn.Module):
    def __init__(self):
        super(SciCiteKeyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size  # 768
        
        # Two-layer classifier head: 768 → 256 → 2
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2)
        )
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        logits = self.classifier(pooled_output)
        return logits

def clean_section_name(text):
    if not isinstance(text, str): return "Other"
    t = text.lower()
    if "intro" in t or "background" in t: return "Introduction"
    elif "method" in t: return "Methods"
    elif "result" in t: return "Results"
    elif "discuss" in t or "conclus" in t: return "Discussion"
    else: return "Other"

class AdaptationDataset(Dataset):
    def __init__(self, df, tokenizer, abstracts):
        self.df = df
        self.tokenizer = tokenizer
        self.context = df.string.to_numpy()
        self.cited_ids = df.citedPaperId.to_numpy()
        self.sections = df.sectionName.apply(clean_section_name).to_numpy()
        self.labels = df.isKeyCitation.astype(int).to_numpy()
        self.abstracts = abstracts

    def __len__(self): return len(self.df)
    
    def __getitem__(self, item):
        context = str(self.context[item])
        cited_id = str(self.cited_ids[item])
        section = str(self.sections[item])
        abstract = self.abstracts.get(cited_id, "")
        text_a = f"Section: {section}. {context}"
        text_b = abstract
        
        encoding = self.tokenizer(
            text_a, text_b, add_special_tokens=True, max_length=MAX_LEN, 
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def evaluate(model, loader, device):
    model.eval()
    preds, true_labels, latencies = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            start_t = time.time()
            logits = model(input_ids, attention_mask)
            elapsed = time.time() - start_t
            latencies.append(elapsed * 1000 / input_ids.size(0))
            
            _, p = torch.max(logits, dim=1)
            preds.extend(p.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    lat = np.array(latencies)
    p50 = np.percentile(lat, 50) if len(lat)>0 else 0
    p90 = np.percentile(lat, 90) if len(lat)>0 else 0
    
    return acc, f1, p50, p90

def train_adaptation(model, train_loader, device, epochs=2):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    return model

def main():
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SciCiteKeyModel().to(device)
    
    model_path = "best_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("Loaded original best_model.pt")
        
    print("Loading abstracts mapping...")
    with open('data/raw/scicite/scicite/abstracts_mapping.json', 'r', encoding='utf-8') as f:
        abstracts = json.load(f)

    # Load Test Data to form a shifted/drifted dataset
    test_path = "data/raw/scicite/scicite/test.jsonl"
    df_test = pd.read_json(test_path, lines=True)
    
    # Simulate drift: Change class prior (heavily oversample minority class) 
    # and restrict to "Method" section texts to simulate a domain shift.
    print("Simulating Drift: Filtering to only 'Method' sections and balancing classes...")
    df_drifted = df_test[df_test['sectionName'].astype(str).str.lower().str.contains('method')].copy()
    
    # Duplicate the minority class to drastically change priors
    key_cites = df_drifted[df_drifted['isKeyCitation'] == True]
    not_key = df_drifted[df_drifted['isKeyCitation'] == False]
    
    if len(key_cites) > 0 and len(not_key) > 0:
        # Oversample key_cites 5x
        df_drifted = pd.concat([not_key] + [key_cites]*5, ignore_index=True)
    
    # Shuffle and trim
    df_drifted = df_drifted.sample(frac=1.0, random_state=42).reset_index(drop=True).head(60)
    print(f"Drifted evaluation set size: {len(df_drifted)} samples.")
    
    # Split drifted data: 20 for adaptation training, rest for evaluation
    train_size = min(20, len(df_drifted) // 2)
    df_adapt_train = df_drifted.iloc[:train_size]
    df_adapt_eval = df_drifted.iloc[train_size:]
    
    ds_eval = AdaptationDataset(df_adapt_eval, tokenizer, abstracts)
    loader_eval = DataLoader(ds_eval, batch_size=BATCH_SIZE)
    
    ds_train = AdaptationDataset(df_adapt_train, tokenizer, abstracts)
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    
    # 1. EVALUATE BEFORE ADAPTATION
    print("\nEvaluating Model BEFORE Adaptation on Drifted Data...")
    acc_before, f1_before, p50_b, p90_b = evaluate(model, loader_eval, device)
    print(f"Before -> Acc: {acc_before:.4f} | F1: {f1_before:.4f} | p50: {p50_b:.2f}ms | p90: {p90_b:.2f}ms")
    
    size_before = os.path.getsize(model_path) / (1024**2) if os.path.exists(model_path) else 0
    
    # 2. PERFORM INCEMENTAL RETRAINING / FINE-TUNING
    print(f"\nAdapting Model... (Fine-tuning on {len(df_adapt_train)} new labels for 3 epochs)")
    start_train = time.time()
    model = train_adaptation(model, loader_train, device, epochs=3)
    train_time = time.time() - start_train
    print(f"Adaptation Training completed in {train_time:.2f} seconds.")
    
    # Save new model weight 
    new_model_path = "results/adapted_model.pt"
    torch.save(model.state_dict(), new_model_path)
    size_after = os.path.getsize(new_model_path) / (1024**2)
    
    # 3. EVALUATE AFTER ADAPTATION
    print("\nEvaluating Model AFTER Adaptation on Drifted Data...")
    acc_after, f1_after, p50_a, p90_a = evaluate(model, loader_eval, device)
    print(f"After -> Acc: {acc_after:.4f} | F1: {f1_after:.4f} | p50: {p50_a:.2f}ms | p90: {p90_a:.2f}ms")

    # Serialize results
    results = {
        "metrics_before": {"accuracy": acc_before, "f1_macro": f1_before, "p50_latency_ms": p50_b, "p90_latency_ms": p90_b, "model_size_mb": size_before},
        "metrics_after": {"accuracy": acc_after, "f1_macro": f1_after, "p50_latency_ms": p50_a, "p90_latency_ms": p90_a, "model_size_mb": size_after},
        "adaptation_details": {"samples_used": len(df_adapt_train), "epochs": 3, "training_time_seconds": train_time}
    }
    
    with open('results/adaptation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nResults saved to results/adaptation_results.json")

if __name__ == "__main__":
    main()
