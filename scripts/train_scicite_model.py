
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = "data/raw/scicite/scicite"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
DEV_PATH = os.path.join(DATA_DIR, "dev.jsonl")
ABSTRACTS_PATH = os.path.join(DATA_DIR, "abstracts_mapping.json")

MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LEN = 512 
BATCH_SIZE = 8
EPOCHS = 5  # Increased for better convergence
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01  # L2 regularization

# Load Abstracts Mapping
print(f"Loading abstracts from {ABSTRACTS_PATH}...")
with open(ABSTRACTS_PATH, 'r', encoding='utf-8') as f:
    ABSTRACTS = json.load(f)
print(f"Loaded {len(ABSTRACTS)} abstract mappings.")

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

# Dataset Class for Single Head KeyCitation
class SciCiteKeyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.context = df.string.to_numpy()
        self.cited_ids = df.citedPaperId.to_numpy()
        self.sections = df.sectionName.apply(clean_section_name).to_numpy()
        
        # Target: Importance (isKeyCitation)
        self.labels = df.isKeyCitation.astype(int).to_numpy()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        context = str(self.context[item])
        cited_id = str(self.cited_ids[item])
        section = str(self.sections[item])
        
        # Fetch abstract if available
        abstract = ABSTRACTS.get(cited_id)
        if not abstract:
            abstract = ""
        
        # Input construction: 
        # [CLS] Section: <Section> [SEP] <Context> [SEP] <Abstract>
        # We can simulate this by putting Section + Context in text_a, and Abstract in text_b
        # Or Section in text_a, Context + Abstract in text_b
        # Let's try: text_a = "Section: Methods. Context...", text_b = Abstract
        
        text_a = f"Section: {section}. {context}"
        text_b = abstract
        
        encoding = self.tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# Model Class: Single Head with LayerNorm
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

def main():
    print(f"Loading data from {DATA_DIR}...")
    df_train = pd.read_json(TRAIN_PATH, lines=True)
    df_dev = pd.read_json(DEV_PATH, lines=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = SciCiteKeyDataset(df_train, tokenizer, MAX_LEN)
    val_dataset = SciCiteKeyDataset(df_dev, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SciCiteKeyModel().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_f1': [], 'val_acc': []}
    best_f1 = 0.0
    best_epoch = 0
    
    print("Starting Single-Head KeyCitation training (Enhanced)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        preds, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask)
                _, p = torch.max(logits, dim=1)
                
                preds.extend(p.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        # Macro F1 is what we care about for overall performance, but also check Key class
        val_f1_macro = f1_score(true_labels, preds, average='macro')
        
        history['val_f1'].append(val_f1_macro)
        
        # Early stopping tracking
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            best_epoch = epoch + 1
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_train_loss:.4f} - Val F1 (Macro): {val_f1_macro:.4f}  NEW BEST")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_train_loss:.4f} - Val F1 (Macro): {val_f1_macro:.4f}")
        
    print(f"\nTraining complete. Best F1: {best_f1:.4f} at Epoch {best_epoch}")
    
    # Final Report
    target_names = ["Not Key", "Key"]
    print("\n--- Classification Report ---")
    print(classification_report(true_labels, preds, target_names=target_names))
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_f1'], label='Val F1 (Macro)')
    plt.legend()
    plt.title('Single-Head KeyCitation Training')
    plt.savefig('key_citation_results.png')
    print("Results plotted to key_citation_results.png")

if __name__ == '__main__':
    main()
