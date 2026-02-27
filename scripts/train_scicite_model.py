
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

# Configuration
DATA_DIR = "data/raw/scicite/scicite"
TRAIN_PATH = os.path.join(DATA_DIR, "train_augmented.jsonl")
DEV_PATH = os.path.join(DATA_DIR, "dev.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
ABSTRACTS_PATH = os.path.join(DATA_DIR, "abstracts_mapping.json")

MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LEN = 512 
BATCH_SIZE = 8
EPOCHS = 10  # Increased for better convergence
PATIENCE = 3 # Early stopping patience
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01  # L2 regularization
ACCUMULATION_STEPS = 4 # Effective Batch Size = 32

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with small dataset')
    args = parser.parse_args()

    print(f"Loading data from {DATA_DIR}...")
    df_train = pd.read_json(TRAIN_PATH, lines=True)
    df_dev = pd.read_json(DEV_PATH, lines=True)
    
    # Remove data leakage (overlap with dev set)
    initial_train_len = len(df_train)
    df_train = df_train[~df_train['string'].isin(df_dev['string'])]
    if len(df_train) < initial_train_len:
        print(f"Removed {initial_train_len - len(df_train)} overlapping examples from train set to prevent leakage.")
    
    if args.debug:
        print("DEBUG MODE: Overfitting test on small dataset")
        # Use a very small subset to ensure the model can overfit (memorize)
        df_train = df_train.head(10)
        df_dev = df_train  # Evaluate on training data to check for overfitting/memorization
        global EPOCHS
        EPOCHS = 50 
        print(f"Debug Grid: Train Size={len(df_train)}, Epochs={EPOCHS}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = SciCiteKeyDataset(df_train, tokenizer, MAX_LEN)
    val_dataset = SciCiteKeyDataset(df_dev, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Compute Class Weights
    labels = df_train.isKeyCitation.astype(int).values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Computed Class Weights: {class_weights}")
    
    model = SciCiteKeyModel().to(device)
    
    # --- Efficiency Metrics: FLOPS ---
    macs, params = None, None
    try:
        from thop import profile
        print("Calculating FLOPS...")
        # Dummy input for FLOPS calculation
        dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN)).to(device)
        dummy_mask = torch.ones((1, MAX_LEN), dtype=torch.long).to(device)
        macs, params = profile(model, inputs=(dummy_input_ids, dummy_mask), verbose=False)
        print(f"Model Parameters: {params:,}")
        print(f"Model MACs (approx FLOPS/2): {macs:,}")
        print(f"Model GFLOPS: {macs * 2 / 1e9:.2f} GFLOPS (estimated for batch size 1)")
        print("-" * 50)
    except ImportError:
        print("thop package not found, skipping FLOPS calculation.")
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': []
    }
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"Starting Single-Head KeyCitation training (Enhanced: {EPOCHS} Epochs, Patience={PATIENCE}, Accumulation={ACCUMULATION_STEPS})...")
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        
    overall_start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()
        
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Gradient Accumulation
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': loss.item() * ACCUMULATION_STEPS})
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        preds, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item() * input_ids.size(0)
                
                _, p = torch.max(logits, dim=1)
                
                preds.extend(p.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        
        # Macro F1 is what we care about for overall performance, but also check Key class
        val_f1_macro = f1_score(true_labels, preds, average='macro')
        val_acc = accuracy_score(true_labels, preds)
        
        history['val_f1'].append(val_f1_macro)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # Early stopping tracking
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1 (Macro): {val_f1_macro:.4f} - Val Acc: {val_acc:.4f} - Time: {epoch_duration:.2f}s")

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  -> New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("Early Stopping Triggered!")
            break
            
    total_training_time = time.time() - overall_start_time
    print(f"\nTraining complete. Best F1: {best_f1:.4f} at Epoch {best_epoch}")
    print(f"Total Training Time: {total_training_time:.2f}s (avg {total_training_time/(epoch+1):.2f}s/epoch)")
    
    if device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"Peak VRAM allocated during training: {peak_vram:.2f} GB")
    print("-" * 50)
    
    # Load test dataset for final metric reporting
    print("\nLoading Test Data for Final Report...")
    df_test = pd.read_json(TEST_PATH, lines=True)
    test_dataset = SciCiteKeyDataset(df_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    if os.path.exists("best_model.pt"):
        print("Loading best model weights for final evaluation...")
        model.load_state_dict(torch.load("best_model.pt", map_location=device), strict=False)

    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            _, p = torch.max(logits, dim=1)
            preds.extend(p.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    # Final Report
    target_names = ["Not Key", "Key"]
    print("\n--- Classification Report (Best Model) ---")
    print(classification_report(true_labels, preds, target_names=target_names))
    print(f"Final Validation Accuracy: {accuracy_score(true_labels, preds):.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red', linestyle='dashed')
    plt.plot(history['val_f1'], label='Val F1 (Macro)', color='green')
    plt.plot(history['val_acc'], label='Val Accuracy', color='orange')
    plt.legend()
    plt.title('Single-Head KeyCitation Training')
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/learning curve.png')
    print("Results plotted to results/learning curve.png")

    # Save training metrics
    metrics = {
        'total_training_time_s': total_training_time,
        'time_per_epoch_s': total_training_time / max(best_epoch, 1),
        'peak_vram_gb': peak_vram if device.type == 'cuda' else 0,
        'best_val_f1_macro': best_f1,
        'best_epoch': best_epoch,
        'final_val_accuracy': accuracy_score(true_labels, preds)
    }
    
    if macs is not None and params is not None:
        metrics['flops_per_sample_gflops'] = (macs * 2) / 1e9
        metrics['model_parameters'] = params
        
    if os.path.exists("best_model.pt"):
        metrics['model_size_mb'] = os.path.getsize("best_model.pt") / (1024**2)

    with open('results/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Training metrics saved to results/training_metrics.json")

if __name__ == '__main__':
    main()
