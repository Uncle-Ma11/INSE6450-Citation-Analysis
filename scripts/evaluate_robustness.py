import os
import sys
import json
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import SciCiteKeyModel, load_model, load_abstracts
from model_utils import MODEL_NAME, MAX_LEN, TEST_PATH

BATCH_SIZE = 16

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


class RobustnessDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, abstracts):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.context = df.string.to_numpy()
        self.cited_ids = df.citedPaperId.to_numpy()
        self.sections = df.sectionName.apply(clean_section_name).to_numpy()
        self.labels = df.isKeyCitation.astype(int).to_numpy()
        self.abstracts = abstracts

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        context = str(self.context[item])
        cited_id = str(self.cited_ids[item])
        section = str(self.sections[item])
        
        abstract = self.abstracts.get(cited_id, "")
        
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

def evaluate_model(model, loader, device):
    model.eval()
    preds, true_labels, confidences, latencies = [], [], [], []
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            start_time = time.time()
            logits = model(input_ids, attention_mask)
            elapsed = time.time() - start_time
            latencies.append(elapsed * 1000 / input_ids.size(0)) # ms per sample
            
            probs = softmax(logits)
            conf, p = torch.max(probs, dim=1)
            
            preds.extend(p.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            
    # Calculate Latency Metrics
    latencies = np.array(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    throughput = 1000 / np.mean(latencies) if len(latencies) > 0 else 0
            
    return true_labels, preds, confidences, p50_latency, p90_latency, throughput

def apply_corruption(text, corrupt_type, strength):
    if not corrupt_type:
        return text
    
    if corrupt_type == "mask":
        words = text.split()
        if len(words) > 0:
            num_to_mask = max(1, int(len(words) * strength))
            mask_indices = np.random.choice(len(words), size=num_to_mask, replace=False)
            for idx in mask_indices:
                words[idx] = "[MASK]"
            return " ".join(words)
    
    elif corrupt_type == "noise":
        aug = nac.KeyboardAug(aug_char_p=strength, aug_word_p=strength)
        return aug.augment(text)[0]
    
    elif corrupt_type == "synonym":
        aug = naw.SynonymAug(aug_p=strength)
        return aug.augment(text)[0]
        
    return text

def extract_failure_cases(df, true_labels, preds, confidences, max_cases=5):
    failures = []
    for i in range(len(df)):
        if true_labels[i] != preds[i]:
            failures.append({
                'text': df.iloc[i].string,
                'true_label': "Key" if true_labels[i] == 1 else "Not Key",
                'pred_label': "Key" if preds[i] == 1 else "Not Key",
                'confidence': float(confidences[i]),
                'failure_cause': 'Unknown',
                'mitigation_idea': 'Require threshold / Test-time aug'
            })
        if len(failures) >= max_cases:
            break
    return failures

def main():
    from transformers import AutoTokenizer
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model("best_model.pt", device)

    # Load abstracts mapping
    print("Loading abstracts mapping...")
    abstracts = load_abstracts()

    # Load Stratified Test Data (resplit_test.jsonl)
    print(f"Loading test data from {TEST_PATH}...")
    df_test = pd.read_json(TEST_PATH, lines=True)
    df_test["isKeyCitation"] = df_test["isKeyCitation"].fillna(False).astype(bool)

    # Evaluation Scenarios
    scenarios = [
        {"name": "Clean Baseline", "corrupt_type": None, "strength": 0.0},
        {"name": "Stress: Token Masking (Low)", "corrupt_type": "mask", "strength": 0.1},
        {"name": "Stress: Token Masking (High)", "corrupt_type": "mask", "strength": 0.3},
        {"name": "Stress: Char Noise (Low)", "corrupt_type": "noise", "strength": 0.05},
        {"name": "Stress: Char Noise (Med)", "corrupt_type": "noise", "strength": 0.1},
        {"name": "Stress: Char Noise (High)", "corrupt_type": "noise", "strength": 0.2},
        {"name": "Adversarial: Synonym (White-box proxy)", "corrupt_type": "synonym", "strength": 0.15},
    ]

    all_results = {}
    
    for scenario in scenarios:
        print(f"\nEvaluating Scenario: {scenario['name']}")
        
        # Apply corruption to df copy
        df_corrupt = df_test.copy()
        if scenario['corrupt_type']:
            print(f"Applying corruption: {scenario['corrupt_type']} at strength {scenario['strength']}...")
            df_corrupt['string'] = df_corrupt['string'].apply(lambda x: apply_corruption(x, scenario['corrupt_type'], scenario['strength']))
        
        dataset = RobustnessDataset(df_corrupt, tokenizer, MAX_LEN, abstracts)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        
        true_labels, preds, confs, p50, p90, tput = evaluate_model(model, loader, device)
        
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='macro')
        
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | p50: {p50:.2f}ms | p90: {p90:.2f}ms | Tput: {tput:.2f} samples/s")
        
        all_results[scenario['name']] = {
            "accuracy": acc,
            "f1_macro": f1,
            "latency_p50": p50,
            "latency_p90": p90,
            "throughput_samples_per_sec": tput
        }
        
        if scenario['name'] == "Clean Baseline":
            print("Extracting failures for Clean Baseline...")
            failures = extract_failure_cases(df_corrupt, true_labels, preds, confs)
            with open('results/failure_cases.json', 'w') as f:
                json.dump(failures, f, indent=4)

    # Scenario: OOD (Short Contexts)
    print("\nEvaluating Scenario: OOD: Short Context")
    df_ood = df_test[df_test['string'].str.len() < 100].copy()
    if len(df_ood) > 0:
        dataset_ood = RobustnessDataset(df_ood, tokenizer, MAX_LEN, abstracts)
        loader_ood = DataLoader(dataset_ood, batch_size=BATCH_SIZE)
        true_labels, preds, confs, p50, p90, tput = evaluate_model(model, loader_ood, device)
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='macro')
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | p50: {p50:.2f}ms | p90: {p90:.2f}ms | Tput: {tput:.2f} samples/s")
        all_results["OOD: Short Context"] = {
            "accuracy": acc,
            "f1_macro": f1,
            "latency_p50": p50,
            "latency_p90": p90,
            "throughput_samples_per_sec": tput
        }

    with open('results/robustness_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\nRobustness evaluation complete! Results saved in results/")

if __name__ == "__main__":
    main()
