import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys
from sklearn.metrics import accuracy_score, f1_score

# Adding current folder to path so we can import from train_scicite_model without issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_scicite_model import SciCiteKeyModel, SciCiteKeyDataset, MODEL_NAME, MAX_LEN, DATA_DIR, DEV_PATH

def benchmark_inference():
    print("--- Inference Benchmarking ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = SciCiteKeyModel()
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", map_location=device), strict=False)
        print("Loaded best_model.pt")
    else:
        print("best_model.pt not found. Using untrained model for benchmark.")
    
    model.to(device)
    model.eval()

    print("\n1. Dynamic Quantization (CPU only)")
    # PyTorch Dynamic Quantization reduces Linear layers to Int8
    model_quantized = torch.quantization.quantize_dynamic(
        model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
    )
    model.to(device) # put original model back to GPU
    print("Quantization applied.")

    # Load a subset of dev data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # print("Loading Dev Data for benchmark...")
    # df_dev = pd.read_json(DEV_PATH, lines=True).head(500) # subset for faster benchmark
    # dataset = SciCiteKeyDataset(df_dev, tokenizer, MAX_LEN)
    
    print("Loading Test Data for final benchmark...")
    from train_scicite_model import TEST_PATH
    df_test = pd.read_json(TEST_PATH, lines=True).head(500) # subset for faster benchmark
    dataset = SciCiteKeyDataset(df_test, tokenizer, MAX_LEN)
    
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n2. Measuring Throughput and peak VRAM (Batch size {batch_size})")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        
    start_time = time.time()
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)
            total_samples += input_ids.size(0)
    
    total_time = time.time() - start_time
    throughput = total_samples / total_time
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Total time for {total_samples} samples: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Accuracy (Original GPU): {acc:.4f} | F1 Macro: {f1:.4f}")
    
    if device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"Peak VRAM during inference (batch {batch_size}): {peak_vram:.2f} GB")

    print("\n3. Measuring Latency p50/p90 (Batch size 1)")
    latency_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    latencies = []
    
    # Warmup
    for i, batch in enumerate(latency_loader):
        if i > 10: break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            model(input_ids, attention_mask)
            
    with torch.no_grad():
        for i, batch in enumerate(latency_loader):
            if i >= 100: break # measure 100 samples
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Start timer
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            
            model(input_ids, attention_mask)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            t1 = time.time()
            
            latencies.append((t1 - t0) * 1000) # in ms

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    print(f"Latency over 100 samples (GPU):")
    print(f"p50: {p50:.2f} ms")
    print(f"p90: {p90:.2f} ms")

    print("\n4. Testing Quantized Model (INT8 on CPU)")
    print(f"Measuring Throughput and Accuracy (Batch size {batch_size})")
    cpu_device = torch.device('cpu')
    total_samples = 0
    all_preds_q = []
    all_labels_q = []
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(cpu_device)
            attention_mask = batch['attention_mask'].to(cpu_device)
            labels = batch['labels'].numpy()
            logits = model_quantized(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            all_preds_q.extend(preds.cpu().numpy())
            all_labels_q.extend(labels)
            total_samples += input_ids.size(0)
    total_time = time.time() - start_time
    quant_throughput = total_samples / total_time
    acc_q = accuracy_score(all_labels_q, all_preds_q)
    f1_q = f1_score(all_labels_q, all_preds_q, average='macro')
    print(f"Quantized Throughput (CPU): {quant_throughput:.2f} samples/sec")
    print(f"Accuracy (Quantized CPU): {acc_q:.4f} | F1 Macro: {f1_q:.4f}")

    latencies_quant = []
    with torch.no_grad():
        for i, batch in enumerate(latency_loader):
            if i >= 100: break
            input_ids = batch['input_ids'].to(cpu_device)
            attention_mask = batch['attention_mask'].to(cpu_device)
            t0 = time.time()
            model_quantized(input_ids, attention_mask)
            t1 = time.time()
            latencies_quant.append((t1 - t0) * 1000)

    p50_quant = np.percentile(latencies_quant, 50)
    p90_quant = np.percentile(latencies_quant, 90)
    print(f"Quantized Latency (p50): {p50_quant:.2f} ms")
    print(f"Quantized Latency (p90): {p90_quant:.2f} ms")

    print(f"\nOriginal CPU Latency (for fair comparison)")
    model.to("cpu")
    latencies_cpu_orig = []
    with torch.no_grad():
        for i, batch in enumerate(latency_loader):
            if i >= 100: break
            input_ids = batch['input_ids'].to(cpu_device)
            attention_mask = batch['attention_mask'].to(cpu_device)
            t0 = time.time()
            model(input_ids, attention_mask)
            t1 = time.time()
            latencies_cpu_orig.append((t1 - t0) * 1000)
    print(f"Original Latency CPU (p90): {np.percentile(latencies_cpu_orig, 90):.2f} ms")

    # Model Size comparison
    if os.path.exists("best_model.pt"):
        original_size = os.path.getsize("best_model.pt") / (1024**2)
        torch.save(model_quantized.state_dict(), "best_model_quantized.pt")
        quantized_size = os.path.getsize("best_model_quantized.pt") / (1024**2)
        print("\n5. Model Size")
        print(f"Original Model Size: {original_size:.2f} MB")
        print(f"Quantized Model Size: {quantized_size:.2f} MB")

    # Save inference metrics
    import json
    os.makedirs('results', exist_ok=True)
    metrics = {
        'gpu_throughput_bz32': throughput,
        'gpu_latency_p50_ms': p50,
        'gpu_latency_p90_ms': p90,
        'gpu_peak_vram_gb': peak_vram if device.type == 'cuda' else 0,
        'gpu_accuracy': acc,
        'gpu_f1_macro': f1,
        'cpu_quantized_throughput_bz32': quant_throughput,
        'cpu_quantized_latency_p50_ms': p50_quant,
        'cpu_quantized_latency_p90_ms': p90_quant,
        'cpu_quantized_accuracy': acc_q,
        'cpu_quantized_f1_macro': f1_q,
        'cpu_original_latency_p90_ms': np.percentile(latencies_cpu_orig, 90)
    }
    
    if os.path.exists("best_model.pt"):
        metrics['original_model_size_mb'] = original_size
        metrics['quantized_model_size_mb'] = quantized_size

    with open('results/inference_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Inference metrics saved to results/inference_metrics.json")
        
if __name__ == '__main__':
    benchmark_inference()
