"""
continual_learning.py
=====================
Milestone 4 — Continual Learning Module
Project: Citation Analysis — Perfunctory vs. Non-Perfunctory Citation Identification

Strategy: Replay-Buffer Fine-tuning
  - Maintains a fixed-size experience replay buffer of old labeled samples.
  - At each drift interval, fine-tunes the model on the new drifted batch PLUS
    a random sample from the replay buffer to prevent catastrophic forgetting.
  - Simulates 4 temporal drift intervals (T0 baseline → T3 compound drift).

Drift Simulation:
  T0: Clean test set (baseline, no update)
  T1: Domain shift — restrict to "Method" sections only
  T2: Prior shift — oversample key citations 5× to flip class balance
  T3: Compound — Method filter + prior shift + 20% char noise injection

Outputs (all saved to results/):
  cl_results.json          — per-step metrics (before/after F1, accuracy, latency, update time)
  cl_model_T{n}.pt         — checkpoint after each CL update
  cl_metric_plot.png       — Figure 2: F1 over drift intervals
  cl_efficiency_plot.png   — Figure 3: update time + memory over drift intervals
  cl_memory_plot.png       — Figure 4: model size and VRAM over drift intervals
"""

import os
import sys
import json
import time
import copy
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import SciCiteKeyModel, load_model, load_abstracts
from model_utils import MODEL_NAME, MAX_LEN, TEST_PATH

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME     = "allenai/scibert_scivocab_uncased"
MAX_LEN        = 512
BATCH_SIZE     = 16
CL_EPOCHS      = 3          # epochs per continual update step
CL_LR          = 1e-5       # lower LR to prevent catastrophic forgetting
REPLAY_BUFFER_SIZE = 100    # total samples kept in replay memory
REPLAY_SAMPLES_PER_UPDATE = 30  # samples drawn from buffer each CL step
ADAPT_TRAIN_SIZE = 20       # new labeled samples per drift interval
RESULTS_DIR    = "results"
DATA_DIR       = "data/raw/scicite/scicite"




# ── Dataset ───────────────────────────────────────────────────────────────────
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
    return "Other"


class CitationDataset(Dataset):
    def __init__(self, records, tokenizer, abstracts):
        """
        records: list of dicts with keys: string, citedPaperId, sectionName, isKeyCitation
        abstracts: dict mapping citedPaperId → abstract text
        """
        self.records   = records
        self.tokenizer = tokenizer
        self.abstracts = abstracts

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        context  = str(rec.get("string", ""))
        cited_id = str(rec.get("citedPaperId", ""))
        section  = clean_section_name(rec.get("sectionName", ""))
        abstract = self.abstracts.get(cited_id, "")
        label    = int(bool(rec.get("isKeyCitation", False)))

        text_a = f"Section: {section}. {context}"
        enc = self.tokenizer(
            text_a, abstract,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
def inject_char_noise(text: str, rate: float = 0.2) -> str:
    """Randomly substitutes characters with adjacent keyboard keys."""
    keyboard_neighbors = {
        'a': 's', 'b': 'v', 'c': 'x', 'd': 'f', 'e': 'r', 'f': 'd',
        'g': 'h', 'h': 'g', 'i': 'u', 'j': 'k', 'k': 'j', 'l': 'k',
        'm': 'n', 'n': 'm', 'o': 'p', 'p': 'o', 'q': 'w', 'r': 'e',
        's': 'a', 't': 'r', 'u': 'y', 'v': 'b', 'w': 'q', 'x': 'z',
        'y': 'u', 'z': 'x',
    }
    chars = list(text)
    for i, ch in enumerate(chars):
        if random.random() < rate and ch.lower() in keyboard_neighbors:
            sub = keyboard_neighbors[ch.lower()]
            chars[i] = sub.upper() if ch.isupper() else sub
    return "".join(chars)


def df_to_records(df: pd.DataFrame) -> list:
    return df.to_dict(orient="records")


def evaluate(model, loader, device):
    model.eval()
    preds, true_labels, latencies = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            t0 = time.perf_counter()
            logits = model(ids, mask)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / ids.size(0)
            latencies.append(elapsed_ms)

            _, p = torch.max(logits, 1)
            preds.extend(p.cpu().numpy())
            true_labels.extend(lbls.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1  = f1_score(true_labels, preds, average="macro", zero_division=0)
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p90 = float(np.percentile(latencies, 90)) if latencies else 0.0
    return acc, f1, p50, p90


def cl_update(model, new_records, replay_buffer, tokenizer, abstracts, device):
    """
    Fine-tune model on new_records + replay_buffer samples.
    Returns (updated_model, training_time_s, peak_vram_gb).
    """
    # Sample from replay buffer
    replay_sample = list(replay_buffer)
    if len(replay_sample) > REPLAY_SAMPLES_PER_UPDATE:
        replay_sample = random.sample(replay_sample, REPLAY_SAMPLES_PER_UPDATE)

    combined = new_records + replay_sample
    dataset  = CitationDataset(combined, tokenizer, abstracts)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=CL_LR)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.perf_counter()
    for _ in range(CL_EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)
            loss  = criterion(model(ids, mask), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    train_time = time.perf_counter() - t_start

    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return model, train_time, peak_vram


def build_drift_batch(df_test: pd.DataFrame, drift_step: int, rng_seed: int = 42) -> pd.DataFrame:
    """
    Returns a drifted subset of df_test based on drift_step.
      0: Clean baseline (random sample)
      1: Domain shift — only 'method' sections
      2: Prior shift — 5× oversample of key citations
      3: Compound — method filter + prior shift + char noise
    """
    df = df_test.copy()

    if drift_step == 0:
        return df.sample(n=min(60, len(df)), random_state=rng_seed).reset_index(drop=True)

    if drift_step >= 1:
        df = df[df["sectionName"].astype(str).str.lower().str.contains("method", na=False)]

    if drift_step >= 2:
        key   = df[df["isKeyCitation"] == True]
        notkey = df[df["isKeyCitation"] == False]
        if len(key) > 0 and len(notkey) > 0:
            df = pd.concat([notkey] + [key] * 5, ignore_index=True)

    if drift_step == 3:
        df["string"] = df["string"].apply(
            lambda x: inject_char_noise(str(x), rate=0.2) if isinstance(x, str) else x
        )

    df = df.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True).head(60)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer & model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_path = "best_model.pt"
    model = load_model(model_path, device)

    # Load abstracts
    print("Loading abstracts...")
    abstracts = load_abstracts()
    print(f"  {len(abstracts)} abstracts loaded.")

    # Load stratified test set (simulation playground)
    print(f"Loading test data from {TEST_PATH}...")
    df_test = pd.read_json(TEST_PATH, lines=True)
    df_test["isKeyCitation"] = df_test["isKeyCitation"].fillna(False).astype(bool)
    print(f"  {len(df_test)} test samples loaded.")

    # Initialize replay buffer (seeded with a slice of test set)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    seed_df = df_test.sample(n=min(REPLAY_BUFFER_SIZE, len(df_test)), random_state=0)
    replay_buffer.extend(df_to_records(seed_df))
    print(f"  Replay buffer initialized with {len(replay_buffer)} samples.")

    # ── CL Experiment Loop ────────────────────────────────────────────────────
    drift_names = [
        "T0: Clean Baseline",
        "T1: Domain Shift (Method sections)",
        "T2: Prior Shift (Key citations 5×)",
        "T3: Compound (Method + Prior + Char Noise)",
    ]

    results = []
    model_size_mb = os.path.getsize(model_path) / (1024 ** 2) if os.path.exists(model_path) else 0.0

    for step in range(4):
        print(f"\n{'='*60}")
        print(f"Drift Interval {step}: {drift_names[step]}")

        # Build drifted batch
        df_drift   = build_drift_batch(df_test, drift_step=step)
        print(f"  Drifted batch size: {len(df_drift)}")

        # Split: evaluation set (hold-out) / train set (adaptation)
        n_train   = min(ADAPT_TRAIN_SIZE, len(df_drift) // 2)
        df_eval   = df_drift.iloc[n_train:].reset_index(drop=True)
        df_train  = df_drift.iloc[:n_train].reset_index(drop=True)

        eval_ds  = CitationDataset(df_to_records(df_eval), tokenizer, abstracts)
        eval_ldr = DataLoader(eval_ds, batch_size=BATCH_SIZE)

        # ── Evaluate BEFORE update ────────────────────────────────────────────
        print(f"  Evaluating BEFORE update on {len(df_eval)} samples...")
        acc_b, f1_b, p50_b, p90_b = evaluate(model, eval_ldr, device)
        print(f"    Before → Acc: {acc_b:.4f}  F1: {f1_b:.4f}  "
              f"p50: {p50_b:.2f}ms  p90: {p90_b:.2f}ms")

        if step == 0:
            # T0: baseline — no update
            acc_a, f1_a, p50_a, p90_a = acc_b, f1_b, p50_b, p90_b
            update_time_s = 0.0
            peak_vram_gb  = 0.0
            print("  [T0] No CL update applied (baseline).")
        else:
            # ── CL Update ─────────────────────────────────────────────────────
            new_records = df_to_records(df_train)
            print(f"  Applying CL update: {len(new_records)} new + "
                  f"{min(len(replay_buffer), REPLAY_SAMPLES_PER_UPDATE)} replay samples...")
            model, update_time_s, peak_vram_gb = cl_update(
                model, new_records, replay_buffer, tokenizer, abstracts, device
            )

            # Add new samples to replay buffer
            replay_buffer.extend(new_records)
            print(f"  Replay buffer size after update: {len(replay_buffer)}")

            # Save checkpoint
            ckpt_path = os.path.join(RESULTS_DIR, f"cl_model_T{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            model_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
            print(f"  Checkpoint saved → {ckpt_path}  ({model_size_mb:.1f} MB)")

            # ── Evaluate AFTER update ─────────────────────────────────────────
            print(f"  Evaluating AFTER update on {len(df_eval)} samples...")
            acc_a, f1_a, p50_a, p90_a = evaluate(model, eval_ldr, device)
            print(f"    After  → Acc: {acc_a:.4f}  F1: {f1_a:.4f}  "
                  f"p50: {p50_a:.2f}ms  p90: {p90_a:.2f}ms")
            print(f"  Update time: {update_time_s:.2f}s  "
                  f"  Peak VRAM: {peak_vram_gb:.3f} GB")

        delta_f1  = f1_a  - f1_b
        delta_acc = acc_a - acc_b
        results.append({
            "step":            step,
            "drift_name":      drift_names[step],
            "before_accuracy": round(acc_b, 4),
            "before_f1_macro": round(f1_b, 4),
            "before_p50_ms":   round(p50_b, 3),
            "before_p90_ms":   round(p90_b, 3),
            "after_accuracy":  round(acc_a, 4),
            "after_f1_macro":  round(f1_a, 4),
            "after_p50_ms":    round(p50_a, 3),
            "after_p90_ms":    round(p90_a, 3),
            "delta_f1":        round(delta_f1, 4),
            "delta_accuracy":  round(delta_acc, 4),
            "update_time_s":   round(update_time_s, 3),
            "peak_vram_gb":    round(peak_vram_gb, 4),
            "model_size_mb":   round(model_size_mb, 2),
            "new_samples":     n_train if step > 0 else 0,
            "replay_samples":  min(len(replay_buffer), REPLAY_SAMPLES_PER_UPDATE) if step > 0 else 0,
        })

    # ── Save JSON Results ─────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "cl_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path}")

    # ── Plot Figure 2: F1 over drift intervals ────────────────────────────────
    steps     = [r["step"] for r in results]
    f1_before = [r["before_f1_macro"] for r in results]
    f1_after  = [r["after_f1_macro"]  for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, f1_before, marker="o", linestyle="--", color="#e74c3c",
            label="F1 Before CL Update", linewidth=2)
    ax.plot(steps, f1_after,  marker="s", linestyle="-",  color="#2ecc71",
            label="F1 After CL Update",  linewidth=2)
    ax.fill_between(steps, f1_before, f1_after,
                    where=[a >= b for a, b in zip(f1_after, f1_before)],
                    alpha=0.15, color="#2ecc71", label="Improvement")
    ax.fill_between(steps, f1_before, f1_after,
                    where=[a < b for a, b in zip(f1_after, f1_before)],
                    alpha=0.15, color="#e74c3c", label="Regression")
    ax.set_xticks(steps)
    ax.set_xticklabels([r["drift_name"] for r in results], rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_xlabel("Drift Interval", fontsize=11)
    ax.set_title("Figure 2: Macro F1 Score Before vs. After CL Update\n"
                 "Replay-Buffer Fine-tuning across drift intervals", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "cl_metric_plot.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"Figure 2 saved → {fig2_path}")

    # ── Plot Figure 3: Update time + latency ─────────────────────────────────
    update_times = [r["update_time_s"] for r in results]
    p50_after    = [r["after_p50_ms"]  for r in results]
    p90_after    = [r["after_p90_ms"]  for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    bars = ax1.bar(steps, update_times, color=["#95a5a6", "#3498db", "#9b59b6", "#e67e22"])
    ax1.set_xticks(steps)
    ax1.set_xticklabels([r["drift_name"] for r in results], rotation=12, ha="right", fontsize=9)
    ax1.set_ylabel("Update Time (seconds)", fontsize=11)
    ax1.set_title("Figure 3: CL Update Time and Inference Latency", fontsize=12)
    ax1.bar_label(bars, fmt="%.2fs", padding=3, fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, max(update_times) * 1.3 + 0.1)

    ax2.plot(steps, p50_after, marker="o", label="p50 latency (ms)", color="#3498db", linewidth=2)
    ax2.plot(steps, p90_after, marker="s", label="p90 latency (ms)", color="#e74c3c",
             linewidth=2, linestyle="--")
    ax2.set_xticks(steps)
    ax2.set_xticklabels([r["drift_name"] for r in results], rotation=12, ha="right", fontsize=9)
    ax2.set_ylabel("Inference Latency (ms/sample)", fontsize=11)
    ax2.set_xlabel("Drift Interval", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3_path = os.path.join(RESULTS_DIR, "cl_efficiency_plot.png")
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"Figure 3 saved → {fig3_path}")

    # ── Plot Figure 4: Model Size and VRAM ───────────────────────────────────
    model_sizes  = [r["model_size_mb"]  for r in results]
    vram_values  = [r["peak_vram_gb"]   for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(steps, model_sizes, color=["#95a5a6", "#1abc9c", "#27ae60", "#2ecc71"])
    ax1.set_xticks(steps)
    ax1.set_xticklabels([f"T{s}" for s in steps])
    ax1.set_ylabel("Model Size (MB)", fontsize=11)
    ax1.set_title("Model Checkpoint Size per CL Step", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(model_sizes):
        ax1.text(i, v + 2, f"{v:.1f} MB", ha="center", fontsize=9)

    ax2.bar(steps, vram_values, color=["#95a5a6", "#e74c3c", "#c0392b", "#922b21"])
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f"T{s}" for s in steps])
    ax2.set_ylabel("Peak VRAM (GB)", fontsize=11)
    ax2.set_title("Peak VRAM Usage During CL Update", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(vram_values):
        ax2.text(i, v + 0.01, f"{v:.3f} GB", ha="center", fontsize=9)

    fig.suptitle("Figure 4: Resource Usage During Continual Learning", fontsize=12, y=1.01)
    plt.tight_layout()
    fig4_path = os.path.join(RESULTS_DIR, "cl_memory_plot.png")
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved → {fig4_path}")

    # ── Print Summary Table ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("CONTINUAL LEARNING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Step':<6} {'Drift Scenario':<42} {'F1 Before':>9} {'F1 After':>9} "
          f"{'ΔF1':>8} {'UpdateTime':>11} {'VRAM(GB)':>9}")
    print("-"*80)
    for r in results:
        print(f"T{r['step']:<5} {r['drift_name']:<42} {r['before_f1_macro']:>9.4f} "
              f"{r['after_f1_macro']:>9.4f} {r['delta_f1']:>+8.4f} "
              f"{r['update_time_s']:>10.2f}s {r['peak_vram_gb']:>9.4f}")
    print("="*80)
    print("\nContinual learning experiment complete!")


if __name__ == "__main__":
    main()
