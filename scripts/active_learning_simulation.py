"""
active_learning_simulation.py
==============================
Milestone 4 — Active Learning & Human-in-the-Loop Simulation
Strategy: Uncertainty Sampling — queries samples with lowest max-softmax confidence.
Human annotation is simulated using ground-truth labels from the SciCite dataset.
Compares uncertainty sampling efficiency against a random selection baseline.

Outputs (saved to results/):
  al_results.json       — per-cycle metrics
  al_efficiency_plot.png — Figure 5: effort vs. performance
"""

import os
import sys
import json
import time
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import SciCiteKeyModel, load_model, load_abstracts
from model_utils import MODEL_NAME, MAX_LEN, TRAIN_PATH, TEST_PATH

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 8
AL_EPOCHS         = 3
AL_LR             = 2e-5
INITIAL_POOL_SIZE = 100
QUERY_BATCH_SIZE  = 20
NUM_CYCLES        = 4
RANDOM_SEED       = 42
RESULTS_DIR       = "results"
DATA_DIR          = "data/raw/scicite/scicite"



# ── Dataset ───────────────────────────────────────────────────────────────────
def clean_section(text):
    if not isinstance(text, str):
        return "Other"
    t = text.lower()
    if "intro" in t or "background" in t: return "Introduction"
    if "method" in t: return "Methods"
    if "result" in t: return "Results"
    if "discuss" in t or "conclus" in t: return "Discussion"
    return "Other"


class CitationDataset(Dataset):
    def __init__(self, records, tokenizer, abstracts):
        self.records, self.tokenizer, self.abstracts = records, tokenizer, abstracts

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text_a = f"Section: {clean_section(r.get('sectionName',''))}. {r.get('string','')}"
        text_b = self.abstracts.get(str(r.get("citedPaperId", "")), "")
        label  = int(bool(r.get("isKeyCitation", False)))
        enc = self.tokenizer(text_a, text_b, add_special_tokens=True,
                             max_length=MAX_LEN, padding="max_length",
                             truncation=True, return_attention_mask=True,
                             return_tensors="pt")
        return {"input_ids": enc["input_ids"].flatten(),
                "attention_mask": enc["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long)}


# ── Helpers ───────────────────────────────────────────────────────────────────
def train_model(model, records, tokenizer, abstracts, device, epochs, lr):
    ds  = CitationDataset(records, tokenizer, abstracts)
    ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    labels_arr = np.array([int(bool(r.get("isKeyCitation", False))) for r in records])
    cls = np.unique(labels_arr)
    if len(cls) > 1:
        w = compute_class_weight("balanced", classes=cls, y=labels_arr)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))
    else:
        crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        for batch in ldr:
            opt.zero_grad()
            loss = crit(model(batch["input_ids"].to(device),
                              batch["attention_mask"].to(device)),
                        batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model, time.perf_counter() - t0


def evaluate(model, records, tokenizer, abstracts, device):
    if not records: return 0., 0., 0., 0.
    ds  = CitationDataset(records, tokenizer, abstracts)
    ldr = DataLoader(ds, batch_size=BATCH_SIZE)
    model.eval()
    preds, trues, lats = [], [], []
    with torch.no_grad():
        for batch in ldr:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            t    = time.perf_counter()
            out  = model(ids, mask)
            lats.append((time.perf_counter() - t) * 1000 / ids.size(0))
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            trues.extend(batch["labels"].numpy())
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    return acc, f1, float(np.percentile(lats, 50)), float(np.percentile(lats, 90))


def score_uncertainty(model, pool, tokenizer, abstracts, device):
    """Return pool sorted by ascending max-softmax confidence (most uncertain first)."""
    ds  = CitationDataset(pool, tokenizer, abstracts)
    ldr = DataLoader(ds, batch_size=BATCH_SIZE)
    sm  = nn.Softmax(dim=1)
    confs = []
    model.eval()
    with torch.no_grad():
        for batch in ldr:
            probs = sm(model(batch["input_ids"].to(device),
                             batch["attention_mask"].to(device)))
            confs.extend(probs.max(dim=1).values.cpu().numpy().tolist())
    scored = sorted(zip(confs, pool), key=lambda x: x[0])
    return scored  # ascending confidence → most uncertain first


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = random.Random(RANDOM_SEED)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    abstracts = load_abstracts()

    df_train = pd.read_json(TRAIN_PATH, lines=True)
    df_train["isKeyCitation"] = df_train["isKeyCitation"].fillna(False).astype(bool)
    df_test  = pd.read_json(TEST_PATH,  lines=True)
    df_test["isKeyCitation"]  = df_test["isKeyCitation"].fillna(False).astype(bool)
    test_recs = df_test.to_dict("records")

    pool_all = df_train.to_dict("records")
    rng.shuffle(pool_all)
    labeled_al  = pool_all[:INITIAL_POOL_SIZE]
    unlbl_al    = pool_all[INITIAL_POOL_SIZE:]
    labeled_rnd = list(labeled_al)
    unlbl_rnd   = list(unlbl_al)

    print(f"Initial labeled: {len(labeled_al)}, Unlabeled pool: {len(unlbl_al)}, Test: {len(test_recs)}")

    # Warm-start both models from best_model.pt
    model_al  = load_model("best_model.pt", device)
    model_rnd = load_model("best_model.pt", device)

    results_al  = []
    results_rnd = []

    # Cycle 0: train on initial pool
    print("\n── Cycle 0: Initial Training ──")
    model_al,  t_al  = train_model(model_al,  labeled_al,  tokenizer, abstracts, device, AL_EPOCHS, AL_LR)
    model_rnd, t_rnd = train_model(model_rnd, labeled_rnd, tokenizer, abstracts, device, AL_EPOCHS, AL_LR)
    acc_al,  f1_al,  p50_al,  _  = evaluate(model_al,  test_recs, tokenizer, abstracts, device)
    acc_rnd, f1_rnd, p50_rnd, _ = evaluate(model_rnd, test_recs, tokenizer, abstracts, device)
    print(f"  AL : Acc={acc_al:.4f}  F1={f1_al:.4f}  Update={t_al:.2f}s")
    print(f"  RND: Acc={acc_rnd:.4f}  F1={f1_rnd:.4f}  Update={t_rnd:.2f}s")

    results_al.append({"cycle": 0, "labeled_pool_size": len(labeled_al),
                        "new_samples": 0, "avg_uncertainty": None,
                        "accuracy": round(acc_al, 4), "f1_macro": round(f1_al, 4),
                        "p50_ms": round(p50_al, 3), "update_time_s": round(t_al, 3)})
    results_rnd.append({"cycle": 0, "labeled_pool_size": len(labeled_rnd),
                         "new_samples": 0,
                         "accuracy": round(acc_rnd, 4), "f1_macro": round(f1_rnd, 4),
                         "p50_ms": round(p50_rnd, 3), "update_time_s": round(t_rnd, 3)})

    # AL cycles
    for cycle in range(1, NUM_CYCLES + 1):
        print(f"\n── Cycle {cycle}: Query {QUERY_BATCH_SIZE} samples ──")
        if not unlbl_al or not unlbl_rnd:
            print("  Pool exhausted. Stopping.")
            break

        # Uncertainty sampling
        k = min(QUERY_BATCH_SIZE, len(unlbl_al))
        scored   = score_uncertainty(model_al, unlbl_al, tokenizer, abstracts, device)
        queried  = [r for _, r in scored[:k]]
        unlbl_al = [r for _, r in scored[k:]]
        avg_unc  = float(np.mean([1 - c for c, _ in scored[:k]]))
        labeled_al.extend(queried)

        model_al, t_al = train_model(model_al, labeled_al, tokenizer, abstracts, device, AL_EPOCHS, AL_LR)
        acc_al, f1_al, p50_al, _ = evaluate(model_al, test_recs, tokenizer, abstracts, device)
        print(f"  AL : Acc={acc_al:.4f}  F1={f1_al:.4f}  Pool={len(labeled_al)}  "
              f"Unc={avg_unc:.4f}  Update={t_al:.2f}s")

        # Random baseline
        k_rnd = min(QUERY_BATCH_SIZE, len(unlbl_rnd))
        pool_copy = list(unlbl_rnd)
        rng.shuffle(pool_copy)
        queried_rnd = pool_copy[:k_rnd]
        unlbl_rnd   = pool_copy[k_rnd:]
        labeled_rnd.extend(queried_rnd)

        model_rnd, t_rnd = train_model(model_rnd, labeled_rnd, tokenizer, abstracts, device, AL_EPOCHS, AL_LR)
        acc_rnd, f1_rnd, p50_rnd, _ = evaluate(model_rnd, test_recs, tokenizer, abstracts, device)
        print(f"  RND: Acc={acc_rnd:.4f}  F1={f1_rnd:.4f}  Pool={len(labeled_rnd)}  "
              f"Update={t_rnd:.2f}s")

        results_al.append({"cycle": cycle, "labeled_pool_size": len(labeled_al),
                            "new_samples": k, "avg_uncertainty": round(avg_unc, 4),
                            "accuracy": round(acc_al, 4), "f1_macro": round(f1_al, 4),
                            "p50_ms": round(p50_al, 3), "update_time_s": round(t_al, 3)})
        results_rnd.append({"cycle": cycle, "labeled_pool_size": len(labeled_rnd),
                             "new_samples": k_rnd,
                             "accuracy": round(acc_rnd, 4), "f1_macro": round(f1_rnd, 4),
                             "p50_ms": round(p50_rnd, 3), "update_time_s": round(t_rnd, 3)})

    # Save results
    out = {"config": {"initial_pool_size": INITIAL_POOL_SIZE, "query_batch_size": QUERY_BATCH_SIZE,
                      "num_cycles": NUM_CYCLES, "epochs_per_cycle": AL_EPOCHS,
                      "query_strategy": "uncertainty_sampling (min max-softmax)",
                      "annotation_method": "simulated (ground truth labels)"},
           "uncertainty_sampling": results_al, "random_baseline": results_rnd}
    out_path = os.path.join(RESULTS_DIR, "al_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
    print(f"\nResults saved to {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cycles    = [r["cycle"] for r in results_al]
    f1_al_v   = [r["f1_macro"] for r in results_al]
    f1_rnd_v  = [r["f1_macro"] for r in results_rnd]
    pool_al_v = [r["labeled_pool_size"] for r in results_al]
    pool_rnd_v= [r["labeled_pool_size"] for r in results_rnd]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(cycles, f1_al_v,  marker="o", color="#2ecc71", linewidth=2,
             label="Uncertainty Sampling")
    ax1.plot(cycles, f1_rnd_v, marker="s", color="#e74c3c", linewidth=2,
             linestyle="--", label="Random Baseline")
    ax1.fill_between(cycles, f1_rnd_v, f1_al_v,
                     where=[a >= b for a, b in zip(f1_al_v, f1_rnd_v)],
                     alpha=0.15, color="#2ecc71")
    ax1.set_xlabel("Annotation Cycle"); ax1.set_ylabel("Macro F1")
    ax1.set_title("F1 Score per AL Cycle\nUncertainty Sampling vs. Random Baseline")
    ax1.legend(); ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cycles)

    ax2.plot(pool_al_v,  f1_al_v,  marker="o", color="#3498db", linewidth=2,
             label="Uncertainty Sampling")
    ax2.plot(pool_rnd_v, f1_rnd_v, marker="s", color="#e74c3c", linewidth=2,
             linestyle="--", label="Random Baseline")
    ax2.set_xlabel("Labeled Pool Size (Annotation Effort)")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Figure 5: Annotation Effort vs. Performance\n(Active Learning Efficiency)")
    ax2.legend(); ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "al_efficiency_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Figure 5 saved → {plot_path}")

    # Print summary
    print("\n" + "="*70)
    print("ACTIVE LEARNING SUMMARY")
    print("="*70)
    print(f"{'Cycle':>5} {'Pool(AL)':>9} {'F1(AL)':>8} {'F1(RND)':>9} {'AvgUnc':>8} {'Time(s)':>9}")
    print("-"*70)
    for r_al, r_rnd in zip(results_al, results_rnd):
        unc = f"{r_al['avg_uncertainty']:.4f}" if r_al.get("avg_uncertainty") else "  —  "
        print(f"{r_al['cycle']:>5} {r_al['labeled_pool_size']:>9} "
              f"{r_al['f1_macro']:>8.4f} {r_rnd['f1_macro']:>9.4f} "
              f"{unc:>8} {r_al['update_time_s']:>9.2f}")
    print("="*70)
    print("Active learning simulation complete!")


if __name__ == "__main__":
    main()
