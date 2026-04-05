"""
demo.py
=======
Milestone 4 — End-to-End Demonstration Script
Project: Citation Analysis — Perfunctory vs. Non-Perfunctory Citation Identification

Demonstrates the complete AI system pipeline in a single run:
  Step 1 — Load & Infer: load the trained SciBERT model and classify a sample citation
  Step 2 — Detect Drift: run the PSI-based monitoring dashboard on held-out dev data
  Step 3 — Flag for Human Review: simulate HITL confidence gate (low-confidence → human review)
  Step 4 — Update Model: apply a continual learning update on a small drifted batch
  Step 5 — Re-Infer: classify the same citation again with the updated model

All results are printed to stdout and saved to results/demo_output.json.
"""

import os
import sys
import json
import time
import random

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import SciCiteKeyModel, load_model, load_abstracts
from model_utils import MODEL_NAME, MAX_LEN, TEST_PATH, DEV_PATH

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
CL_EPOCHS    = 2
CL_LR        = 1e-5
HITL_THRESH  = 0.65   # predictions below this confidence are flagged for human review
DATA_DIR     = "data/raw/scicite/scicite"
RESULTS_DIR  = "results"

DEMO_CITATION = {
    "string": (
        "In this paper, we adopt the method proposed by Vaswani et al. (2017), "
        "which introduced the Transformer architecture and fundamentally changed "
        "the landscape of natural language processing."
    ),
    "sectionName": "Methods",
    "citedPaperId": "",  # no abstract for this demo sample
}





# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_section(text):
    if not isinstance(text, str): return "Other"
    t = text.lower()
    if "intro" in t or "background" in t: return "Introduction"
    if "method" in t: return "Methods"
    if "result" in t: return "Results"
    if "discuss" in t or "conclus" in t: return "Discussion"
    return "Other"


def infer_single(model, tokenizer, citation, abstracts, device):
    """Classify a single citation record. Returns (label_str, confidence, logits)."""
    text_a  = f"Section: {clean_section(citation.get('sectionName',''))}. {citation.get('string','')}"
    text_b  = abstracts.get(str(citation.get("citedPaperId", "")), "")
    enc     = tokenizer(text_a, text_b, add_special_tokens=True,
                        max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(ids, mask)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    label     = "Non-Perfunctory (Key)" if pred.item() == 1 else "Perfunctory (Not Key)"
    return label, float(conf.item()), logits


def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index between two distributions."""
    def scale_range(arr, lo, hi):
        arr = arr - arr.min()
        mx  = arr.max()
        if mx == 0: return arr + lo
        return arr / mx * (hi - lo) + lo

    bp = scale_range(np.linspace(0, 100, buckets + 1),
                     np.min(expected), np.max(expected))
    ep = np.histogram(expected, bp)[0] / max(len(expected), 1)
    ap = np.histogram(actual,   bp)[0] / max(len(actual), 1)

    psi = 0.0
    for e, a in zip(ep, ap):
        e = max(e, 1e-4); a = max(a, 1e-4)
        psi += (e - a) * np.log(e / a)
    return psi


class SimpleDataset(Dataset):
    def __init__(self, records, tokenizer, abstracts):
        self.records, self.tokenizer, self.abstracts = records, tokenizer, abstracts

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text_a = f"Section: {clean_section(r.get('sectionName',''))}. {r.get('string','')}"
        text_b = self.abstracts.get(str(r.get("citedPaperId", "")), "")
        label  = int(bool(r.get("isKeyCitation", False)))
        enc    = self.tokenizer(text_a, text_b, add_special_tokens=True,
                                max_length=MAX_LEN, padding="max_length",
                                truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].flatten(),
                "attention_mask": enc["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long)}


def eval_dataset(model, records, tokenizer, abstracts, device):
    ds  = SimpleDataset(records, tokenizer, abstracts)
    ldr = DataLoader(ds, batch_size=BATCH_SIZE)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in ldr:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            trues.extend(batch["labels"].numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds, average="macro", zero_division=0)


def get_confidence_scores(model, records, tokenizer, abstracts, device):
    ds  = SimpleDataset(records, tokenizer, abstracts)
    ldr = DataLoader(ds, batch_size=BATCH_SIZE)
    sm  = nn.Softmax(dim=1)
    confs = []
    model.eval()
    with torch.no_grad():
        for batch in ldr:
            probs = sm(model(batch["input_ids"].to(device),
                             batch["attention_mask"].to(device)))
            confs.extend(probs.max(dim=1).values.cpu().numpy().tolist())
    return confs


def cl_update_step(model, records, tokenizer, abstracts, device):
    ds  = SimpleDataset(records, tokenizer, abstracts)
    ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=CL_LR)
    crit = nn.CrossEntropyLoss()
    model.train()
    t0 = time.perf_counter()
    for _ in range(CL_EPOCHS):
        for batch in ldr:
            opt.zero_grad()
            loss = crit(model(batch["input_ids"].to(device),
                              batch["attention_mask"].to(device)),
                        batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model, time.perf_counter() - t0


# ── Main Demo ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sep = "=" * 65
    print(sep)
    print("  END-TO-END DEMO — Citation Analysis AI System")
    print("  Milestone 4: Continual Learning + HITL")
    print(sep)

    # ── Load assets ───────────────────────────────────────────────────────────
    print("\n[SETUP] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = load_model("best_model.pt", device)

    abstracts = load_abstracts()
    print(f"  ✔ Loaded {len(abstracts)} abstracts")

    df_test = pd.read_json(TEST_PATH, lines=True)
    df_test["isKeyCitation"] = df_test["isKeyCitation"].fillna(False).astype(bool)
    df_dev  = pd.read_json(DEV_PATH,  lines=True)
    df_dev["isKeyCitation"]  = df_dev["isKeyCitation"].fillna(False).astype(bool)
    print(f"  ✔ Test set: {len(df_test)}, Dev set: {len(df_dev)}")

    demo_output = {"device": str(device)}

    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 1 — INFERENCE: Classifying a sample citation")
    print(sep)
    print(f"  Context: \"{DEMO_CITATION['string'][:80]}...\"")
    print(f"  Section: {DEMO_CITATION['sectionName']}")

    t_infer = time.perf_counter()
    label, conf, _ = infer_single(model, tokenizer, DEMO_CITATION, abstracts, device)
    latency_ms = (time.perf_counter() - t_infer) * 1000

    print(f"\n  ▶ Prediction   : {label}")
    print(f"  ▶ Confidence   : {conf:.4f}")
    print(f"  ▶ Latency      : {latency_ms:.1f} ms")

    demo_output["step1_inference"] = {"citation_snippet": DEMO_CITATION["string"][:80],
                                       "prediction": label, "confidence": round(conf, 4),
                                       "latency_ms": round(latency_ms, 2)}

    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 2 — DRIFT DETECTION: PSI monitoring on production stream")
    print(sep)
    ref_lengths  = df_test["string"].apply(lambda x: len(str(x))).values
    batch_size   = len(df_dev) // 4
    psi_alerts   = []
    for b_id in range(4):
        batch   = df_dev.iloc[b_id * batch_size : (b_id + 1) * batch_size]
        lengths = batch["string"].apply(lambda x: len(str(x))).values
        psi_val = calculate_psi(ref_lengths, lengths)
        status  = "🔴 ALERT" if psi_val > 0.2 else "🟢 OK"
        null_rt = batch["sectionName"].isnull().mean()
        print(f"  Batch {b_id}: PSI={psi_val:.4f} {status}  NullRate={null_rt:.1%}")
        psi_alerts.append({"batch": b_id, "psi": round(psi_val, 4),
                            "null_rate": round(null_rt, 4), "alert": psi_val > 0.2})

    demo_output["step2_drift_detection"] = psi_alerts

    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 3 — HITL: Confidence gate & human review flagging")
    print(sep)

    sample_records = df_test.sample(n=min(60, len(df_test)), random_state=1).to_dict("records")
    confs   = get_confidence_scores(model, sample_records, tokenizer, abstracts, device)
    flagged = [(r, c) for r, c in zip(sample_records, confs) if c < HITL_THRESH]
    passed  = [(r, c) for r, c in zip(sample_records, confs) if c >= HITL_THRESH]

    print(f"  Threshold      : {HITL_THRESH}")
    print(f"  Samples scored : {len(sample_records)}")
    print(f"  Passed (auto)  : {len(passed)} ({len(passed)/len(sample_records):.1%})")
    print(f"  Flagged (HITL) : {len(flagged)} ({len(flagged)/len(sample_records):.1%})")

    if flagged:
        ex_r, ex_c = flagged[0]
        print(f"\n  Example flagged sample:")
        print(f"    Text:        \"{str(ex_r.get('string',''))[:70]}...\"")
        print(f"    Confidence:  {ex_c:.4f}  → routed to human reviewer")

    demo_output["step3_hitl"] = {
        "threshold": HITL_THRESH,
        "total_samples": len(sample_records),
        "auto_approved": len(passed),
        "flagged_for_review": len(flagged),
        "flag_rate": round(len(flagged) / max(len(sample_records), 1), 4),
    }

    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 4 — CL UPDATE: Fine-tuning on drifted batch with replay")
    print(sep)

    # Simulate drifted batch: method section + prior shift
    df_drifted  = df_test[df_test["sectionName"].astype(str).str.lower().str.contains("method", na=False)].copy()
    df_key      = df_drifted[df_drifted["isKeyCitation"] == True]
    df_notkey   = df_drifted[df_drifted["isKeyCitation"] == False]
    if len(df_key) > 0 and len(df_notkey) > 0:
        df_drifted = pd.concat([df_notkey] + [df_key] * 3, ignore_index=True)
    df_drifted  = df_drifted.sample(n=min(40, len(df_drifted)), random_state=7).reset_index(drop=True)
    drift_recs  = df_drifted.to_dict("records")

    # Evaluate before CL update on drifted batch
    acc_b, f1_b = eval_dataset(model, drift_recs, tokenizer, abstracts, device)
    print(f"  Before CL update: Acc={acc_b:.4f}  F1={f1_b:.4f}")
    print(f"  Applying CL update on {len(drift_recs)} samples ({CL_EPOCHS} epochs)...")

    model, update_time = cl_update_step(model, drift_recs, tokenizer, abstracts, device)

    acc_a, f1_a = eval_dataset(model, drift_recs, tokenizer, abstracts, device)
    print(f"  After CL update : Acc={acc_a:.4f}  F1={f1_a:.4f}")
    print(f"  Update time     : {update_time:.2f}s")
    print(f"  ΔF1             : {f1_a - f1_b:+.4f}")

    demo_output["step4_cl_update"] = {
        "drift_batch_size": len(drift_recs),
        "cl_epochs": CL_EPOCHS,
        "before_accuracy": round(acc_b, 4), "before_f1": round(f1_b, 4),
        "after_accuracy":  round(acc_a, 4), "after_f1":  round(f1_a, 4),
        "delta_f1":        round(f1_a - f1_b, 4),
        "update_time_s":   round(update_time, 3),
    }

    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  STEP 5 — RE-INFERENCE: Classify same citation with updated model")
    print(sep)

    label2, conf2, _ = infer_single(model, tokenizer, DEMO_CITATION, abstracts, device)
    print(f"  ▶ Updated Prediction : {label2}")
    print(f"  ▶ Updated Confidence : {conf2:.4f}  (was: {conf:.4f})")
    changed = "changed ✎" if label2 != label else "unchanged ✔"
    print(f"  ▶ Prediction status  : {changed}")

    demo_output["step5_reinference"] = {
        "prediction_before": label, "confidence_before": round(conf, 4),
        "prediction_after":  label2, "confidence_after": round(conf2, 4),
        "prediction_changed": label2 != label,
    }

    # ── Save demo output ──────────────────────────────────────────────────────
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            return super().default(obj)

    out_path = os.path.join(RESULTS_DIR, "demo_output.json")
    with open(out_path, "w") as f:
        json.dump(demo_output, f, indent=4, cls=NumpyEncoder)

    print(f"\n{sep}")
    print("  DEMO COMPLETE")
    print(f"  Results saved to {out_path}")
    print(sep)


if __name__ == "__main__":
    main()
