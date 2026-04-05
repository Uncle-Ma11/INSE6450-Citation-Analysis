"""
train_scicite_model.py
======================
Canonical training script for isKeyCitation binary classification.

Usage →standard training (with abstract enrichment):
    python scripts/train_scicite_model.py

Usage →ablation: context-only (no abstract enrichment):
    python scripts/train_scicite_model.py --no_abstracts

Usage →explicit data files:
    python scripts/train_scicite_model.py \
        --train_file data/raw/scicite/scicite/resplit_train.jsonl \
        --dev_file   data/raw/scicite/scicite/resplit_val.jsonl   \
        --test_file  data/raw/scicite/scicite/resplit_test.jsonl

Outputs (all in --results_dir, default: results/):
    best_model.pt                  →best checkpoint (by val Macro F1)
    training_metrics.json          →full metrics and config
    learning_curve.png             →loss + F1 curves

Input modes:
    Default      : "Section: {section}. {context}" [SEP] {cited_abstract}
    --no_abstracts: "Section: {section}. {context}"  (no second segment)

Model:
    allenai/scibert_scivocab_uncased  +  Dropout →Linear(768→56) →LayerNorm →GELU
                                       →Dropout →Linear(256→)
Loss:
    Focal Loss (gamma=2.0) with per-class weights derived from training split
Optimizer:
    AdamW, LR=3e-5, linear warmup (ratio=0.06), gradient clipping
    Optimal hyperparameters identified by grid search →see results/sweep_results.json
"""

import os
import json
import time
import copy
import random
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, precision_recall_fscore_support,
)

torch.backends.cudnn.benchmark = True

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    data_dir   = "data/raw/scicite/scicite",
    train_file = None,           # overrides data_dir-based path when set
    dev_file   = None,
    test_file  = None,
    model_name = "allenai/scibert_scivocab_uncased",
    results_dir= "results",
    checkpoint = "best_model.pt",
    max_len    = 512,
    batch_size = 16,
    accum_steps= 2,             # effective batch = batch_size * accum_steps
    epochs     = 8,
    patience   = 2,
    lr         = 3e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.06,
    focal_gamma  = 2.0,
    label_smoothing = 0.0,
    dropout1   = 0.3,           # dropout before first linear
    dropout2   = 0.2,           # dropout before second linear
    mask_aug_prob = 0.5,        # prob each training sample gets word masking
    mask_word_frac= 0.10,       # fraction of words to mask when augmented
    seed       = 42,
    debug      = False,
    no_abstracts = False,       # if True, disable cited-paper abstract enrichment
)


# ── Utilities ─────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_section(text: str) -> str:
    if not isinstance(text, str):
        return "Other"
    t = text.lower()
    if "intro" in t or "background" in t:
        return "Introduction/Background"
    if "method" in t or "approach" in t or "material" in t:
        return "Methods"
    if "result" in t or "experiment" in t or "evaluat" in t:
        return "Results/Experiments"
    if "discuss" in t or "conclus" in t or "future" in t:
        return "Discussion/Conclusion"
    if "related" in t or "prior" in t or "review" in t:
        return "Related Work"
    return "Other"


def load_abstracts(data_dir: str) -> dict:
    path = os.path.join(data_dir, "abstracts_mapping.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["isKeyCitation"] = df["isKeyCitation"].fillna(False).astype(bool)
    return df


def resolve_paths(args) -> tuple[str, str, str]:
    dd = args.data_dir
    train = args.train_file or os.path.join(dd, "resplit_train.jsonl")
    dev   = args.dev_file   or os.path.join(dd, "resplit_val.jsonl")
    test  = args.test_file  or os.path.join(dd, "resplit_test.jsonl")
    # fall back to original splits if resplit files don't exist
    if not os.path.exists(train):
        train = os.path.join(dd, "train.jsonl")
        dev   = os.path.join(dd, "dev.jsonl")
        test  = os.path.join(dd, "test.jsonl")
    return train, dev, test


# ── Dataset ───────────────────────────────────────────────────────────────────
class CitationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, abstracts: dict,
                 max_len: int, is_train: bool = False,
                 mask_aug_prob: float = 0.5, mask_word_frac: float = 0.10):
        self.records       = df.reset_index(drop=True).to_dict("records")
        self.tokenizer     = tokenizer
        self.abstracts     = abstracts
        self.max_len       = max_len
        self.is_train      = is_train
        self.mask_aug_prob = mask_aug_prob
        self.mask_word_frac= mask_word_frac

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r       = self.records[idx]
        context = str(r.get("string", ""))
        section = clean_section(r.get("sectionName", ""))
        label   = int(bool(r.get("isKeyCitation", False)))

        # Dynamic word masking augmentation during training
        if self.is_train and random.random() < self.mask_aug_prob:
            words = context.split()
            if words:
                n_mask = max(1, int(len(words) * self.mask_word_frac))
                for i in random.sample(range(len(words)), min(n_mask, len(words))):
                    words[i] = "[MASK]"
                context = " ".join(words)

        text_a = f"Section: {section}. {context}"
        abst   = self.abstracts.get(str(r.get("citedPaperId", ""))) or ""

        enc = self.tokenizer(
            text_a, abst,
            add_special_tokens=True,
            max_length=self.max_len,
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


# ── Model ─────────────────────────────────────────────────────────────────────
class SciCiteKeyModel(nn.Module):
    """SciBERT + two-layer classifier head (768 →256 →2)."""
    def __init__(self, model_name: str, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(h, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return self.classifier(pooled)


# ── Loss ──────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device) -> dict:
    """Return metrics dict + raw scores for threshold calibration."""
    model.eval()
    preds, trues, scores = [], [], []
    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            prob = sm(logits)
            pred = torch.argmax(prob, dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(batch["labels"].tolist())
            scores.extend(prob[:, 1].cpu().tolist())  # P(Key)

    y_true  = np.array(trues)
    y_pred  = np.array(preds)
    y_score = np.array(scores)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0)

    return dict(
        f1_macro  = round(float(f1_macro), 4),
        accuracy  = round(float(accuracy_score(y_true, y_pred)), 4),
        f1_notkey = round(float(f1_per[0]),   4),
        f1_key    = round(float(f1_per[1]),   4),
        prec_notkey = round(float(prec_per[0]), 4),
        prec_key    = round(float(prec_per[1]), 4),
        rec_notkey  = round(float(rec_per[0]),  4),
        rec_key     = round(float(rec_per[1]),  4),
        report    = classification_report(
            y_true, y_pred,
            target_names=["Perfunctory(0)", "Key(1)"],
            digits=4, zero_division=0,
        ),
        y_true    = y_true,
        y_score   = y_score,
    )


def calibrate_threshold(y_true, y_score,
                         start: float = 0.10,
                         end:   float = 0.90,
                         step:  float = 0.01) -> tuple[float, float, float]:
    """Sweep decision threshold on val set to maximize Macro F1."""
    best_thr, best_f1, best_key_f1 = 0.5, -1.0, -1.0
    for thr in np.arange(start, end + 1e-9, step):
        preds    = (y_score >= thr).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro",  zero_division=0)
        key_f1   = f1_score(y_true, preds, average="binary", pos_label=1, zero_division=0)
        if macro_f1 > best_f1 or (np.isclose(macro_f1, best_f1) and key_f1 > best_key_f1):
            best_thr, best_f1, best_key_f1 = float(thr), float(macro_f1), float(key_f1)
    return best_thr, best_f1, best_key_f1


# ── Training run (one full config) ────────────────────────────────────────────
def run_training(
    cfg: dict,
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    tokenizer,
    abstracts: dict,
    device,
    checkpoint_path: str,
    results_dir: str,
    verbose: bool = True,
) -> dict:
    """Train the model for one config; return metrics dict."""
    seed_everything(cfg["seed"])

    # Build datasets
    ds_train = CitationDataset(df_train, tokenizer, abstracts, cfg["max_len"],
                               is_train=True,
                               mask_aug_prob=cfg["mask_aug_prob"],
                               mask_word_frac=cfg["mask_word_frac"])
    ds_val   = CitationDataset(df_val,   tokenizer, abstracts, cfg["max_len"])
    ds_test  = CitationDataset(df_test,  tokenizer, abstracts, cfg["max_len"])

    pin = device.type == "cuda"
    ldr_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=0, pin_memory=pin)
    ldr_val   = DataLoader(ds_val,   batch_size=cfg["batch_size"], shuffle=False,
                           num_workers=0, pin_memory=pin)
    ldr_test  = DataLoader(ds_test,  batch_size=cfg["batch_size"], shuffle=False,
                           num_workers=0, pin_memory=pin)

    # Class weights (balanced)
    n_pos    = int(df_train["isKeyCitation"].sum())
    n_neg    = len(df_train) - n_pos
    w_neg    = len(df_train) / (2.0 * n_neg)
    w_pos    = len(df_train) / (2.0 * n_pos)
    weights  = torch.tensor([w_neg, w_pos], dtype=torch.float, device=device)

    # Model, loss, optimizer
    model     = SciCiteKeyModel(cfg["model_name"],
                                dropout1=cfg["dropout1"],
                                dropout2=cfg["dropout2"]).to(device)
    criterion = FocalLoss(alpha=weights,
                          gamma=cfg["focal_gamma"],
                          label_smoothing=cfg["label_smoothing"])
    optimizer = AdamW(model.parameters(), lr=cfg["lr"],
                      weight_decay=cfg["weight_decay"])

    total_optimizer_steps = (len(ldr_train) // cfg["accum_steps"]) * cfg["epochs"]
    warmup_steps = int(total_optimizer_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training loop
    best_val_f1   = 0.0
    best_state    = None
    patience_cnt  = 0
    history       = {k: [] for k in ["train_loss", "val_f1", "val_f1_key",
                                      "val_f1_notkey", "val_acc"]}
    t_start       = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        t_ep = time.time()
        optimizer.zero_grad(set_to_none=True)

        iter_bar = tqdm(ldr_train, desc=f"Ep {epoch}/{cfg['epochs']}", leave=False,
                        disable=not verbose)
        for step, batch in enumerate(iter_bar, 1):
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device))
                loss   = criterion(logits, batch["labels"].to(device))
                loss   = loss / cfg["accum_steps"]

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg["accum_steps"]

            if step % cfg["accum_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            iter_bar.set_postfix(loss=f"{epoch_loss/step:.4f}")

        avg_loss = epoch_loss / len(ldr_train)

        # ── Validate ───────────────────────────────────────────────────────────
        val_m = evaluate(model, ldr_val, device)
        history["train_loss"].append(round(avg_loss, 4))
        history["val_f1"].append(val_m["f1_macro"])
        history["val_f1_key"].append(val_m["f1_key"])
        history["val_f1_notkey"].append(val_m["f1_notkey"])
        history["val_acc"].append(val_m["accuracy"])

        ep_time = time.time() - t_ep
        if verbose:
            print(f"  Epoch {epoch:02d}  loss={avg_loss:.4f}  "
                  f"val_F1={val_m['f1_macro']:.4f}  "
                  f"[Key={val_m['f1_key']:.4f}  NotKey={val_m['f1_notkey']:.4f}]  "
                  f"acc={val_m['accuracy']:.4f}  {ep_time:.1f}s")

        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = copy.deepcopy(model.state_dict())
            patience_cnt = 0
            if verbose:
                print(f"    →New best val F1: {best_val_f1:.4f}")
        else:
            patience_cnt += 1
            if verbose:
                print(f"    No improvement ({patience_cnt}/{cfg['patience']})")
            if patience_cnt >= cfg["patience"]:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    total_time  = time.time() - t_start
    peak_vram   = (torch.cuda.max_memory_allocated(device) / 1024**3
                   if device.type == "cuda" else 0.0)
    epochs_ran  = len(history["train_loss"])

    # ── Save best checkpoint ───────────────────────────────────────────────────
    torch.save(best_state, checkpoint_path)

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)

    val_final  = evaluate(model, ldr_val,  device)
    test_final = evaluate(model, ldr_test, device)

    # Threshold calibration on val →apply to test
    best_thr, cal_val_f1, cal_val_key_f1 = calibrate_threshold(
        val_final["y_true"], val_final["y_score"]
    )
    test_preds_cal = (test_final["y_score"] >= best_thr).astype(int)
    cal_test_f1    = f1_score(test_final["y_true"], test_preds_cal,
                               average="macro", zero_division=0)
    cal_test_key_f1 = f1_score(test_final["y_true"], test_preds_cal,
                                average="binary", pos_label=1, zero_division=0)

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  {'Metric':<26} {'Val':>8} {'Test@0.5':>10} {'Test@{:.2f}'.format(best_thr):>10}")
        print(f"  {'─'*60}")
        rows = [
            ("Macro F1",    val_final['f1_macro'],  test_final['f1_macro'],   cal_test_f1),
            ("Accuracy",    val_final['accuracy'],   test_final['accuracy'],   None),
            ("F1 Key(1)",   val_final['f1_key'],     test_final['f1_key'],     cal_test_key_f1),
            ("F1 NotKey(0)",val_final['f1_notkey'],  test_final['f1_notkey'],  None),
            ("Prec Key",    val_final['prec_key'],   test_final['prec_key'],   None),
            ("Rec Key",     val_final['rec_key'],    test_final['rec_key'],    None),
        ]
        for name, v, t05, tcal in rows:
            tcal_s = f"{tcal:10.4f}" if tcal is not None else "          "
            print(f"  {name:<26} {v:>8.4f} {t05:>10.4f} {tcal_s}")
        print(f"  {'─'*60}")
        print(f"\n  Val→Test gap (F1, @0.5) : {val_final['f1_macro']-test_final['f1_macro']:+.4f}")
        print(f"  Calibrated threshold    : {best_thr:.2f}  "
              f"→Test F1={cal_test_f1:.4f}  Key F1={cal_test_key_f1:.4f}")
        print(f"  Training time           : {total_time:.1f}s  "
              f"({total_time/epochs_ran:.1f}s/epoch)  Peak VRAM={peak_vram:.2f}GB")
        print(f"\n  Test classification report (@0.50 threshold):\n{test_final['report']}")

    return dict(
        config          = {k: v for k, v in cfg.items() if k != "debug"},
        split_sizes     = dict(train=len(df_train), val=len(df_val), test=len(df_test)),
        key_ratios      = dict(
            train = round(float(df_train["isKeyCitation"].mean()), 4),
            val   = round(float(df_val["isKeyCitation"].mean()),   4),
            test  = round(float(df_test["isKeyCitation"].mean()),  4),
        ),
        training        = dict(
            epochs_ran   = epochs_ran,
            best_val_f1  = round(best_val_f1,  4),
            total_time_s = round(total_time,   1),
            time_per_epoch_s = round(total_time / epochs_ran, 1),
            peak_vram_gb = round(peak_vram,    3),
            history      = history,
        ),
        val_metrics     = {k: v for k, v in val_final.items()
                           if k not in ("report", "y_true", "y_score")},
        test_metrics    = {k: v for k, v in test_final.items()
                           if k not in ("report", "y_true", "y_score")},
        calibration     = dict(
            best_threshold      = round(best_thr,        2),
            val_f1_calibrated   = round(cal_val_f1,      4),
            val_key_f1_calibrated = round(cal_val_key_f1, 4),
            test_f1_calibrated  = round(float(cal_test_f1), 4),
            test_key_f1_calibrated = round(float(cal_test_key_f1), 4),
        ),
        val_test_f1_gap = round(val_final["f1_macro"] - test_final["f1_macro"], 4),
    )


# ── Plot ──────────────────────────────────────────────────────────────────────
def save_learning_curve(history: dict, results: dict, out_path: str):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        f"SciBERT Training  |  "
        f"Val F1={results['val_metrics']['f1_macro']:.4f}  "
        f"Test F1={results['test_metrics']['f1_macro']:.4f}  "
        f"Gap={results['val_test_f1_gap']:+.4f}",
        fontsize=11, fontweight="bold",
    )

    # Loss
    ax = axes[0]
    ax.plot(ep, history["train_loss"], "o-", color="#e74c3c", label="Train Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # F1
    ax = axes[1]
    ax.plot(ep, history["val_f1"],      "o-",  color="#2ecc71", lw=2,  label="Val Macro F1")
    ax.plot(ep, history["val_f1_key"],  "s--", color="#3498db",        label="Val F1 Key(1)")
    ax.plot(ep, history["val_f1_notkey"],"^--",color="#9b59b6",        label="Val F1 NotKey(0)")
    ax.axhline(results["test_metrics"]["f1_macro"],  color="#2ecc71", lw=1.5,
               linestyle=":", label=f"Test Macro F1={results['test_metrics']['f1_macro']:.4f}")
    ax.axhline(results["calibration"]["test_f1_calibrated"], color="#f39c12", lw=1.5,
               linestyle=":", label=f"Test F1 cal@{results['calibration']['best_threshold']:.2f}={results['calibration']['test_f1_calibrated']:.4f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
    ax.set_title("Validation F1 + Test Benchmark")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()



# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train SciBERT for isKeyCitation binary classification."
    )
    p.add_argument("--data_dir",   default=DEFAULTS["data_dir"])
    p.add_argument("--train_file", default=DEFAULTS["train_file"])
    p.add_argument("--dev_file",   default=DEFAULTS["dev_file"])
    p.add_argument("--test_file",  default=DEFAULTS["test_file"])
    p.add_argument("--model_name", default=DEFAULTS["model_name"])
    p.add_argument("--results_dir",default=DEFAULTS["results_dir"])
    p.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    p.add_argument("--max_len",    type=int,   default=DEFAULTS["max_len"])
    p.add_argument("--batch_size", type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--accum_steps",type=int,   default=DEFAULTS["accum_steps"])
    p.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--patience",   type=int,   default=DEFAULTS["patience"])
    p.add_argument("--lr",         type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay",type=float,default=DEFAULTS["weight_decay"])
    p.add_argument("--warmup_ratio",type=float,default=DEFAULTS["warmup_ratio"])
    p.add_argument("--focal_gamma", type=float,default=DEFAULTS["focal_gamma"])
    p.add_argument("--label_smoothing",type=float,default=DEFAULTS["label_smoothing"])
    p.add_argument("--dropout1",   type=float, default=DEFAULTS["dropout1"])
    p.add_argument("--dropout2",   type=float, default=DEFAULTS["dropout2"])
    p.add_argument("--seed",       type=int,   default=DEFAULTS["seed"])
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--no_abstracts", action="store_true",
                   help="Disable cited-paper abstract enrichment. "
                        "Trains on citation context + section name only. "
                        "Useful for ablation studies comparing context-only "
                        "vs context+abstract performance.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = vars(args)
    cfg["mask_aug_prob"]  = DEFAULTS["mask_aug_prob"]
    cfg["mask_word_frac"] = DEFAULTS["mask_word_frac"]

    os.makedirs(args.results_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  SciCite Key-Citation SciBERT Trainer")
    print(f"  Device: {device}  |  Debug: {args.debug}")
    print("=" * 65)

    # Resolve data file paths
    train_path, dev_path, test_path = resolve_paths(args)
    print(f"\n  Train : {train_path}")
    print(f"  Val   : {dev_path}")
    print(f"  Test  : {test_path}\n")

    df_train = load_split(train_path)
    df_val   = load_split(dev_path)
    df_test  = load_split(test_path)

    if args.debug:
        print("  [DEBUG] Using 64/32/32 samples")
        df_train, df_val, df_test = (df_train.head(64), df_val.head(32), df_test.head(32))
        cfg["epochs"] = 2

    # Print class stats
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        kr = df["isKeyCitation"].mean()
        print(f"  {name:<6}: {len(df):,} samples  |  Key={kr:.1%}  NotKey={1-kr:.1%}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.no_abstracts:
        abstracts = {}
        print("  Abstract enrichment : DISABLED (--no_abstracts)")
        print("  Input mode          : context + section name only")
    else:
        abstracts = load_abstracts(args.data_dir)
        print(f"  Abstract enrichment : ENABLED  ({len(abstracts):,} mappings loaded)")
        if len(abstracts) == 0:
            print("  → abstracts_mapping.json not found →"
                  "run fetch_abstracts.py first, or pass --no_abstracts")

    # ── Training run         ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING  lr={cfg['lr']}  gamma={cfg['focal_gamma']}  "
          f"bs={cfg['batch_size']}×{cfg['accum_steps']}  seed={cfg['seed']}")
    print(f"{'='*65}")

    results = run_training(
        cfg, df_train, df_val, df_test,
        tokenizer, abstracts, device,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        verbose=True,
    )


    metrics_path = os.path.join(args.results_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Metrics saved →{metrics_path}")

    # Learning curve
    lc_path = os.path.join(args.results_dir, "learning_curve.png")
    save_learning_curve(results["training"]["history"], results, lc_path)
    print(f"  Learning curve →{lc_path}")

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"  Val  F1 Macro : {results['val_metrics']['f1_macro']:.4f}")
    print(f"  Test F1 Macro : {results['test_metrics']['f1_macro']:.4f}  (@0.50)")
    print(f"  Test F1 Macro : {results['calibration']['test_f1_calibrated']:.4f}"
          f"  (@{results['calibration']['best_threshold']:.2f} calibrated)")
    print(f"  Val→Test gap  : {results['val_test_f1_gap']:+.4f}")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
