import os
import json
import time
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_DATA_DIR = "data/raw/scicite/scicite"
DEFAULT_MODEL_NAME = "allenai/scibert_scivocab_uncased"

torch.backends.cudnn.benchmark = True


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_section_name(text):
    if not isinstance(text, str):
        return "Other"
    t = text.lower()
    if "intro" in t or "background" in t:
        return "Introduction"
    if "method" in t:
        return "Methods"
    if "result" in t:
        return "Results"
    if "discuss" in t or "conclus" in t:
        return "Discussion"
    return "Other"


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


class SciCiteKeyDataset(Dataset):
    """
    Binary classification for isKeyCitation.
    Uses section title + local citation context only.
    """
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.contexts = self.df["string"].fillna("").astype(str).tolist()
        self.sections = self.df["sectionName"].apply(clean_section_name).tolist()
        self.labels = self.df["isKeyCitation"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context = normalize_text(self.contexts[idx])
        section = self.sections[idx]

        text = f"Section: {section}. Citation context: {context}"

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(float(self.labels[idx]), dtype=torch.float),
        }


class SciCiteKeyModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls)).squeeze(-1)
        return logits


def compute_binary_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    key_f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "key_f1": float(key_f1),
        "key_precision": float(precision),
        "key_recall": float(recall),
    }


def find_best_threshold_macro_f1(y_true, scores, start=0.05, end=0.95, step=0.01):
    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_key_f1 = -1.0

    for thr in np.arange(start, end + 1e-9, step):
        preds = (scores >= thr).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        key_f1 = f1_score(y_true, preds, average="binary", pos_label=1, zero_division=0)

        if (macro_f1 > best_macro_f1) or (
            np.isclose(macro_f1, best_macro_f1) and key_f1 > best_key_f1
        ):
            best_threshold = float(thr)
            best_macro_f1 = float(macro_f1)
            best_key_f1 = float(key_f1)

    return best_threshold, best_macro_f1, best_key_f1


def maybe_remove_dev_overlap(df_train, df_dev):
    initial_len = len(df_train)
    df_train = df_train[~df_train["string"].isin(df_dev["string"])].copy()
    removed = initial_len - len(df_train)
    return df_train, removed


def make_loader(dataset, batch_size, shuffle, device):
    pin_memory = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, scaler, device, use_amp):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(data_loader.dataset), 1)


def evaluate(model, data_loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            scores = torch.sigmoid(logits).detach().cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(len(data_loader.dataset), 1)
    all_scores = np.array(all_scores, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    return avg_loss, all_scores, all_labels


def save_learning_curve(history, output_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc_05"], label="Val Acc @0.50")
    plt.plot(history["val_macro_f1_05"], label="Val Macro F1 @0.50")
    plt.plot(history["val_key_f1_05"], label="Val Key F1 @0.50")
    plt.legend()
    plt.title("Validation Metrics @0.50")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    dev_path = os.path.join(args.data_dir, "dev.jsonl")
    test_path = os.path.join(args.data_dir, "test.jsonl")

    print(f"Loading data from {args.data_dir}...")
    df_train = pd.read_json(train_path, lines=True)
    df_dev = pd.read_json(dev_path, lines=True)
    df_test = pd.read_json(test_path, lines=True)

    df_train, removed = maybe_remove_dev_overlap(df_train, df_dev)
    if removed > 0:
        print(f"Removed {removed} overlapping train examples found in dev set.")

    if args.debug:
        print("DEBUG MODE: tiny overfit test")
        df_train = df_train.head(32).copy()
        df_dev = df_train.copy()
        df_test = df_train.copy()

    print(f"Train size: {len(df_train)}")
    print(f"Dev size:   {len(df_dev)}")
    print(f"Test size:  {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")

    train_dataset = SciCiteKeyDataset(df_train, tokenizer, args.max_len)
    dev_dataset = SciCiteKeyDataset(df_dev, tokenizer, args.max_len)
    test_dataset = SciCiteKeyDataset(df_test, tokenizer, args.max_len)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, device=device)
    dev_loader = make_loader(dev_dataset, args.batch_size, shuffle=False, device=device)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, device=device)

    labels_array = df_train["isKeyCitation"].astype(int).values
    n_pos = int(labels_array.sum())
    n_neg = int(len(labels_array) - n_pos)
    print(f"Class distribution: {n_neg} negative, {n_pos} positive")

    model = SciCiteKeyModel(args.model_name).to(device)

    try:
        from thop import profile
        dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.max_len)).to(device)
        dummy_mask = torch.ones((1, args.max_len), dtype=torch.long).to(device)
        macs, params = profile(model, inputs=(dummy_input_ids, dummy_mask), verbose=False)
        print(f"Model Parameters: {params:,}")
        print(f"Estimated GFLOPS/sample: {macs * 2 / 1e9:.2f}")
    except Exception:
        macs, params = None, None
        print("Skipping FLOPS calculation.")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc_05": [],
        "val_macro_f1_05": [],
        "val_key_f1_05": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    overall_start = time.time()

    print(
        f"\nStarting training: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"max_len={args.max_len}, lr={args.lr}, warmup_steps={warmup_steps}, patience={args.patience}"
    )

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        val_loss, val_scores, val_labels = evaluate(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        val_preds_05 = (val_scores >= 0.50).astype(int)
        val_metrics_05 = compute_binary_metrics(val_labels, val_preds_05)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc_05"].append(val_metrics_05["accuracy"])
        history["val_macro_f1_05"].append(val_metrics_05["macro_f1"])
        history["val_key_f1_05"].append(val_metrics_05["key_f1"])

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc@0.50: {val_metrics_05['accuracy']:.4f} | "
            f"Val Macro F1@0.50: {val_metrics_05['macro_f1']:.4f} | "
            f"Val Key F1@0.50: {val_metrics_05['key_f1']:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "config": vars(args),
                },
                "best_model.pt",
            )
            print(f"  -> Saved new best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No val-loss improvement. Patience {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    total_training_time = time.time() - overall_start
    epochs_ran = len(history["train_loss"])

    print(f"\nTraining complete. Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Average time/epoch: {total_training_time / max(epochs_ran, 1):.2f}s")

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Peak VRAM allocated: {peak_vram_gb:.2f} GB")

    checkpoint = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded best checkpoint from epoch {checkpoint['best_epoch']}")

    dev_loss, dev_scores, dev_labels = evaluate(
        model=model,
        data_loader=dev_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    best_threshold, best_dev_macro_f1, best_dev_key_f1 = find_best_threshold_macro_f1(
        dev_labels, dev_scores, start=0.05, end=0.95, step=0.01
    )

    print(
        f"Tuned dev threshold: {best_threshold:.2f} | "
        f"Dev Macro F1: {best_dev_macro_f1:.4f} | "
        f"Dev Key F1: {best_dev_key_f1:.4f}"
    )

    test_loss, test_scores, test_labels = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    test_preds = (test_scores >= best_threshold).astype(int)
    test_metrics = compute_binary_metrics(test_labels, test_preds)

    print(f"\n--- Classification Report (threshold={best_threshold:.2f}) ---")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=["Not Key", "Key"],
            digits=4,
            zero_division=0,
        )
    )
    print(f"Final Test Loss:      {test_loss:.4f}")
    print(f"Final Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Final Test Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"Final Test Key F1:    {test_metrics['key_f1']:.4f}")
    print(f"Final Test Precision: {test_metrics['key_precision']:.4f}")
    print(f"Final Test Recall:    {test_metrics['key_recall']:.4f}")

    os.makedirs("results", exist_ok=True)

    save_learning_curve(history, "results/learning_curve.png")
    print("Saved plot to results/learning_curve.png")

    metrics = {
        "model_name": args.model_name,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "n_train": int(len(df_train)),
        "n_dev": int(len(df_dev)),
        "n_test": int(len(df_test)),
        "n_positive_train": n_pos,
        "n_negative_train": n_neg,
        "best_val_loss": float(best_val_loss),
        "tuned_dev_threshold": float(best_threshold),
        "tuned_dev_macro_f1": float(best_dev_macro_f1),
        "tuned_dev_key_f1": float(best_dev_key_f1),
        "final_test_loss": float(test_loss),
        "final_test_accuracy": float(test_metrics["accuracy"]),
        "final_test_macro_f1": float(test_metrics["macro_f1"]),
        "final_test_key_f1": float(test_metrics["key_f1"]),
        "final_test_precision": float(test_metrics["key_precision"]),
        "final_test_recall": float(test_metrics["key_recall"]),
        "total_training_time_s": float(total_training_time),
        "avg_time_per_epoch_s": float(total_training_time / max(epochs_ran, 1)),
        "peak_vram_gb": float(peak_vram_gb),
    }

    if macs is not None and params is not None:
        metrics["flops_per_sample_gflops"] = float((macs * 2) / 1e9)
        metrics["model_parameters"] = int(params)

    if os.path.exists("best_model.pt"):
        metrics["best_model_size_mb"] = os.path.getsize("best_model.pt") / (1024 ** 2)

    with open("results/training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics to results/training_metrics.json")


if __name__ == "__main__":
    main()