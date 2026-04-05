"""
model_utils.py
==============
Shared model definition and data-loading utilities for the SciCite pipeline.
ALL scripts must import SciCiteKeyModel from here so the architecture stays
consistent with best_model.pt at all times.

Architecture: SciBERT + Dropout → Linear(768→256) → LayerNorm(256) → GELU
                                 → Dropout → Linear(256→2)

Optimal hyperparameters (determined by grid search, April 2026):
    LR=3e-5, focal_gamma=2.0, warmup_ratio=0.06, label_smoothing=0.0
    batch_size=16, accum_steps=2, max_len=512, threshold=0.58 (calibrated)
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

# ── Global constants ──────────────────────────────────────────────────────────
MODEL_NAME = "allenai/scibert_scivocab_uncased"
DATA_DIR   = "data/raw/scicite/scicite"
MAX_LEN    = 512

# Resplit data files (stratified 80-10-10; use these for all evaluation)
TRAIN_PATH  = os.path.join(DATA_DIR, "resplit_train.jsonl")
DEV_PATH    = os.path.join(DATA_DIR, "resplit_val.jsonl")
TEST_PATH   = os.path.join(DATA_DIR, "resplit_test.jsonl")

# Fallback to original splits if resplit files don't exist
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
    DEV_PATH   = os.path.join(DATA_DIR, "dev.jsonl")
    TEST_PATH  = os.path.join(DATA_DIR, "test.jsonl")

ABSTRACTS_PATH = os.path.join(DATA_DIR, "abstracts_mapping.json")


# ── Section normalizer ────────────────────────────────────────────────────────
def clean_section_name(text: str) -> str:
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


# ── Abstract loader ───────────────────────────────────────────────────────────
def load_abstracts() -> dict:
    if os.path.exists(ABSTRACTS_PATH):
        with open(ABSTRACTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Canonical model class ─────────────────────────────────────────────────────
class SciCiteKeyModel(nn.Module):
    """
    SciBERT + two-layer classifier head with LayerNorm.
    768 → Dropout(0.3) → Linear(256) → LayerNorm → GELU
        → Dropout(0.2) → Linear(2)
    """
    def __init__(self, model_name: str = MODEL_NAME,
                 dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size      # 768 for SciBERT
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


def load_model(checkpoint: str = "best_model.pt",
               device: torch.device | None = None) -> SciCiteKeyModel:
    """Load the canonical model from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SciCiteKeyModel().to(device)
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        # Handle both plain state-dicts and wrapped checkpoints
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"  Loaded checkpoint: {checkpoint}")
    else:
        print(f"  WARNING: {checkpoint} not found — using random weights")
    model.eval()
    return model


# ── Dataset ───────────────────────────────────────────────────────────────────
class CitationDataset(Dataset):
    """Text-pair dataset for isKeyCitation classification."""
    def __init__(self, df: pd.DataFrame, tokenizer, abstracts: dict,
                 max_len: int = MAX_LEN, is_train: bool = False):
        self.records  = df.reset_index(drop=True).to_dict("records")
        self.tok      = tokenizer
        self.abstracts= abstracts
        self.max_len  = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r       = self.records[idx]
        context = str(r.get("string", ""))
        section = clean_section_name(r.get("sectionName", ""))
        label   = int(bool(r.get("isKeyCitation", False)))
        abst    = self.abstracts.get(str(r.get("citedPaperId", ""))) or ""
        text_a  = f"Section: {section}. {context}"
        enc = self.tok(
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


def load_dataset_split(path: str,
                       tokenizer,
                       abstracts: dict,
                       max_len: int = MAX_LEN,
                       is_train: bool = False) -> CitationDataset:
    """Load a JSONL split and return a CitationDataset."""
    df = pd.read_json(path, lines=True)
    df["isKeyCitation"] = df["isKeyCitation"].fillna(False).astype(bool)
    return CitationDataset(df, tokenizer, abstracts, max_len, is_train)
