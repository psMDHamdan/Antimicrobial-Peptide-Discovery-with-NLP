"""
Dataset and tokenization utilities for AMP classification.
Supports both ProtBERT tokenizer and lightweight custom tokenizer.
"""
import re
import logging
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# Simple character-level tokenizer for lightweight model
AA_VOCAB = {aa: i + 3 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_VOCAB["[PAD]"] = 0
AA_VOCAB["[CLS]"] = 1
AA_VOCAB["[SEP]"] = 2
AA_VOCAB["[UNK]"] = 23
AA_VOCAB["X"] = 24    # Unknown AA
ID_TO_AA = {v: k for k, v in AA_VOCAB.items()}


def tokenize_sequence_custom(seq: str, max_len: int = 200) -> dict:
    """Tokenize a peptide sequence using the simple character vocab."""
    tokens = [AA_VOCAB.get("[CLS]")]
    for aa in seq[:max_len - 2]:
        tokens.append(AA_VOCAB.get(aa, AA_VOCAB["[UNK]"]))
    tokens.append(AA_VOCAB.get("[SEP]"))

    padding_len = max_len - len(tokens)
    attention_mask = [1] * len(tokens) + [0] * padding_len
    tokens = tokens + [AA_VOCAB["[PAD]"]] * padding_len

    return {
        "input_ids": tokens[:max_len],
        "attention_mask": attention_mask[:max_len],
    }


def format_sequence_for_protbert(seq: str) -> str:
    """Add spaces between amino acids for ProtBERT tokenizer."""
    return " ".join(list(seq))


class AMPDataset(Dataset):
    """PyTorch Dataset for AMP sequences."""

    def __init__(
        self,
        sequences: list,
        labels: list,
        tokenizer=None,       # HuggingFace tokenizer (for ProtBERT)
        max_len: int = 128,
        use_protbert: bool = True,
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_protbert = use_protbert

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.use_protbert and self.tokenizer is not None:
            # Format for ProtBERT: "A B C D E ..."
            formatted = format_sequence_for_protbert(seq)
            enc = self.tokenizer(
                formatted,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
            }
        else:
            # Lightweight model tokenizer
            enc = tokenize_sequence_custom(seq, max_len=self.max_len)
            return {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
            }


def load_splits(
    csv_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataset CSV and split into train/val/test."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sequence", "label"]).copy()

    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), stratify=df["label"], random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_dataloaders(
    csv_path: str,
    tokenizer=None,
    batch_size: int = 32,
    max_len: int = 128,
    use_protbert: bool = True,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders."""
    train_df, val_df, test_df = load_splits(csv_path)

    def make_loader(df, shuffle):
        dataset = AMPDataset(
            sequences=df["sequence"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=tokenizer,
            max_len=max_len,
            use_protbert=use_protbert,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return make_loader(train_df, True), make_loader(val_df, False), make_loader(test_df, False)
