"""
Training pipeline for the AMP Classifier.
Supports both ProtBERT (heavy) and Lightweight CNN-BiLSTM models.
"""
import os
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm
import numpy as np

from src.model import AMPClassifier, LightweightAMPClassifier
from src.dataset import get_dataloaders

logger = logging.getLogger(__name__)


def compute_metrics(all_labels, all_preds, all_probs):
    """Compute evaluation metrics."""
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    report = classification_report(all_labels, all_preds, target_names=["non-AMP", "AMP"])
    return {"accuracy": acc, "f1": f1, "roc_auc": auc, "report": report}


def evaluate(model, loader, device, criterion):
    """Run evaluation loop."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output["logits"]

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    return metrics


def train(
    model_type: str = "lightweight",
    csv_path: str = "data/processed/amp_dataset.csv",
    output_dir: str = "models",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 2e-4,
    max_len: int = 128,
    weight_decay: float = 0.01,
    device: str = "auto",
    freeze_encoder: bool = False,
):
    """Full training loop."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Model & Tokenizer ---
    use_protbert = (model_type == "protbert")
    tokenizer = None

    if use_protbert:
        logger.info("Loading ProtBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        model = AMPClassifier(
            model_name="Rostlab/prot_bert_bfd",
            freeze_encoder=freeze_encoder,
        ).to(device)
        lr = 2e-5  # Lower LR for fine-tuning
    else:
        logger.info("Using Lightweight CNN-BiLSTM model")
        model = LightweightAMPClassifier().to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=csv_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        use_protbert=use_protbert,
    )

    # Compute class weights to handle imbalance
    import pandas as pd
    df = pd.read_csv(csv_path)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    weight = torch.tensor([n_pos / n_neg, 1.0], dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    history = []

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output["logits"]
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion)
        train_loss = total_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        })

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint_path = os.path.join(output_dir, f"best_{model_type}_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  ✓ Saved best model (F1={best_val_f1:.4f}) → {checkpoint_path}")

    # --- Final Test Evaluation ---
    logger.info("\n=== Final Test Evaluation ===")
    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(output_dir, f"best_{model_type}_model.pt"), map_location=device)
    )
    test_metrics = evaluate(model, test_loader, device, criterion)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1:       {test_metrics['f1']:.4f}")
    logger.info(f"Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    logger.info(f"\n{test_metrics['report']}")

    # Save training history & metrics
    with open(os.path.join(output_dir, f"{model_type}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, f"{model_type}_test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in test_metrics.items() if k != "report"}, f, indent=2)

    logger.info("Training complete.")
    return model, history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMP Classifier")
    parser.add_argument("--model", choices=["lightweight", "protbert"], default="lightweight")
    parser.add_argument("--csv", default="data/processed/amp_dataset.csv")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(
        model_type=args.model,
        csv_path=args.csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        freeze_encoder=args.freeze_encoder,
        device=args.device,
    )
