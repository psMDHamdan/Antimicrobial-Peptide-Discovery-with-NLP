#!/usr/bin/env python3
"""
Master pipeline runner for AMP Discovery project.
Runs all steps in order:
  1. Data preparation
  2. EDA plots
  3. Lightweight model training
  4. (Optional) ProtBERT fine-tuning
  5. Evaluation & visualization
"""
import logging
import sys
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_data_pipeline():
    logger.info("=== Step 1: Data Pipeline ===")
    from src.data_utils import build_dataset
    df = build_dataset(
        apd_path="data/raw/apd_natural.fasta",
        dbaasp_path="data/raw/peptides-fasta.txt",
        uniprot_path="data/raw/uniprot_non_amp.fasta",
        output_path="data/processed/amp_dataset.csv",
    )
    logger.info(f"Dataset ready: {len(df):,} sequences")
    return df


def run_eda():
    logger.info("=== Step 2: EDA Plots ===")
    from src.visualization import plot_eda
    plot_eda("data/processed/amp_dataset.csv", output_dir="results/plots")


def run_training(model_type: str, epochs: int, batch_size: int):
    logger.info(f"=== Step 3: Training ({model_type}) ===")
    from src.train import train
    model, history, metrics = train(
        model_type=model_type,
        csv_path="data/processed/amp_dataset.csv",
        output_dir="models",
        epochs=epochs,
        batch_size=batch_size,
    )
    return model, history, metrics


def run_visualizations(model_type: str):
    logger.info("=== Step 4: Training Curves ===")
    from src.visualization import plot_training_history
    history_path = f"models/{model_type}_history.json"
    plot_training_history(history_path, f"results/plots/{model_type}_training.png")


def run_vae_training(epochs: int = 30):
    """Optional: train the VAE generator on AMP sequences."""
    import torch
    import pandas as pd
    from torch.optim import Adam
    from src.generator import PeptideVAE, vae_loss
    from src.dataset import tokenize_sequence_custom, AA_VOCAB

    logger.info("=== Step 5: VAE Training ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("data/processed/amp_dataset.csv")
    amp_seqs = df[df["label"] == 1]["sequence"].tolist()

    logger.info(f"Training VAE on {len(amp_seqs)} AMP sequences")
    max_len = 64

    # Tokenize
    def tokenize(seq):
        enc = tokenize_sequence_custom(seq, max_len=max_len)
        return torch.tensor(enc["input_ids"], dtype=torch.long)

    import torch.utils.data as data
    class SeqDataset(data.Dataset):
        def __init__(self, seqs): self.seqs = seqs
        def __len__(self): return len(self.seqs)
        def __getitem__(self, i): return tokenize(self.seqs[i])

    loader = data.DataLoader(SeqDataset(amp_seqs), batch_size=64, shuffle=True, num_workers=2)

    vae = PeptideVAE(max_seq_len=max_len).to(device)
    optimizer = Adam(vae.parameters(), lr=1e-3)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        vae.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Targets are input shifted by 1 (skip [CLS])
            logits, mu, logvar = vae(batch)
            target = batch[:, 1:]
            # Pad or trim to match logits length
            L = min(logits.size(1), target.size(1))
            loss, recon, kl = vae_loss(logits[:, :L], target[:, :L], mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        if epoch % 5 == 0:
            logger.info(f"VAE Epoch {epoch}/{epochs} | Loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(vae.state_dict(), "models/peptide_vae.pt")

    logger.info(f"VAE training complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="AMP Discovery Pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plots")
    parser.add_argument("--model", choices=["lightweight", "protbert", "both"],
                        default="lightweight", help="Which model(s) to train")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train-vae", action="store_true", help="Also train VAE generator")
    parser.add_argument("--vae-epochs", type=int, default=30)
    args = parser.parse_args()

    # Step 1: Data
    if not args.skip_data:
        run_data_pipeline()

    # Step 2: EDA
    if not args.skip_eda:
        run_eda()

    # Step 3: Train classifier(s)
    models_to_train = ["lightweight", "protbert"] if args.model == "both" else [args.model]
    for m in models_to_train:
        run_training(m, args.epochs, args.batch_size)
        run_visualizations(m)

    # Step 4: Optional VAE training
    if args.train_vae:
        run_vae_training(epochs=args.vae_epochs)

    logger.info("\n✅ Pipeline complete!")
    logger.info("Run `streamlit run app.py` to launch the web interface.")


if __name__ == "__main__":
    main()
