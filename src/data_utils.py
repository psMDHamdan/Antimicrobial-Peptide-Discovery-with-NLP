"""
Data loading and preprocessing module for AMP Discovery.
Handles FASTA parsing, sequence cleaning, and physicochemical property calculation.
"""
import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)

# Standard 20 amino acids
STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

def parse_fasta(filepath: str, label: int, source: str) -> pd.DataFrame:
    """Parse a FASTA file and return a DataFrame with sequences and labels."""
    records = []
    filepath = Path(filepath)

    # Handle .txt extension from DBAASP
    with open(filepath, "r") as f:
        content = f.read()

    # Each header may have spaces after the ID — normalize
    # Use simple manual parser to handle edge cases
    current_id, current_seq = None, []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append({
                    "id": current_id,
                    "sequence": "".join(current_seq).upper(),
                    "label": label,
                    "source": source
                })
            parts = line[1:].strip().split()
            current_id = parts[0] if parts else "unknown"
            current_seq = []
        else:
            current_seq.append(line)

    # Last record
    if current_id is not None and current_seq:
        records.append({
            "id": current_id,
            "sequence": "".join(current_seq).upper(),
            "label": label,
            "source": source
        })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} sequences from {filepath.name} (label={label})")
    return df


def is_canonical(seq: str) -> bool:
    """Check if sequence contains only standard amino acids."""
    return bool(seq) and all(aa in STANDARD_AAS for aa in seq)


def clean_dataset(df: pd.DataFrame, min_len: int = 5, max_len: int = 200) -> pd.DataFrame:
    """Filter sequences by length and canonicality."""
    original_len = len(df)

    # Remove empty sequences
    df = df[df["sequence"].str.len() > 0].copy()

    # Filter by length
    df = df[(df["sequence"].str.len() >= min_len) & (df["sequence"].str.len() <= max_len)].copy()

    # Keep only canonical amino acids
    df = df[df["sequence"].apply(is_canonical)].copy()

    # Deduplicate by sequence
    df = df.drop_duplicates(subset=["sequence"]).copy()

    logger.info(f"Cleaned dataset: {original_len} → {len(df)} sequences")
    return df.reset_index(drop=True)


def compute_physico_properties(seq: str) -> dict:
    """Compute physicochemical properties of a peptide sequence."""
    try:
        analysis = ProteinAnalysis(seq)
        return {
            "length": len(seq),
            "molecular_weight": round(analysis.molecular_weight(), 2),
            "isoelectric_point": round(analysis.isoelectric_point(), 3),
            "charge_at_ph7": round(analysis.charge_at_pH(7.0), 3),
            "instability_index": round(analysis.instability_index(), 3),
            "gravy": round(analysis.gravy(), 4),       # Grand average of hydropathicity
            "aromaticity": round(analysis.aromaticity(), 4),
        }
    except Exception:
        return {
            "length": len(seq),
            "molecular_weight": np.nan,
            "isoelectric_point": np.nan,
            "charge_at_ph7": np.nan,
            "instability_index": np.nan,
            "gravy": np.nan,
            "aromaticity": np.nan,
        }


def build_dataset(
    apd_path: str,
    dbaasp_path: str,
    uniprot_path: str,
    output_path: str,
    max_neg_samples: Optional[int] = None,
    min_len: int = 5,
    max_len: int = 200,
) -> pd.DataFrame:
    """
    Load all sources, clean, balance, compute properties, and save to CSV.
    """
    logger.info("=== Building AMP Dataset ===")

    # --- Positive samples (AMP) ---
    apd_df = parse_fasta(apd_path, label=1, source="APD")
    dbaasp_df = parse_fasta(dbaasp_path, label=1, source="DBAASP")

    amp_df = pd.concat([apd_df, dbaasp_df], ignore_index=True)
    amp_df = clean_dataset(amp_df, min_len=min_len, max_len=max_len)

    # --- Negative samples (non-AMP) ---
    non_amp_df = parse_fasta(uniprot_path, label=0, source="UniProt")
    non_amp_df = clean_dataset(non_amp_df, min_len=min_len, max_len=max_len)

    # Balance negatives to 2x positives for mild imbalance (or use custom cap)
    n_neg = max_neg_samples if max_neg_samples else min(len(non_amp_df), len(amp_df) * 2)
    non_amp_df = non_amp_df.sample(n=n_neg, random_state=42).reset_index(drop=True)

    # --- Merge & cross-dedup ---
    full_df = pd.concat([amp_df, non_amp_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["sequence"]).copy()

    # Compute physicochemical properties
    logger.info("Computing physicochemical properties...")
    props = full_df["sequence"].apply(compute_physico_properties).apply(pd.Series)
    full_df = pd.concat([full_df, props], axis=1)

    # Final shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)

    logger.info(f"Saved dataset: {len(full_df)} sequences → {output_path}")
    logger.info(f"  AMP (label=1): {(full_df['label']==1).sum()}")
    logger.info(f"  non-AMP (label=0): {(full_df['label']==0).sum()}")
    return full_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = build_dataset(
        apd_path="data/raw/apd_natural.fasta",
        dbaasp_path="data/raw/peptides-fasta.txt",
        uniprot_path="data/raw/uniprot_non_amp.fasta",
        output_path="data/processed/amp_dataset.csv",
    )
    print(df.head())
    print(df["label"].value_counts())
