"""
Visualization utilities: attention maps, physicochemical property EDA,
training history plots, and ROC curves.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
#  Training History
# ──────────────────────────────────────────
def plot_training_history(history_path: str, output_path: str = "results/plots/training_history.png"):
    """Plot loss and metrics curves from training JSON."""
    with open(history_path) as f:
        history = json.load(f)

    df = pd.DataFrame(history)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    axes[0].plot(df["epoch"], df["train_loss"], color="#58a6ff", label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], color="#ff7b72", label="Val")
    axes[0].set_title("Loss")
    axes[0].legend(facecolor="#30363d", labelcolor="white")

    axes[1].plot(df["epoch"], df["val_f1"], color="#3fb950", label="F1")
    axes[1].plot(df["epoch"], df["val_accuracy"], color="#f0883e", label="Accuracy")
    axes[1].set_title("Val Metrics")
    axes[1].legend(facecolor="#30363d", labelcolor="white")

    axes[2].plot(df["epoch"], df["val_roc_auc"], color="#a371f7", label="ROC-AUC")
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("ROC-AUC")
    axes[2].legend(facecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved training history plot → {output_path}")
    plt.close()


# ──────────────────────────────────────────
#  EDA: Physicochemical Properties
# ──────────────────────────────────────────
def plot_eda(csv_path: str, output_dir: str = "results/plots"):
    """Generate EDA plots for the AMP dataset."""
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    label_map = {0: "non-AMP", 1: "AMP"}
    df["Label"] = df["label"].map(label_map)

    cols = ["length", "isoelectric_point", "charge_at_ph7", "gravy", "molecular_weight"]
    cols = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    fig.suptitle("Physicochemical Property Distributions", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#0e1117")

    palette = {"AMP": "#58a6ff", "non-AMP": "#ff7b72"}

    for ax, col in zip(axes, cols):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        for label, grp in df.groupby("Label"):
            ax.hist(
                grp[col].dropna(), bins=40, alpha=0.7,
                label=label, color=palette[label], density=True
            )
        ax.set_title(col.replace("_", " ").title())
        ax.legend(facecolor="#30363d", labelcolor="white", fontsize=8)

    plt.tight_layout()
    out = Path(output_dir) / "eda_properties.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved EDA plot → {out}")
    plt.close()


# ──────────────────────────────────────────
#  Attention Map Visualization
# ──────────────────────────────────────────
def plot_attention_map(
    sequence: str,
    attention: np.ndarray,
    layer: int = -1,
    head: int = 0,
    output_path: str = "results/plots/attention.png",
):
    """
    Visualize attention weights for a single sequence.

    Args:
        sequence: Amino acid string (e.g., "GIKEFK")
        attention: numpy array of shape (num_layers, num_heads, seq_len, seq_len)
        layer: Which layer to visualize (-1 = last)
        head: Which head to visualize
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokens = ["[CLS]"] + list(sequence) + ["[SEP]"]
    tokens = tokens[:attention.shape[-1]]

    attn_matrix = attention[layer][head][:len(tokens), :len(tokens)]

    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), max(5, len(tokens) * 0.4)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.imshow(attn_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9, color="white")
    ax.set_yticklabels(tokens, fontsize=9, color="white")
    ax.set_title(f"Attention Map (Layer {layer}, Head {head})", color="white", fontsize=12)
    ax.tick_params(colors="white")

    fig.colorbar(im).ax.yaxis.label.set_color("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved attention map → {output_path}")
    plt.close()


# ──────────────────────────────────────────
#  Per-position attention highlight (1D)
# ──────────────────────────────────────────
def plot_sequence_importance(
    sequence: str,
    importance_scores: np.ndarray,
    output_path: str = "results/plots/sequence_importance.png",
    title: str = "Per-Position Attention Scores",
):
    """Bar chart of per-position attention scores."""
    tokens = list(sequence)
    scores = importance_scores[:len(tokens)]

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.4), 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    colors = cm.viridis(scores / scores.max())
    bars = ax.bar(range(len(tokens)), scores, color=colors)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=10, color="white")
    ax.tick_params(colors="white")
    ax.set_ylabel("Attention Score", color="white")
    ax.set_title(title, color="white", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved sequence importance plot → {output_path}")
    plt.close()
