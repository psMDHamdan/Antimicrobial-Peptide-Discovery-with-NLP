"""
AMP Classifier model using HuggingFace ProtBERT as backbone.
Fine-tuned for binary classification: AMP vs non-AMP.
"""
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
)
from typing import Optional


class AMPClassifier(nn.Module):
    """
    AMP Classifier using ProtBERT (or any BERT-style protein LM) backbone.
    Uses [CLS] token embedding fed through a classification head.
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert_bfd",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        # Load backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Optionally freeze encoder layers for faster fine-tuning
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )

        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        result = {"logits": logits}
        if output_attentions:
            result["attentions"] = outputs.attentions
        return result


class LightweightAMPClassifier(nn.Module):
    """
    Lightweight CNN-BiLSTM model for fast experimentation.
    Good baseline / comparison to ProtBERT.
    """

    def __init__(
        self,
        vocab_size: int = 25,      # 20 AA + special tokens
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: tuple = (3, 5, 7),
        lstm_hidden: int = 128,
        num_labels: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 200,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Parallel CNN filters with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, k, padding=k // 2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            for k in kernel_sizes
        ])

        cnn_out_size = num_filters * len(kernel_sizes)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout,
        )

        # Combine CNN + LSTM
        combined_size = cnn_out_size + lstm_hidden * 2

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embedding(input_ids)  # (B, L, E)

        # CNN path
        x_cnn = x.permute(0, 2, 1)   # (B, E, L)
        cnn_out = torch.cat([conv(x_cnn).squeeze(-1) for conv in self.convs], dim=1)

        # LSTM path
        x_lstm, _ = self.lstm(x)
        if attention_mask is not None:
            # Use last valid token's hidden state
            lengths = attention_mask.sum(dim=1).long()
            lstm_out = x_lstm[torch.arange(x_lstm.size(0)), lengths - 1]
        else:
            lstm_out = x_lstm[:, -1]

        # Combine
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        logits = self.classifier(combined)
        return {"logits": logits}
