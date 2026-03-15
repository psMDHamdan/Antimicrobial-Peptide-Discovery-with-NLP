"""
Inference module: predict AMP probability for a single or batch of sequences.
Supports ProtBERT and Lightweight models. Extracts attention weights for visualization.
"""
import logging
import torch
import numpy as np
from pathlib import Path

from src.model import AMPClassifier, LightweightAMPClassifier
from src.dataset import tokenize_sequence_custom, format_sequence_for_protbert
from src.data_utils import STANDARD_AAS

logger = logging.getLogger(__name__)


class AMPPredictor:
    """Unified predictor for both ProtBERT and Lightweight models."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "lightweight",  # "lightweight" | "protbert"
        device: str = "auto",
        max_len: int = 128,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_type = model_type
        self.max_len = max_len
        self.tokenizer = None

        if model_type == "protbert":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
            self.model = AMPClassifier(model_name="Rostlab/prot_bert_bfd").to(device)
        else:
            self.model = LightweightAMPClassifier().to(device)

        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(f"Loaded {model_type} model from {model_path}")

    def _clean_sequence(self, seq: str) -> str:
        seq = seq.upper().strip().replace(" ", "")
        return "".join(aa for aa in seq if aa in STANDARD_AAS)

    def predict(self, sequence: str, return_attention: bool = False) -> dict:
        """
        Predict AMP probability for a single sequence.

        Returns:
            dict with 'probability', 'label', and optionally 'attention'.
        """
        seq = self._clean_sequence(sequence)
        if not seq:
            return {"error": "No valid amino acids in sequence"}

        with torch.no_grad():
            if self.model_type == "protbert":
                formatted = format_sequence_for_protbert(seq)
                enc = self.tokenizer(
                    formatted, max_length=self.max_len,
                    padding="max_length", truncation=True, return_tensors="pt"
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=return_attention,
                )
            else:
                enc = tokenize_sequence_custom(seq, max_len=self.max_len)
                input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long).to(self.device)
                attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long).to(self.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)

            logits = output["logits"]
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_label = int(probs.argmax())

        result = {
            "sequence": seq,
            "amp_probability": float(probs[1]),
            "non_amp_probability": float(probs[0]),
            "label": "AMP" if pred_label == 1 else "non-AMP",
            "confidence": float(probs.max()),
        }

        if return_attention and "attentions" in output:
            # Stack all layers: (num_layers, num_heads, seq_len, seq_len)
            attn = torch.stack(output["attentions"], dim=0)
            attn = attn[:, 0].cpu().numpy()  # Remove batch dim
            result["attention"] = attn
            # Per-position importance: mean across layers and heads of CLS attention
            result["position_importance"] = attn.mean(axis=(0, 1))[0, 1:-1]  # CLS row, skip special tokens

        return result

    def predict_batch(self, sequences: list[str]) -> list[dict]:
        """Predict for a list of sequences."""
        return [self.predict(seq) for seq in sequences]

    def screen_candidates(self, sequences: list[str], threshold: float = 0.7) -> list[dict]:
        """Filter candidates above a probability threshold, sorted by confidence."""
        results = self.predict_batch(sequences)
        candidates = [r for r in results if r.get("amp_probability", 0) >= threshold]
        return sorted(candidates, key=lambda x: x["amp_probability"], reverse=True)
