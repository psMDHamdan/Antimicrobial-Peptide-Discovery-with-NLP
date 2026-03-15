"""
Variational Autoencoder (VAE) for de novo AMP sequence generation.
Treats peptide sequences as discrete sequences, learning a smooth
latent space from which new candidates can be sampled.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.dataset import AA_VOCAB, ID_TO_AA

VOCAB_SIZE = len(AA_VOCAB)
PAD_ID = AA_VOCAB["[PAD]"]
CLS_ID = AA_VOCAB["[CLS]"]
SEP_ID = AA_VOCAB["[SEP]"]


class PeptideVAE(nn.Module):
    """
    VAE for peptide sequence generation.
    Encoder: BiGRU → μ, log σ²
    Decoder: GRU with teacher forcing during training
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_seq_len: int = 60,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        # Encoder
        self.encoder_rnn = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.decoder_rnn = nn.GRU(
            embed_dim + latent_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        """x: (B, L) token ids"""
        emb = self.embedding(x)
        _, h = self.encoder_rnn(emb)
        # h: (num_layers * 2, B, hidden_dim)
        h_fwd = h[-2]  # Last forward layer
        h_bwd = h[-1]  # Last backward layer
        h_concat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden_dim * 2)
        mu = self.fc_mu(h_concat)
        logvar = self.fc_logvar(h_concat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target=None, teacher_forcing_ratio=0.5):
        """
        z: (B, latent_dim)
        target: (B, L) for teacher forcing during training
        """
        B = z.size(0)
        device = z.device

        # Initialize hidden state from z
        h = torch.tanh(self.latent_to_hidden(z))  # (B, hidden_dim * num_layers)
        h = h.view(B, self.num_layers, -1).permute(1, 0, 2).contiguous()

        # Start token
        inp = torch.full((B,), CLS_ID, dtype=torch.long, device=device)

        outputs = []
        max_len = target.size(1) if target is not None else self.max_seq_len

        for t in range(max_len):
            inp_emb = self.embedding(inp)  # (B, embed_dim)
            rnn_inp = torch.cat([inp_emb, z], dim=-1).unsqueeze(1)  # (B, 1, embed_dim+latent_dim)
            out, h = self.decoder_rnn(rnn_inp, h)
            logit = self.output_proj(out.squeeze(1))  # (B, vocab_size)
            outputs.append(logit)

            # Teacher forcing
            if target is not None and np.random.random() < teacher_forcing_ratio:
                inp = target[:, t]
            else:
                inp = logit.argmax(dim=-1)

        return torch.stack(outputs, dim=1)  # (B, L, vocab_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, target=x[:, 1:])  # Skip [CLS] for decoding targets
        return logits, mu, logvar

    def generate(
        self,
        num_samples: int = 10,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[str]:
        """Generate novel peptide sequences by sampling from latent space."""
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)

        with torch.no_grad():
            h = torch.tanh(self.latent_to_hidden(z))
            h = h.view(num_samples, self.num_layers, -1).permute(1, 0, 2).contiguous()

            inp = torch.full((num_samples,), CLS_ID, dtype=torch.long, device=device)
            sequences = [[] for _ in range(num_samples)]
            done = [False] * num_samples

            for _ in range(self.max_seq_len):
                inp_emb = self.embedding(inp)
                rnn_inp = torch.cat([inp_emb, z], dim=-1).unsqueeze(1)
                out, h = self.decoder_rnn(rnn_inp, h)
                logit = self.output_proj(out.squeeze(1)) / temperature

                # Mask special tokens to avoid generating them mid-sequence
                logit[:, PAD_ID] = -float("inf")
                logit[:, CLS_ID] = -float("inf")

                probs = torch.softmax(logit, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                for i in range(num_samples):
                    if not done[i]:
                        tok = next_tokens[i].item()
                        if tok == SEP_ID:
                            done[i] = True
                        else:
                            aa = ID_TO_AA.get(tok, "")
                            if aa and aa not in {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}:
                                sequences[i].append(aa)

                inp = next_tokens

        return ["".join(s) for s in sequences if len(s) >= 5]


def vae_loss(logits, targets, mu, logvar, kl_weight=0.001):
    """ELBO = Reconstruction Loss + KL Divergence."""
    B, L, V = logits.shape
    recon_loss = F.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        ignore_index=PAD_ID,
    )
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
