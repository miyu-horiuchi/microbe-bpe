"""TinyGPT genome LM + pooled feature extraction.

Adapted from miyu-horiuchi/BPE (experiments/train_tiny_lm.py). The model is
deliberately small and tokenizer-agnostic: we train the *same* architecture
under the single-nucleotide and domain-BPE tokenizers, then freeze it and
mean-pool the final hidden states over a genome's windows to get one feature
vector per genome. Those vectors are what microbe-foundation's model.py consumes.

This is the honest, controlled half of the comparison: single-nt vs domain-BPE
at *matched capacity*. Evo2 is the SOTA single-nt reference at vastly larger
scale (see extract_evo2_features.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0


@dataclass
class LMConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    max_len: int = 512
    dropout: float = 0.1


def collate_pad(batch: list[torch.Tensor], pad_id: int = PAD_ID) -> torch.Tensor:
    max_len = max(x.size(0) for x in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : x.size(0)] = x
    return out


class WindowDataset(Dataset):
    def __init__(self, windows: list[str], encode_fn, max_len: int):
        self.samples: list[list[int]] = []
        for w in windows:
            ids = encode_fn(w)[:max_len]
            if len(ids) >= 8:
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.samples[idx], dtype=torch.long)


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Final hidden states (after LN, before the LM head)."""
        b, t = x.shape
        t = min(t, self.cfg.max_len)
        x = x[:, :t]
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.blocks(h, mask=mask)
        return self.ln(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))


def train_lm(
    windows: list[str],
    tokenizer,
    cfg: LMConfig,
    *,
    steps: int = 400,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[TinyGPT, dict]:
    """Train a TinyGPT causal LM on genome windows; return (model, stats)."""
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WindowDataset(windows, tokenizer.encode, cfg.max_len)
    if len(ds) == 0:
        raise ValueError("no usable windows to train on (need >= 8 tokens each)")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_pad
    )
    model = TinyGPT(tokenizer.vocab_size, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    losses: list[float] = []
    while step < steps:
        for batch in loader:
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits = model(inp)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD_ID
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if verbose and step % 50 == 0:
                print(f"    step {step}/{steps}  loss={loss.item():.4f}", flush=True)
            if step >= steps:
                break

    stats = {
        "n_windows": len(ds),
        "vocab_size": int(tokenizer.vocab_size),
        "params": int(sum(p.numel() for p in model.parameters())),
        "final_loss": round(losses[-1], 4) if losses else float("nan"),
        "steps": steps,
    }
    return model, stats


@torch.no_grad()
def genome_embedding(
    model: TinyGPT,
    tokenizer,
    windows: list[str],
    *,
    device: torch.device | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Mean-pool final hidden states over all (non-pad) positions of all windows.

    Returns a single [d_model] float32 vector for the genome.
    """
    device = device or next(model.parameters()).device
    model.eval()
    if not windows:
        return np.zeros(model.cfg.d_model, dtype=np.float32)

    enc = [torch.tensor(tokenizer.encode(w)[: model.cfg.max_len], dtype=torch.long)
           for w in windows]
    enc = [e for e in enc if e.numel() >= 1]
    if not enc:
        return np.zeros(model.cfg.d_model, dtype=np.float32)

    summed = torch.zeros(model.cfg.d_model, device=device)
    count = 0
    for i in range(0, len(enc), batch_size):
        chunk = enc[i : i + batch_size]
        padded = collate_pad(chunk).to(device)
        h = model.hidden(padded)               # [B, T, D]
        valid = (padded != PAD_ID).unsqueeze(-1).float()  # [B, T, 1]
        summed += (h * valid).sum(dim=(0, 1))
        count += int(valid.sum().item())
    vec = (summed / max(count, 1)).float().cpu().numpy()
    return vec.astype(np.float32)


@torch.no_grad()
def eval_bits_per_residue(
    model: TinyGPT, tokenizer, windows: list[str],
    *, device: torch.device | None = None, batch_size: int = 32,
) -> float:
    """Held-out compression metric (bonus diagnostic): NLL in bits / raw nucleotide."""
    device = device or next(model.parameters()).device
    model.eval()
    ds = WindowDataset(windows, tokenizer.encode, model.cfg.max_len)
    if len(ds) == 0:
        return float("nan")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    total_nll = 0.0
    total_res = 0
    for w in windows:
        total_res += len(w)
    for batch in loader:
        batch = batch.to(device)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = model(inp)
        nll = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=PAD_ID, reduction="sum",
        )
        total_nll += float(nll.item())
    return (total_nll / max(total_res, 1)) / math.log(2)
