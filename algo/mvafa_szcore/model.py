"""MV-AFA model — exact architecture matching the saved weights.

Reconstructed from eeg_chbmit_multiview_gated.py. Do NOT modify without
retraining, as the layer names and shapes must match best_model.pt exactly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLPBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerBranch(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, patch_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                                     stride=patch_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=4096)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).transpose(1, 2)
        x = self.pos_enc(x)
        x = self.norm(self.encoder(x))
        return x.mean(dim=1)


class MultiScaleTemporalTransformer(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, patch_sizes=(8, 16, 32), out_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            TemporalTransformerBranch(in_channels, d_model, nhead, num_layers, p, dropout)
            for p in patch_sizes
        ])
        self.proj = nn.Sequential(
            nn.Linear(d_model * len(patch_sizes), out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class Conv2DBranch(nn.Module):
    def __init__(self, kernel_size: int = 3, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class MultiScaleFrequencyCNN(nn.Module):
    def __init__(self, out_dim: int = 128, kernel_sizes=(3, 5, 7), dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleList([
            Conv2DBranch(k, out_dim, dropout) for k in kernel_sizes
        ])
        self.proj = nn.Sequential(
            nn.Linear(out_dim * len(kernel_sizes), out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, F)
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class GatedFusion(nn.Module):
    def __init__(self, n_experts: int = 4, feat_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_experts * feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, feats: list) -> tuple:
        stack = torch.stack(feats, dim=1)
        flat = torch.cat(feats, dim=1)
        weights = torch.softmax(self.gate(flat), dim=1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return self.post(fused), weights


class MultiViewSeizureNet(nn.Module):
    def __init__(self, time_channels: int, stat_dim: int, tda_dim: int,
                 common_dim: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.stat_branch = MLPBranch(stat_dim, 128, common_dim, dropout)
        self.tda_branch  = MLPBranch(tda_dim,  64,  common_dim, dropout)
        self.time_branch = MultiScaleTemporalTransformer(
            time_channels, 128, 4, 2, (8, 16, 32), common_dim, dropout)
        self.freq_branch = MultiScaleFrequencyCNN(common_dim, (3, 5, 7), dropout)
        self.fusion = GatedFusion(4, common_dim, common_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(common_dim, num_classes),
        )

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor,
                x_stat: torch.Tensor, x_tda: torch.Tensor) -> tuple:
        f_stat = self.stat_branch(x_stat)
        f_tda  = self.tda_branch(x_tda)
        f_time = self.time_branch(x_time)
        f_freq = self.freq_branch(x_freq)
        fused, gates = self.fusion([f_stat, f_tda, f_time, f_freq])
        return self.classifier(fused), gates
