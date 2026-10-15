from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn

from pytorch_wavelets import DWT1DForward, DWT1DInverse


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.encoding.size(0):
            raise ValueError("sequence exceeds the positional-encoding capacity")
        return x + self.encoding[: x.size(1)].unsqueeze(0).to(x)


class ScaleEncoding(nn.Module):
    def __init__(self, d_model: int, scale_count: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(scale_count, d_model)

    def forward(
        self, x: torch.Tensor, scale_boundaries: Sequence[int],
    ) -> torch.Tensor:
        groups = []
        for scale, (left, right) in enumerate(
            zip(scale_boundaries[:-1], scale_boundaries[1:])
        ):
            index = torch.full(
                (x.size(0), right - left), scale,
                device=x.device, dtype=torch.long,
            )
            groups.append(x[:, left:right] + self.embedding(index))
        return torch.cat(groups, dim=1)


class DomainTransformDecoderLayer(nn.Module):
    def __init__(
        self,
        scale_boundaries: Sequence[int],
        time_lengths: Sequence[int],
        wavelet: str,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        boundary_mode: str = "symmetric",
    ) -> None:
        super().__init__()
        self.scale_boundaries = tuple(int(v) for v in scale_boundaries)
        self.time_lengths = tuple(int(v) for v in time_lengths)
        self.scale_attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True,
        )
        self.time_attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True,
        )
        self.scale_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(len(self.time_lengths))
        )
        self.feed_forward = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(len(self.time_lengths))
        )
        self.dropout = nn.Dropout(dropout)
        self.dwt = DWT1DForward(
            J=len(self.time_lengths) - 1,
            wave=wavelet,
            mode=boundary_mode,
        )
        self.idwt = DWT1DInverse(wave=wavelet, mode=boundary_mode)

    def _scale_causal_mask(self, device: torch.device) -> torch.Tensor:
        length = self.scale_boundaries[-1]
        allowed = torch.zeros(length, length, dtype=torch.bool, device=device)
        for left, right in zip(
            self.scale_boundaries[:-1], self.scale_boundaries[1:]
        ):
            allowed[left:right, left:right] = torch.tril(
                torch.ones(right - left, right - left, dtype=torch.bool, device=device)
            )
        return ~allowed

    def _to_time_domain(self, coefficients: torch.Tensor) -> torch.Tensor:
        groups = [
            coefficients[:, left:right].transpose(1, 2)
            for left, right in zip(
                self.scale_boundaries[:-1], self.scale_boundaries[1:]
            )
        ]
        return self.idwt((groups[-1], groups[:-1])).transpose(1, 2)[
            :, : self.time_lengths[0]
        ]

    def _to_wavelet_domain(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        low, highs = self.dwt(sequence.transpose(1, 2))
        return tuple(high.transpose(1, 2) for high in highs) + (
            low.transpose(1, 2),
        )

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_update, scale_weights = self.scale_attention(
            query,
            query,
            query,
            attn_mask=self._scale_causal_mask(query.device),
            need_weights=True,
            average_attn_weights=True,
        )
        query = self.scale_norm(query + self.dropout(scale_update))
        cross_update, cross_weights = self.cross_attention(
            query,
            memory,
            memory,
            need_weights=True,
            average_attn_weights=True,
        )
        cross_state = self.cross_norm(query + self.dropout(cross_update))
        time_state = self._to_time_domain(query)
        time_update, _ = self.time_attention(
            time_state,
            time_state,
            time_state,
            need_weights=False,
        )
        time_state = self.time_norm(time_state + self.dropout(time_update))
        transformed = self._to_wavelet_domain(time_state)
        cross_groups = [
            cross_state[:, left:right]
            for left, right in zip(
                self.scale_boundaries[:-1], self.scale_boundaries[1:]
            )
        ]
        outputs = []
        for wavelet_group, cross_group, feed_forward, norm in zip(
            transformed, cross_groups, self.feed_forward, self.output_norm
        ):
            state = wavelet_group + cross_group
            outputs.append(norm(state + self.dropout(feed_forward(state))))
        return torch.cat(outputs, dim=1), scale_weights, cross_weights


__all__ = [
    "PositionalEncoding",
    "ScaleEncoding",
    "DomainTransformDecoderLayer",
]
