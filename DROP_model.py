from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DRI import DynamicRiskInteractionEncoder
from LOP import LOPEncoder
from module import DomainTransformDecoderLayer, PositionalEncoding, ScaleEncoding


@dataclass(frozen=True)
class DROPConfig:
    use_lop: bool = True
    use_dri: bool = True
    input_dim: int = 6
    output_dim: int = 6
    lop_dim: int = 64
    dri_dim: int = 64
    d_model: int = 192
    n_head: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 3
    dropout: float = 0.10
    wavelet: str = "haar"
    boundary_mode: str = "symmetric"
    wavelet_levels: int = 2
    fusion_total_residual_cap: float = 0.10
    fusion_eps: float = 1e-6
    decoder_lengths: Tuple[int, ...] = (30, 15, 8)
    scale_boundaries: Tuple[int, ...] = (0, 15, 23, 31)
    sample_interval_s: float = 2.0
    dri_hidden_dim: int = 96
    trajectory_min: Optional[Tuple[float, ...]] = None
    trajectory_max: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        for name in ("use_lop", "use_dri"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"model.{name} must be a boolean")


def _align_time(x: torch.Tensor, target_length: int) -> torch.Tensor:
    if x.size(1) == target_length:
        return x
    return F.interpolate(
        x.transpose(1, 2), size=target_length, mode="linear", align_corners=False,
    ).transpose(1, 2)


def _align_mask(x: torch.Tensor, target_length: int) -> torch.Tensor:
    if x.size(1) == target_length:
        return x
    return F.interpolate(
        x.transpose(1, 2), size=target_length, mode="nearest",
    ).transpose(1, 2)


def _smooth_three_points(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) < 2:
        return x
    padded = F.pad(x.transpose(1, 2), (1, 1), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=3, stride=1).transpose(1, 2)


class TrajectoryEncoder(nn.Module):
    def __init__(self, cfg: DROPConfig) -> None:
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg.encoder_layers, enable_nested_tensor=False,
        )

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.input_mlp(observed))


class EmbeddingFusion(nn.Module):
    def __init__(self, cfg: DROPConfig) -> None:
        super().__init__()
        self.lop_norm = nn.LayerNorm(cfg.lop_dim) if cfg.use_lop else None
        self.dri_norm = nn.LayerNorm(cfg.dri_dim) if cfg.use_dri else None
        self.lop_projection = nn.Linear(cfg.lop_dim, cfg.d_model) if cfg.use_lop else None
        self.dri_projection = nn.Linear(cfg.dri_dim, cfg.d_model) if cfg.use_dri else None
        self.fusion_mlp = nn.Sequential(
            nn.Linear((1 + cfg.use_lop + cfg.use_dri) * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        ) if cfg.use_lop or cfg.use_dri else None
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.total_residual_cap = float(cfg.fusion_total_residual_cap)
        self.eps = float(cfg.fusion_eps)

    def _cap_residual(
        self, residual: torch.Tensor, reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual_norm = torch.linalg.vector_norm(residual, dim=(1, 2), keepdim=True)
        reference_norm = torch.linalg.vector_norm(reference, dim=(1, 2), keepdim=True)
        limit = self.total_residual_cap * reference_norm
        ratio = residual_norm / (limit + self.eps)
        scale = torch.where(
            ratio > self.eps,
            torch.tanh(ratio) / ratio.clamp_min(self.eps),
            torch.ones_like(ratio),
        )
        capped = residual * scale
        diagnostics = {
            "fusion_residual_scale": scale.detach().flatten(),
            "fusion_residual_cap_hit_rate": (ratio > 1.0).float().mean().detach(),
            "fusion_residual_ratio": (
                torch.linalg.vector_norm(capped, dim=(1, 2), keepdim=True)
                / (reference_norm + self.eps)
            ).mean().detach(),
        }
        return capped, diagnostics

    def forward(
        self,
        trajectory: torch.Tensor,
        preference: torch.Tensor,
        preference_confidence: torch.Tensor,
        risk: torch.Tensor,
        lop_mask: torch.Tensor,
        dri_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        length = trajectory.size(1)
        inputs = [trajectory]
        if self.lop_projection is not None:
            preference = _align_time(preference, length)
            preference_confidence = _align_time(preference_confidence, length).clamp(0.0, 1.0)
            lop_mask = _align_mask(lop_mask, length)
            preference = _smooth_three_points(
                self.lop_projection(self.lop_norm(preference))
            ) * preference_confidence * lop_mask
            inputs.append(preference)
        else:
            preference = torch.zeros_like(trajectory)
            lop_mask = torch.zeros_like(trajectory[..., :1])
        if self.dri_projection is not None:
            risk = _align_time(risk, length)
            dri_mask = _align_mask(dri_mask, length)
            risk = _smooth_three_points(self.dri_projection(self.dri_norm(risk))) * dri_mask
            inputs.append(risk)
        else:
            risk = torch.zeros_like(trajectory)
            dri_mask = torch.zeros_like(trajectory[..., :1])
        active = torch.maximum(lop_mask, dri_mask)
        residual = (self.fusion_mlp(torch.cat(inputs, dim=-1)) * active
                    if self.fusion_mlp is not None else torch.zeros_like(trajectory))
        residual, diagnostics = self._cap_residual(residual, trajectory)
        memory = self.output_norm(trajectory + residual)
        diagnostics.update(
            {
                "lop_valid_mask_aligned": lop_mask.detach(),
                "dri_active_mask_aligned": dri_mask.detach(),
            }
        )
        return memory, preference, risk, diagnostics


class WaveletComponentPrompt(nn.Module):
    def __init__(self, cfg: DROPConfig) -> None:
        super().__init__()
        self.high_length = sum(cfg.decoder_lengths[1:])
        self.low_length = cfg.scale_boundaries[-1] - cfg.scale_boundaries[-2]
        self.use_lop, self.use_dri = cfg.use_lop, cfg.use_dri
        self.context_mlp = nn.Sequential(
            nn.Linear(cfg.input_dim + (1 + cfg.use_lop + cfg.use_dri) * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.low_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.high_tokens = nn.Parameter(
            torch.empty(1, self.high_length, cfg.d_model)
        )
        nn.init.normal_(self.high_tokens, mean=0.02, std=0.10)
        self.high_film = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.position = PositionalEncoding(cfg.d_model)
        self.scale = ScaleEncoding(cfg.d_model, len(cfg.decoder_lengths))
        self.scale_boundaries = cfg.scale_boundaries

    def forward(
        self,
        tail_difference: torch.Tensor,
        trajectory: torch.Tensor,
        preference: torch.Tensor,
        risk: torch.Tensor,
    ) -> torch.Tensor:
        length = tail_difference.size(1)
        inputs = [tail_difference, _align_time(trajectory, length)]
        if self.use_lop:
            inputs.append(_align_time(preference, length))
        if self.use_dri:
            inputs.append(_align_time(risk, length))
        context = self.context_mlp(torch.cat(inputs, dim=-1))
        low = F.adaptive_avg_pool1d(
            context.transpose(1, 2), self.low_length,
        ).transpose(1, 2)
        low = self.low_projection(low)
        gamma, beta = self.high_film(context.mean(dim=1)).chunk(2, dim=-1)
        high = self.high_tokens.expand(context.size(0), -1, -1)
        high = high * (1.0 + 0.25 * gamma[:, None]) + 0.10 * beta[:, None]
        prompt = torch.cat([high, low], dim=1)
        return self.scale(self.position(prompt), self.scale_boundaries)


class WaveletDecoder(nn.Module):
    def __init__(self, cfg: DROPConfig) -> None:
        super().__init__()
        if len(cfg.decoder_lengths) - 1 != cfg.wavelet_levels:
            raise ValueError("decoder wavelet levels do not match DROPConfig")
        self.scale_boundaries = cfg.scale_boundaries
        self.layers = nn.ModuleList(
            DomainTransformDecoderLayer(
                cfg.scale_boundaries,
                cfg.decoder_lengths,
                cfg.wavelet,
                cfg.d_model,
                cfg.n_head,
                dropout=cfg.dropout,
                boundary_mode=cfg.boundary_mode,
            )
            for _ in range(cfg.decoder_layers)
        )
        self.output_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, cfg.output_dim),
            )
            for _ in cfg.decoder_lengths
        )

    def forward(
        self, prompt: torch.Tensor, memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        state = prompt
        scale_attention = []
        cross_attention = []
        for layer in self.layers:
            state, scale_weights, cross_weights = layer(state, memory)
            scale_attention.append(scale_weights)
            cross_attention.append(cross_weights)
        groups = [
            head(state[:, left:right])
            for left, right, head in zip(
                self.scale_boundaries[:-1],
                self.scale_boundaries[1:],
                self.output_heads,
            )
        ]
        return torch.cat(groups, dim=1), {
            "decoder_state": state,
            "scale_attention": scale_attention,
            "cross_attention": cross_attention,
        }


class DROPModel(nn.Module):
    def __init__(
        self,
        cfg: Optional[DROPConfig] = None,
        dri_encoder: Optional[DynamicRiskInteractionEncoder] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DROPConfig()
        self.trajectory_encoder = TrajectoryEncoder(self.cfg)
        self.lop_encoder = LOPEncoder(output_dim=self.cfg.lop_dim) if self.cfg.use_lop else None
        if not self.cfg.use_dri and dri_encoder is not None:
            raise ValueError("dri_encoder cannot be supplied when use_dri is false")
        self.dri_encoder = dri_encoder or DynamicRiskInteractionEncoder(
            hidden_dim=self.cfg.dri_hidden_dim,
            output_dim=self.cfg.dri_dim,
            dt=self.cfg.sample_interval_s,
            dropout=self.cfg.dropout,
            trajectory_min=self.cfg.trajectory_min,
            trajectory_max=self.cfg.trajectory_max,
        ) if self.cfg.use_dri else None
        if self.dri_encoder is not None and self.dri_encoder.output_dim != self.cfg.dri_dim:
            raise ValueError("DRI output dimension must match DROPConfig.dri_dim")
        if self.dri_encoder is None:
            if self.cfg.trajectory_min is None or self.cfg.trajectory_max is None:
                raise ValueError("trajectory_min and trajectory_max are required")
            minimum = torch.as_tensor(self.cfg.trajectory_min, dtype=torch.float32)
            maximum = torch.as_tensor(self.cfg.trajectory_max, dtype=torch.float32)
            if minimum.shape != (6,) or maximum.shape != (6,) or not torch.all(maximum > minimum):
                raise ValueError("trajectory ranges must contain six increasing channels")
            self.register_buffer("trajectory_min", minimum.view(1, 1, 6))
            self.register_buffer("trajectory_scale", (maximum - minimum).view(1, 1, 6))
        self.fusion = EmbeddingFusion(self.cfg)
        self.prompt = WaveletComponentPrompt(self.cfg)
        self.decoder = WaveletDecoder(self.cfg)

    def _decode_path(
        self,
        trajectory: torch.Tensor,
        tail_difference: torch.Tensor,
        preference: torch.Tensor,
        preference_confidence: torch.Tensor,
        risk: torch.Tensor,
        lop_mask: torch.Tensor,
        dri_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        memory, preference_aligned, risk_aligned, fusion_aux = self.fusion(
            trajectory,
            preference,
            preference_confidence,
            risk,
            lop_mask,
            dri_mask,
        )
        prompt = self.prompt(
            tail_difference,
            trajectory[:, 1:],
            preference_aligned[:, 1:],
            risk_aligned[:, 1:],
        )
        coefficients, decoder_aux = self.decoder(prompt, memory)
        return coefficients, {
            "decoder_memory": memory,
            "q_prompt": prompt,
            "preference_projection": preference_aligned,
            "risk_projection": risk_aligned,
            **fusion_aux,
            **decoder_aux,
        }

    def forward(
        self,
        observed_scaled: torch.Tensor,
        kinematics_or_state: torch.Tensor,
        environment: torch.Tensor,
        u2u: Optional[Dict[str, torch.Tensor]] = None,
        u2o: Optional[Dict[str, torch.Tensor]] = None,
        return_lop_counterfactual: bool = False,
        environment_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if observed_scaled.size(1) != 15:
            raise ValueError("DROP expects 15 observed time steps")
        trajectory = self.trajectory_encoder(observed_scaled)
        if self.lop_encoder is not None:
            preference, preference_confidence, lop_mask = self.lop_encoder(
                kinematics_or_state, environment, environment_valid=environment_valid,
            )
        else:
            preference = observed_scaled.new_zeros(*observed_scaled.shape[:2], self.cfg.lop_dim)
            preference_confidence = lop_mask = torch.zeros_like(observed_scaled[..., :1])
        if self.dri_encoder is not None:
            risk, dri_aux = self.dri_encoder(
                observed_scaled, u2u=u2u, u2o=u2o,
                time_reliable=environment[..., 14:15], return_aux=True,
            )
            dri_mask = self.dri_encoder.last_active_mask.unsqueeze(-1).to(risk)
        else:
            risk = observed_scaled.new_zeros(*observed_scaled.shape[:2], self.cfg.dri_dim)
            dri_mask = torch.zeros_like(observed_scaled[..., :1])
            dri_aux = {}
        return_lop_counterfactual = return_lop_counterfactual and self.cfg.use_lop
        tail_difference = torch.diff(observed_scaled, dim=1)
        cpu_rng_state = None
        cuda_rng_state = None
        if self.training and return_lop_counterfactual:
            cpu_rng_state = torch.get_rng_state()
            if observed_scaled.is_cuda:
                cuda_rng_state = torch.cuda.get_rng_state(observed_scaled.device)
        coefficients, path_aux = self._decode_path(
            trajectory,
            tail_difference,
            preference,
            preference_confidence,
            risk,
            lop_mask,
            dri_mask,
        )
        aux: Dict[str, Any] = {
            "trajectory_embedding": trajectory,
            "preference_embedding": preference,
            "preference_confidence": preference_confidence,
            "lop_valid_mask": lop_mask,
            "risk_embedding": risk,
            "dri_active_mask": dri_mask,
            "dri": dri_aux,
            **path_aux,
        }
        if return_lop_counterfactual:
            if cpu_rng_state is not None:
                torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, observed_scaled.device)
            zero_preference = torch.zeros_like(preference)
            zero_confidence = torch.zeros_like(preference_confidence)
            zero_lop_mask = torch.zeros_like(lop_mask)
            counterfactual, counterfactual_aux = self._decode_path(
                trajectory,
                tail_difference,
                zero_preference,
                zero_confidence,
                risk,
                zero_lop_mask,
                dri_mask,
            )
            aux["lop_counterfactual_coefficients"] = counterfactual
            aux["lop_counterfactual_decoder_memory"] = counterfactual_aux[
                "decoder_memory"
            ]
            aux["lop_counterfactual_q_prompt"] = counterfactual_aux["q_prompt"]
        return coefficients, aux


__all__: Sequence[str] = ("DROPConfig", "DROPModel")
