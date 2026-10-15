from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import safe_atan2, trajectory_to_local_metres


KINEMATIC_FEATURES: Tuple[str, ...] = (
    "sin_heading_change",
    "cos_heading_change_minus_one",
    "signed_yaw_rate",
    "signed_tangential_acceleration",
    "normal_acceleration",
    "signed_climb_angle_rate",
    "signed_log_horizontal_curvature",
    "hover_state",
    "speed",
    "jerk_norm",
)

ENVIRONMENT_FEATURES: Tuple[str, ...] = (
    "weather_sunny",
    "weather_cloudy",
    "weather_fog",
    "weather_rain",
    "weather_snow",
    "wind_speed",
    "wind_direction_sin",
    "wind_direction_cos",
    "air_temperature",
    "open_space_ratio_100m",
    "road_ratio_100m",
    "green_space_ratio_100m",
    "water_ratio_100m",
    "obstacle_ratio_100m",
    "time_match_reliable",
)


def _moving_average(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    if kernel <= 1 or x.size(1) < 3:
        return x
    pad = kernel // 2
    return F.avg_pool1d(
        F.pad(x.unsqueeze(1), (pad, pad), mode="replicate"),
        kernel_size=kernel,
        stride=1,
    ).squeeze(1)


def _forward_difference(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) < 2:
        return torch.zeros_like(x)
    difference = x[:, 1:] - x[:, :-1]
    return torch.cat([difference, difference[:, -1:]], dim=1)


def _wrapped_angle_difference(x: torch.Tensor) -> torch.Tensor:
    difference = _forward_difference(x)
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def build_kinematic_features(
    observed_state: torch.Tensor,
    dt: float = 2.0,
    hover_speed: float = 1.0,
    smooth_kernel: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert [E, N, z, vE, vN, vz] into the ten LOP variables."""
    if observed_state.ndim != 3 or observed_state.size(-1) < 6:
        raise ValueError("observed_state must have shape [B,T,D] with D >= 6")
    if observed_state.size(1) < 2:
        return observed_state.new_zeros(*observed_state.shape[:2], 10)

    vx = _moving_average(observed_state[..., 3], smooth_kernel)
    vy = _moving_average(observed_state[..., 4], smooth_kernel)
    vz = _moving_average(observed_state[..., 5], smooth_kernel)

    velocity = torch.stack([vx, vy, vz], dim=-1)
    acceleration = torch.zeros_like(velocity)
    acceleration[:, 1:] = (velocity[:, 1:] - velocity[:, :-1]) / dt
    acceleration[:, 0] = acceleration[:, 1]
    jerk = torch.zeros_like(acceleration)
    jerk[:, 1:] = (acceleration[:, 1:] - acceleration[:, :-1]) / dt
    jerk[:, 0] = jerk[:, 1]

    speed = torch.linalg.norm(velocity, dim=-1)
    acceleration_norm = torch.linalg.norm(acceleration, dim=-1)
    jerk_norm = torch.linalg.norm(jerk, dim=-1)
    heading = safe_atan2(vy, vx, eps)
    heading_change = _wrapped_angle_difference(heading)
    yaw_rate = heading_change / dt

    tangential_acceleration = (
        (velocity * acceleration).sum(dim=-1) / (speed + eps)
    )
    normal_squared = (acceleration_norm.square() - tangential_acceleration.square()).clamp_min(0.0)
    normal_acceleration = torch.where(
        normal_squared > 0.0, normal_squared.clamp_min(eps * eps).sqrt(),
        torch.zeros_like(normal_squared),
    )

    horizontal_speed = torch.linalg.vector_norm(velocity[..., :2], dim=-1)
    climb_angle = safe_atan2(vz, horizontal_speed + eps, eps)
    climb_angle_rate = _wrapped_angle_difference(climb_angle) / dt
    curvature = (
        vx * acceleration[..., 1] - vy * acceleration[..., 0]
    ) / (horizontal_speed.clamp_min(hover_speed).pow(3) + eps)
    signed_log_curvature = torch.sign(curvature) * torch.log1p(curvature.abs())
    hover_state = (speed < hover_speed).to(observed_state.dtype)

    features = torch.stack(
        [
            torch.sin(heading_change),
            torch.cos(heading_change) - 1.0,
            yaw_rate,
            tangential_acceleration,
            normal_acceleration,
            climb_angle_rate,
            signed_log_curvature,
            hover_state,
            speed,
            jerk_norm,
        ],
        dim=-1,
    )
    return torch.nan_to_num(features)


class LOPEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 64,
        gamma_scale: float = 0.15,
        beta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("kin_mean", torch.zeros(1, 1, 10))
        self.register_buffer("kin_std", torch.ones(1, 1, 10))
        self.register_buffer("env_mean", torch.zeros(1, 1, 15))
        self.register_buffer("env_std", torch.ones(1, 1, 15))

        self.kinematic_conv = nn.Conv1d(10, 32, kernel_size=3, padding=1)
        self.kinematic_lstm = nn.LSTM(
            32, 64, batch_first=True, bidirectional=True,
        )
        self.environment_mlp = nn.Sequential(
            nn.Linear(15, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        self.gamma = nn.Linear(128, 128)
        self.beta = nn.Linear(128, 128)
        self.environment_gate = nn.Linear(128, 1)
        self.manoeuvre_gate = nn.Sequential(
            nn.Linear(10, 20),
            nn.GELU(),
            nn.Linear(20, 1),
        )
        self.preference_projection = nn.Linear(128, output_dim)
        self.preference_norm = nn.LayerNorm(output_dim)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.environment_gate.weight)
        nn.init.constant_(self.environment_gate.bias, -1.5)
        nn.init.zeros_(self.manoeuvre_gate[-1].weight)
        nn.init.constant_(self.manoeuvre_gate[-1].bias, -0.8)

    @torch.no_grad()
    def fit_normalization(self, training_dataset, history_length: int = 15, dt: float = 2.0) -> int:
        """Fit once on valid training observations; buffers are saved in state_dict."""
        count = 0
        mean, squared_deviation = torch.zeros(25, dtype=torch.float64), torch.zeros(25, dtype=torch.float64)
        for record_index, start in training_dataset.windows:
            record = training_dataset.records[record_index]
            stop = start + history_length
            observed = torch.as_tensor(record.trajectory_unscaled[start:stop], dtype=torch.float32)[None]
            if observed.size(1) != history_length:
                raise ValueError("incomplete observed window for LOP normalisation")
            physical, _ = trajectory_to_local_metres(observed)
            kinematics = build_kinematic_features(physical, dt=dt)
            environment = torch.as_tensor(record.environment[start:stop])[None]
            environment_valid = torch.as_tensor(record.environment_valid[start:stop])[None]
            valid = self._valid_mask(kinematics, environment, environment_valid).squeeze(-1)
            values = torch.cat([kinematics, environment], dim=-1)[valid].double()
            number = values.size(0)
            if not number:
                continue
            batch_mean = values.mean(dim=0)
            difference = batch_mean - mean
            total = count + number
            squared_deviation += ((values - batch_mean).square().sum(dim=0)
                                  + difference.square() * (count * number / total))
            mean += difference * (number / total)
            count = total
        if not count:
            raise ValueError("LOP statistics require valid training observations")
        standard_deviation = (squared_deviation / count).clamp_min(0).sqrt()
        standard_deviation = torch.where(standard_deviation < 1e-6, 1., standard_deviation)
        if not torch.isfinite(mean).all() or not torch.isfinite(standard_deviation).all():
            raise ValueError("non-finite LOP training statistics")
        for prefix, columns in (("kin", slice(0, 10)), ("env", slice(10, 25))):
            getattr(self, prefix + "_mean").copy_(mean[columns].reshape(1, 1, -1))
            getattr(self, prefix + "_std").copy_(standard_deviation[columns].reshape(1, 1, -1))
        return count

    def validate_normalization(self) -> None:
        for name in ("kin_mean", "kin_std", "env_mean", "env_std"):
            values = getattr(self, name)
            if not torch.isfinite(values).all() or (name.endswith("std") and (values <= 0).any()):
                raise ValueError(f"invalid LOP normalisation statistic: {name}")

    @staticmethod
    def _valid_mask(kinematics, environment, environment_valid=None):
        valid = (torch.isfinite(kinematics).all(dim=-1, keepdim=True)
                 & torch.isfinite(environment).all(dim=-1, keepdim=True)
                 & (environment[..., :14].abs().sum(dim=-1, keepdim=True) > 1e-8)
                 & (environment[..., 14:15] > 0.5))
        if environment_valid is not None:
            if environment_valid.ndim == 2:
                environment_valid = environment_valid.unsqueeze(-1)
            if environment_valid.shape != valid.shape:
                raise ValueError("environment_valid must have shape [B,T,1] or [B,T]")
            valid = valid & (environment_valid.to(environment.device) > 0.5)
        return valid

    def forward(
        self,
        observed_state_or_kinematics: torch.Tensor,
        environment: torch.Tensor,
        environment_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if environment.ndim != 3 or environment.size(-1) != 15:
            raise ValueError("environment must have shape [B,T,15]")
        if observed_state_or_kinematics.size(-1) == 10:
            kinematics = observed_state_or_kinematics
        else:
            kinematics = build_kinematic_features(observed_state_or_kinematics)
        if kinematics.shape[:2] != environment.shape[:2]:
            raise ValueError("kinematics and environment must share [B,T]")

        valid = self._valid_mask(kinematics, environment, environment_valid)

        kinematics = torch.nan_to_num(kinematics)
        environment = torch.nan_to_num(environment)
        kinematics = ((kinematics - self.kin_mean) / self.kin_std).clamp(-8.0, 8.0)
        environment = ((environment - self.env_mean) / self.env_std).clamp(-8.0, 8.0)

        temporal = F.relu(self.kinematic_conv(kinematics.transpose(1, 2)))
        temporal, _ = self.kinematic_lstm(temporal.transpose(1, 2))
        context = self.environment_mlp(environment)
        gamma = self.gamma_scale * torch.tanh(self.gamma(context))
        beta = self.beta_scale * torch.tanh(self.beta(context))
        environment_gate = torch.sigmoid(self.environment_gate(context))
        conditioned = temporal + environment_gate * (gamma * temporal + beta)

        preference = self.preference_norm(self.preference_projection(conditioned))
        manoeuvre_gate = torch.sigmoid(self.manoeuvre_gate(kinematics))
        confidence = 0.5 * (manoeuvre_gate + environment_gate)
        valid_float = valid.to(preference.dtype)
        return preference * valid_float, confidence * valid_float, valid_float


__all__ = [
    "KINEMATIC_FEATURES",
    "ENVIRONMENT_FEATURES",
    "build_kinematic_features",
    "LOPEncoder",
]
