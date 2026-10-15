from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import safe_atan2


EGO_FEATURES: Tuple[str, ...] = (
    "normalised_longitude",
    "normalised_latitude",
    "normalised_altitude",
    "velocity_east",
    "velocity_north",
    "velocity_up",
    "acceleration_east",
    "acceleration_north",
    "acceleration_up",
    "jerk_east",
    "jerk_north",
    "jerk_up",
    "heading",
)

U2U_FEATURES: Tuple[str, ...] = (
    "relative_east",
    "relative_north",
    "relative_up",
    "neighbour_velocity_east",
    "neighbour_velocity_north",
    "neighbour_velocity_up",
    "neighbour_acceleration_east",
    "neighbour_acceleration_north",
    "neighbour_acceleration_up",
    "neighbour_jerk_east",
    "neighbour_jerk_north",
    "neighbour_jerk_up",
    "distance",
    "horizontal_distance",
    "horizontal_bearing",
    "vertical_angle",
    "time_to_collision",
    "collision_risk",
    "risk_change",
)

U2O_FEATURES: Tuple[str, ...] = (
    "relative_east",
    "relative_north",
    "height_difference",
    "normalised_obstacle_height",
    "validity",
)


def _positive_parameter(value: float, eps: float = 0.05) -> nn.Parameter:
    raw = math.log(math.expm1(max(value - eps, 1e-4)))
    return nn.Parameter(torch.tensor(raw, dtype=torch.float32))


def _previous_by_id(
    values: torch.Tensor, valid: torch.Tensor, ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    previous = torch.zeros_like(values)
    matched = torch.zeros_like(valid)
    if values.size(1) > 1:
        pairs = (
            (ids[:, 1:, :, None] == ids[:, :-1, None, :])
            & (ids[:, 1:, :, None] >= 0)
            & valid[:, 1:, :, None] & valid[:, :-1, None, :]
        )
        matched[:, 1:] = pairs.any(dim=-1)
        index = pairs.to(torch.int64).argmax(dim=-1)
        previous[:, 1:] = torch.gather(
            values[:, :-1], 2, index.unsqueeze(-1).expand(*index.shape, values.size(-1)),
        )
    return previous, matched


class HeterogeneousGATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.u2u_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.u2o_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.u2u_attention = nn.Parameter(torch.empty(2 * output_dim, 1))
        self.u2o_attention = nn.Parameter(torch.empty(2 * output_dim, 1))
        self.u2u_gate = nn.Parameter(torch.zeros(()))
        self.u2o_gate = nn.Parameter(torch.zeros(()))
        self.residual = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.u2u_attention, gain=1.414)
        nn.init.xavier_uniform_(self.u2o_attention, gain=1.414)

    def _message(
        self,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        risk_bias: torch.Tensor,
        projection: nn.Linear,
        attention_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = projection(nodes)
        source = projected @ attention_vector[: self.output_dim]
        target = projected @ attention_vector[self.output_dim :]
        logits = self.activation(source + target.transpose(1, 2)) + risk_bias
        logits = logits.masked_fill(~adjacency, -1e9)
        weights = F.softmax(logits, dim=-1)
        active = adjacency.any(dim=-1, keepdim=True)
        weights = torch.where(active, weights, torch.zeros_like(weights))
        weights = self.dropout(weights)
        return weights @ projected, weights

    def forward(
        self,
        nodes: torch.Tensor,
        adjacency_u2u: torch.Tensor,
        adjacency_u2o: torch.Tensor,
        bias_u2u: torch.Tensor,
        bias_u2o: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        message_u2u, attention_u2u = self._message(
            nodes, adjacency_u2u, bias_u2u,
            self.u2u_projection, self.u2u_attention,
        )
        message_u2o, attention_u2o = self._message(
            nodes, adjacency_u2o, bias_u2o,
            self.u2o_projection, self.u2o_attention,
        )
        updated = (
            self.residual(nodes)
            + torch.sigmoid(self.u2u_gate) * message_u2u
            + torch.sigmoid(self.u2o_gate) * message_u2o
        )
        return F.relu(self.norm(updated)), {
            "u2u_attention": attention_u2u,
            "u2o_attention": attention_u2o,
        }


class DynamicRiskInteractionEncoder(nn.Module):
    U2U_RADIUS_M = 500.0
    U2O_RADIUS_M = 200.0
    MAX_U2U = 8
    BUILDING_K = 6

    def __init__(
        self,
        hidden_dim: int = 96,
        output_dim: int = 64,
        dt: float = 2.0,
        dropout: float = 0.1,
        safe_base_m: float = 35.0,
        safe_speed_gain_s: float = 2.0,
        safe_min_m: float = 20.0,
        safe_max_m: float = 120.0,
        ttc_cap_s: float = 99.0,
        height_reference_m: float = 100.0,
        trajectory_min: Optional[Sequence[float]] = None,
        trajectory_max: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if trajectory_min is None or trajectory_max is None:
            raise ValueError("trajectory_min and trajectory_max are required")
        if len(trajectory_min) != 6 or len(trajectory_max) != 6:
            raise ValueError("trajectory ranges must contain six channels")
        minimum = torch.as_tensor(trajectory_min, dtype=torch.float32)
        maximum = torch.as_tensor(trajectory_max, dtype=torch.float32)
        if not torch.all(maximum > minimum):
            raise ValueError("each trajectory maximum must exceed its minimum")
        self.register_buffer("trajectory_min", minimum.view(1, 1, 6))
        self.register_buffer(
            "trajectory_scale", (maximum - minimum).view(1, 1, 6),
        )
        self.dt = float(dt)
        self.safe_base_m = float(safe_base_m)
        self.safe_speed_gain_s = float(safe_speed_gain_s)
        self.safe_min_m = float(safe_min_m)
        self.safe_max_m = float(safe_max_m)
        self.ttc_cap_s = float(ttc_cap_s)
        self.height_reference_m = float(height_reference_m)
        self.output_dim = int(output_dim)
        self.coefficient_eps = 0.05

        self.alpha_raw = _positive_parameter(0.6, self.coefficient_eps)
        self.beta_raw = _positive_parameter(1.5, self.coefficient_eps)
        self.gamma_raw = _positive_parameter(1.5, self.coefficient_eps)
        self.omega_raw = _positive_parameter(1.5, self.coefficient_eps)
        self.lambda_u2u_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.lambda_u2o_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

        self.ego_projection = nn.Linear(13, hidden_dim)
        self.u2u_projection = nn.Linear(len(U2U_FEATURES), hidden_dim)
        self.u2o_projection = nn.Linear(5, hidden_dim)
        self.gat1 = HeterogeneousGATLayer(hidden_dim, hidden_dim, dropout)
        self.gat2 = HeterogeneousGATLayer(hidden_dim, output_dim, dropout)
        self.last_active_mask: Optional[torch.Tensor] = None
        self.last_aux_info: Dict[str, torch.Tensor] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw) + self.coefficient_eps

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw) + self.coefficient_eps

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw) + self.coefficient_eps

    @property
    def omega(self) -> torch.Tensor:
        return F.softplus(self.omega_raw) + self.coefficient_eps

    @property
    def lambda_u2u(self) -> torch.Tensor:
        return F.softplus(self.lambda_u2u_raw)

    @property
    def lambda_u2o(self) -> torch.Tensor:
        return F.softplus(self.lambda_u2o_raw)

    @staticmethod
    def _pad(
        tensor: torch.Tensor, target: int, value: float = 0.0,
    ) -> torch.Tensor:
        if tensor.size(2) >= target:
            return tensor[:, :, :target]
        shape = list(tensor.shape)
        shape[2] = target - tensor.size(2)
        padding = torch.full(
            shape, value, device=tensor.device, dtype=tensor.dtype,
        )
        return torch.cat([tensor, padding], dim=2)

    def _select_u2u(
        self, u2u: Dict[str, torch.Tensor], ego_velocity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = u2u["rel_pos_m"].to(ego_velocity)
        velocity = u2u["velocity_mps"].to(ego_velocity)
        valid = u2u["valid_mask"].to(device=ego_velocity.device, dtype=torch.bool)
        ids = u2u.get("id_codes")
        if ids is None:
            ids = torch.full(valid.shape, -1, device=valid.device, dtype=torch.long)
        else:
            ids = ids.to(device=valid.device, dtype=torch.long)

        distance = torch.linalg.norm(rel, dim=-1)
        valid = valid & torch.isfinite(distance) & (distance <= self.U2U_RADIUS_M)
        order = torch.argsort(torch.where(
            valid, distance, torch.full_like(distance, float("inf")),
        ), dim=2)[:, :, : self.MAX_U2U]
        index3 = order.unsqueeze(-1).expand(*order.shape, 3)
        rel = torch.gather(rel, 2, index3)
        velocity = torch.gather(velocity, 2, index3)
        valid = torch.gather(valid, 2, order)
        ids = torch.gather(ids, 2, order)
        return (
            self._pad(rel, self.MAX_U2U),
            self._pad(velocity, self.MAX_U2U),
            self._pad(valid, self.MAX_U2U),
            self._pad(ids, self.MAX_U2U, -1),
        )

    def _select_u2o(
        self, u2o: Dict[str, torch.Tensor], reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_xy = u2o["rel_xy_m"].to(reference)
        height = u2o["height_m"].to(reference)
        distance = u2o["dist_m"].to(reference)
        valid = u2o["valid_mask"].to(device=reference.device, dtype=torch.bool)
        valid = valid & torch.isfinite(distance) & (distance <= self.U2O_RADIUS_M)
        order = torch.argsort(torch.where(
            valid, distance, torch.full_like(distance, float("inf")),
        ), dim=2)[:, :, : self.BUILDING_K]
        index2 = order.unsqueeze(-1).expand(*order.shape, 2)
        rel_xy = torch.gather(rel_xy, 2, index2)
        height = torch.gather(height, 2, order)
        valid = torch.gather(valid, 2, order)
        return (
            self._pad(rel_xy, self.BUILDING_K),
            self._pad(height, self.BUILDING_K),
            self._pad(valid, self.BUILDING_K),
        )

    def _safe_radius(self, speed: torch.Tensor) -> torch.Tensor:
        return (self.safe_base_m + self.safe_speed_gain_s * speed).clamp(
            self.safe_min_m, self.safe_max_m,
        )

    def _physical_motion(
        self, normalised_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        physical = (
            normalised_state[..., :6] * self.trajectory_scale
            + self.trajectory_min
        )
        velocity = physical[..., 3:6]
        altitude_m = physical[..., 2] * 10.0
        return velocity, altitude_m

    def forward(
        self,
        ego_state_normalised: torch.Tensor,
        u2u: Optional[Dict[str, torch.Tensor]] = None,
        u2o: Optional[Dict[str, torch.Tensor]] = None,
        time_reliable: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if ego_state_normalised.ndim != 3 or ego_state_normalised.size(-1) < 6:
            raise ValueError("ego_state_normalised must have shape [B,T,6+]")
        if not torch.isfinite(ego_state_normalised[..., :6]).all():
            raise ValueError("ego_state_normalised contains non-finite values")
        batch, steps = ego_state_normalised.shape[:2]
        device, dtype = ego_state_normalised.device, ego_state_normalised.dtype
        velocity, altitude = self._physical_motion(ego_state_normalised)
        acceleration = torch.zeros_like(velocity)
        if steps > 1:
            acceleration[:, 1:] = (velocity[:, 1:] - velocity[:, :-1]) / self.dt
            acceleration[:, 0] = acceleration[:, 1]
        jerk = torch.zeros_like(acceleration)
        if steps > 1:
            jerk[:, 1:] = (acceleration[:, 1:] - acceleration[:, :-1]) / self.dt
            jerk[:, 0] = jerk[:, 1]
        heading = safe_atan2(velocity[..., 1], velocity[..., 0]).unsqueeze(-1)
        ego_features = torch.cat(
            [
                ego_state_normalised[..., :3],
                velocity / 30.0,
                acceleration / 10.0,
                jerk / 10.0,
                heading / math.pi,
            ], dim=-1,
        )

        reliability = torch.ones(batch, steps, 1, device=device, dtype=dtype)
        if time_reliable is not None:
            reliability = time_reliable.to(device=device, dtype=dtype)
            if reliability.ndim == 2:
                reliability = reliability.unsqueeze(-1)
            reliability = reliability.clamp(0.0, 1.0)

        u2u_features = torch.zeros(
            batch, steps, self.MAX_U2U, len(U2U_FEATURES), device=device, dtype=dtype,
        )
        valid_u2u = torch.zeros(
            batch, steps, self.MAX_U2U, device=device, dtype=torch.bool,
        )
        bias_value_u2u = torch.zeros(
            batch, steps, self.MAX_U2U, device=device, dtype=dtype,
        )
        if u2u is not None:
            rel, neighbour_velocity, valid_u2u, ids = self._select_u2u(u2u, velocity)
            distance = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
            relative_velocity = neighbour_velocity - velocity.unsqueeze(2)
            relative_speed = torch.linalg.norm(relative_velocity, dim=-1)
            closing_speed = (-(rel * relative_velocity).sum(dim=-1) / distance).clamp_min(0.0)
            ttc = torch.where(
                closing_speed > 0.1,
                distance / closing_speed.clamp_min(0.1),
                torch.full_like(distance, self.ttc_cap_s),
            ).clamp_max(self.ttc_cap_s)
            vertical_decay = torch.exp(-rel[..., 2].abs() / 50.0)
            safe_radius = self._safe_radius(relative_speed)
            intrusion = ((safe_radius - distance) / safe_radius.clamp_min(1e-6)).clamp_min(0.0)
            collision_risk = (
                torch.exp(-self.alpha * ttc)
                * vertical_decay
                * reliability
            )

            previous_velocity, matched = _previous_by_id(
                neighbour_velocity, valid_u2u, ids,
            )
            neighbour_acceleration = torch.where(
                matched.unsqueeze(-1),
                (neighbour_velocity - previous_velocity) / self.dt,
                torch.zeros_like(neighbour_velocity),
            )
            previous_acceleration, acceleration_matched = _previous_by_id(
                neighbour_acceleration, matched, ids,
            )
            neighbour_jerk = torch.where(
                acceleration_matched.unsqueeze(-1),
                (neighbour_acceleration - previous_acceleration) / self.dt,
                torch.zeros_like(neighbour_acceleration),
            )
            risk_previous, risk_matched = _previous_by_id(
                collision_risk.unsqueeze(-1), valid_u2u, ids,
            )
            risk_previous = torch.where(
                risk_matched, risk_previous.squeeze(-1), collision_risk,
            )
            bearing = safe_atan2(rel[..., 1], rel[..., 0])
            horizontal_distance = torch.linalg.vector_norm(rel[..., :2], dim=-1)
            vertical_angle = safe_atan2(rel[..., 2], horizontal_distance)
            u2u_features = torch.cat(
                [
                    rel / self.U2U_RADIUS_M,
                    neighbour_velocity / 30.0,
                    neighbour_acceleration / 10.0,
                    neighbour_jerk / 10.0,
                    (distance / self.U2U_RADIUS_M).unsqueeze(-1),
                    (horizontal_distance / self.U2U_RADIUS_M).unsqueeze(-1),
                    (bearing / math.pi).unsqueeze(-1),
                    (vertical_angle / (0.5 * math.pi)).unsqueeze(-1),
                    (ttc / self.ttc_cap_s).unsqueeze(-1),
                    collision_risk.unsqueeze(-1),
                    (collision_risk - risk_previous).unsqueeze(-1),
                ], dim=-1,
            ) * valid_u2u.unsqueeze(-1)
            bias_value_u2u = (
                self.lambda_u2u
                * (
                    self.beta * torch.exp(-self.alpha * ttc) * reliability
                    + self.gamma * intrusion
                )
                * vertical_decay
                * valid_u2u
            )

        u2o_features = torch.zeros(
            batch, steps, self.BUILDING_K, 5, device=device, dtype=dtype,
        )
        valid_u2o = torch.zeros(
            batch, steps, self.BUILDING_K, device=device, dtype=torch.bool,
        )
        bias_value_u2o = torch.zeros(
            batch, steps, self.BUILDING_K, device=device, dtype=dtype,
        )
        if u2o is not None:
            rel_xy, obstacle_height, valid_u2o = self._select_u2o(u2o, velocity)
            distance = torch.linalg.norm(rel_xy, dim=-1).clamp_min(1e-6)
            direction = rel_xy / distance.unsqueeze(-1)
            closing_speed = (
                velocity[..., :2].unsqueeze(2) * direction
            ).sum(dim=-1).clamp_min(0.0)
            ttc = torch.where(
                closing_speed > 0.1,
                distance / closing_speed.clamp_min(0.1),
                torch.full_like(distance, self.ttc_cap_s),
            ).clamp_max(self.ttc_cap_s)
            safe_radius = self._safe_radius(closing_speed)
            intrusion = ((safe_radius - distance) / safe_radius.clamp_min(1e-6)).clamp_min(0.0)
            height_risk = torch.sigmoid(
                (obstacle_height + 20.0 - altitude.unsqueeze(-1)) / 10.0,
            )
            u2o_features = torch.stack(
                [
                    rel_xy[..., 0] / self.U2O_RADIUS_M,
                    rel_xy[..., 1] / self.U2O_RADIUS_M,
                    (obstacle_height - altitude.unsqueeze(-1)) / 100.0,
                    (obstacle_height / self.height_reference_m).clamp(0.0, 2.0),
                    valid_u2o.to(dtype),
                ], dim=-1,
            ) * valid_u2o.unsqueeze(-1)
            bias_value_u2o = (
                self.lambda_u2o
                * (
                    self.beta * torch.exp(-self.alpha * ttc) * reliability
                    + self.gamma * intrusion
                    + self.omega * height_risk
                )
                * valid_u2o
            )

        ego_nodes = self.ego_projection(ego_features).unsqueeze(2)
        u2u_nodes = self.u2u_projection(u2u_features)
        u2o_nodes = self.u2o_projection(u2o_features)
        nodes = torch.cat([ego_nodes, u2u_nodes, u2o_nodes], dim=2)
        node_count = 1 + self.MAX_U2U + self.BUILDING_K

        adjacency_u2u = torch.zeros(
            batch, steps, node_count, node_count, device=device, dtype=torch.bool,
        )
        adjacency_u2o = torch.zeros_like(adjacency_u2u)
        bias_u2u = torch.zeros(
            batch, steps, node_count, node_count, device=device, dtype=dtype,
        )
        bias_u2o = torch.zeros_like(bias_u2u)
        adjacency_u2u[:, :, 0, 1 : 1 + self.MAX_U2U] = valid_u2u
        adjacency_u2u[:, :, 1 : 1 + self.MAX_U2U, 0] = valid_u2u
        bias_u2u[:, :, 0, 1 : 1 + self.MAX_U2U] = bias_value_u2u
        bias_u2u[:, :, 1 : 1 + self.MAX_U2U, 0] = bias_value_u2u
        first_obstacle = 1 + self.MAX_U2U
        adjacency_u2o[:, :, 0, first_obstacle:] = valid_u2o
        adjacency_u2o[:, :, first_obstacle:, 0] = valid_u2o
        bias_u2o[:, :, 0, first_obstacle:] = bias_value_u2o
        bias_u2o[:, :, first_obstacle:, 0] = bias_value_u2o

        flat = batch * steps
        layer1, _ = self.gat1(
            nodes.reshape(flat, node_count, -1),
            adjacency_u2u.reshape(flat, node_count, node_count),
            adjacency_u2o.reshape(flat, node_count, node_count),
            bias_u2u.reshape(flat, node_count, node_count),
            bias_u2o.reshape(flat, node_count, node_count),
        )
        layer2, attention = self.gat2(
            layer1,
            adjacency_u2u.reshape(flat, node_count, node_count),
            adjacency_u2o.reshape(flat, node_count, node_count),
            bias_u2u.reshape(flat, node_count, node_count),
            bias_u2o.reshape(flat, node_count, node_count),
        )
        active = valid_u2u.any(dim=2) | valid_u2o.any(dim=2)
        embedding = layer2[:, 0].reshape(batch, steps, self.output_dim)
        embedding = embedding * active.unsqueeze(-1)

        self.last_active_mask = active.detach()
        self.last_aux_info = {
            "u2u_active_ratio": valid_u2u.any(dim=2).float().mean().detach(),
            "u2o_active_ratio": valid_u2o.any(dim=2).float().mean().detach(),
            "active_ratio": active.float().mean().detach(),
            "mean_u2u_neighbours": valid_u2u.float().sum(dim=2).mean().detach(),
            "mean_u2o_neighbours": valid_u2o.float().sum(dim=2).mean().detach(),
            "u2u_risk_bias": bias_value_u2u.detach(),
            "u2o_risk_bias": bias_value_u2o.detach(),
            "u2o_closing_speed": (
                closing_speed.detach() if u2o is not None else bias_value_u2o
            ),
            "u2u_attention": attention["u2u_attention"].reshape(
                batch, steps, node_count, node_count,
            ).detach(),
            "u2o_attention": attention["u2o_attention"].reshape(
                batch, steps, node_count, node_count,
            ).detach(),
        }
        if return_aux:
            return embedding, self.last_aux_info
        return embedding


__all__ = [
    "EGO_FEATURES",
    "U2U_FEATURES",
    "U2O_FEATURES",
    "DynamicRiskInteractionEncoder",
]
