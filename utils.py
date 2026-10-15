from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pytorch_wavelets import DWT1DForward, DWT1DInverse


TRAJECTORY_FIELDS = ("lon", "lat", "alt", "spdx", "spdy", "spdz")


def safe_atan2(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    active = x.square() + y.square() > eps * eps
    return torch.atan2(
        torch.where(active, y, torch.zeros_like(y)),
        torch.where(active, x, torch.ones_like(x)),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Any, device: torch.device) -> Tuple[Any, ...]:
    def move(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    if not isinstance(batch, (tuple, list)) or len(batch) != 6:
        raise ValueError("batch must contain the six fields returned by DataGenerator")
    return tuple(move(value) for value in batch)


def unscale_trajectory(
    scaled: torch.Tensor, data_range: Dict[str, Dict[str, float]],
) -> torch.Tensor:
    columns = []
    for index, name in enumerate(TRAJECTORY_FIELDS):
        minimum = float(data_range[name]["min"])
        maximum = float(data_range[name]["max"])
        columns.append(scaled[..., index] * (maximum - minimum) + minimum)
    return torch.stack(columns, dim=-1)


def trajectory_to_local_metres(
    unscaled: torch.Tensor,
    reference: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if reference is None:
        reference = unscaled[:, 0, :3]
    lon0 = reference[:, 0:1]
    lat0 = reference[:, 1:2]
    alt0 = reference[:, 2:3]
    east_scale = 111320.0 * torch.cos(torch.deg2rad(lat0))
    east = (unscaled[..., 0] - lon0) * east_scale
    north = (unscaled[..., 1] - lat0) * 110540.0
    up = (unscaled[..., 2] - alt0) * 10.0
    velocity = unscaled[..., 3:6]
    return torch.cat([east[..., None], north[..., None], up[..., None], velocity], dim=-1), reference


def buildings_to_local_metres(
    building_lon: torch.Tensor,
    building_lat: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    lon0 = reference[:, 0:1, None]
    lat0 = reference[:, 1:2, None]
    east_scale = 111320.0 * torch.cos(torch.deg2rad(lat0))
    east = (building_lon - lon0) * east_scale
    north = (building_lat - lat0) * 110540.0
    return torch.stack([east, north], dim=-1)


def make_u2o_inputs(
    building_knn: Optional[Dict[str, torch.Tensor]],
    observed_enu: torch.Tensor,
    reference: torch.Tensor,
    history_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if building_knn is None:
        return None
    contour = building_knn["contour_xyz"][:, :history_length]
    building_xy = buildings_to_local_metres(
        contour[..., 0],
        contour[..., 1],
        reference,
    )
    return {
        "rel_xy_m": building_xy - observed_enu[:, :, None, :2],
        "height_m": contour[..., 2],
        "dist_m": building_knn["dist_m"][:, :history_length],
        "valid_mask": building_knn["valid_mask"][:, :history_length].bool(),
    }


def make_future_obstacles(
    building_knn: Optional[Dict[str, torch.Tensor]],
    reference: torch.Tensor,
    history_length: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if building_knn is None:
        return None
    contour = building_knn["contour_xyz"][:, history_length:]
    building_xy = buildings_to_local_metres(
        contour[..., 0],
        contour[..., 1],
        reference,
    )
    return (
        building_xy,
        contour[..., 2],
        building_knn["valid_mask"][:, history_length:].bool(),
    )


def slice_u2u(
    u2u: Optional[Dict[str, torch.Tensor]], history_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if u2u is None:
        return None
    return {
        key: value[:, :history_length] if torch.is_tensor(value) else value
        for key, value in u2u.items()
    }


def future_differences(
    observed: torch.Tensor, future: torch.Tensor,
) -> torch.Tensor:
    sequence = torch.cat([observed[:, -1:], future], dim=1)
    return sequence[:, 1:] - sequence[:, :-1]


def inverse_differences(
    observed: torch.Tensor, future_differences_scaled: torch.Tensor,
) -> torch.Tensor:
    return observed[:, -1:] + torch.cumsum(future_differences_scaled, dim=1)


class HaarWaveletCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dwt = DWT1DForward(J=2, wave="haar", mode="symmetric")
        self.idwt = DWT1DInverse(wave="haar", mode="symmetric")

    def encode(self, differences: torch.Tensor) -> torch.Tensor:
        low, highs = self.dwt(differences.transpose(1, 2))
        groups = [high.transpose(1, 2) for high in highs]
        groups.append(low.transpose(1, 2))
        return torch.cat(groups, dim=1)

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        high_1 = coefficients[:, :15].transpose(1, 2)
        high_2 = coefficients[:, 15:23].transpose(1, 2)
        low = coefficients[:, 23:31].transpose(1, 2)
        return self.idwt((low, [high_1, high_2])).transpose(1, 2)[:, :30]


__all__ = [
    "HaarWaveletCodec",
    "buildings_to_local_metres",
    "future_differences",
    "inverse_differences",
    "make_future_obstacles",
    "make_u2o_inputs",
    "move_batch",
    "seed_everything",
    "slice_u2u",
    "safe_atan2",
    "trajectory_to_local_metres",
    "unscale_trajectory",
]
