from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import DataGenerator, _read_record, fit_data_range
from DRI import DynamicRiskInteractionEncoder
from DROP_model import DROPConfig, DROPModel
from utils import (
    HaarWaveletCodec,
    future_differences,
    inverse_differences,
    make_future_obstacles,
    make_u2o_inputs,
    move_batch,
    seed_everything,
    slice_u2u,
    trajectory_to_local_metres,
    unscale_trajectory,
)


ROOT = Path(__file__).resolve().parent
STATE_NAMES = ("lon", "lat", "alt", "spdx", "spdy", "spdz")


class TrainingLog:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, message: str = "") -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}   {message}"
        print(line, flush=True)
        print(line, file=self.handle, flush=True)

    def close(self) -> None:
        self.handle.close()


def _format_value(value: Optional[float], precision: int = 6) -> str:
    return "NA" if value is None else f"{value:.{precision}g}"


def _log_evaluation(log: TrainingLog, rows: list[dict], epoch: int) -> None:
    log.write(f"Evaluation results of Epoch {epoch}")
    for row in rows:
        horizon = row["horizon_steps"]
        mae = ", ".join(
            f"{name}={_format_value(row[f'mae_{name}_{unit}'])}"
            for name, unit in zip(STATE_NAMES, ("deg", "deg", "m", "mps", "mps", "mps"))
        )
        rmse = ", ".join(
            f"{name}={_format_value(row[f'rmse_{name}_{unit}'])}"
            for name, unit in zip(STATE_NAMES, ("deg", "deg", "m", "mps", "mps", "mps"))
        )
        mape = ", ".join((
            f"lon={_format_value(row['mape_lon_percent'])}%",
            f"lat={_format_value(row['mape_lat_percent'])}%",
            f"alt={_format_value(row['mape_alt_percent'])}%",
        ))
        log.write(f"Horizon {horizon:02d}: MAE [{mae}] (lon/lat=deg, alt=m, velocity=m/s)")
        log.write(f"Horizon {horizon:02d}: MAPE [{mape}]")
        log.write(f"Horizon {horizon:02d}: RMSE [{rmse}] (lon/lat=deg, alt=m, velocity=m/s)")
        log.write(
            f"Horizon {horizon:02d}: MDE={row['mde_m']:.6f} m, "
            f"FDE={row['fde_m']:.6f} m, nMDE={_format_value(row['nmde'])}"
        )
        log.write(
            f"Horizon {horizon:02d}: Safety delta_b={row['delta_b_mae']:.6f}, "
            f"h_r={row['h_r_mae']:.6f}, MSCE={_format_value(row['msce_m'])} m, "
            f"FPETE={_format_value(row['fpete_s'])} s"
        )


def _write_tensorboard(writer: SummaryWriter, train_metrics: Dict[str, float],
                       dev_metrics: Dict[str, float], rows: list[dict], epoch: int) -> None:
    loss_names = {
        "loss_total": "Total", "wavelet_low": "Low_Freq",
        "loss_high": "High_Freq", "loss_l2": "L2",
        "loss_lop": "L_LOP", "loss_dri": "L_DRI",
    }
    for key, label in loss_names.items():
        writer.add_scalars(f"Loss/{label}", {
            "train": train_metrics[key], "validation": dev_metrics[key],
        }, epoch)
    for row in rows:
        horizon = row["horizon_steps"]
        prefix = f"Validation/H{horizon:02d}"
        for key, value in row.items():
            if isinstance(value, (int, float)) and value is not None:
                writer.add_scalar(f"{prefix}/{key}", value, epoch)
    writer.flush()


def split_wavelet_coefficients(
    coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if coefficients.ndim != 3 or coefficients.size(1) != 31:
        raise ValueError("wavelet coefficients must have shape [B,31,D]")
    return coefficients[:, :15], coefficients[:, 15:23], coefficients[:, 23:31]


def wavelet_coefficient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_detail: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_high_1, predicted_high_2, predicted_low = (
        split_wavelet_coefficients(prediction)
    )
    target_high_1, target_high_2, target_low = split_wavelet_coefficients(target)
    low = F.mse_loss(predicted_low, target_low)
    high_1 = F.mse_loss(predicted_high_1, target_high_1)
    high_2 = F.mse_loss(predicted_high_2, target_high_2)
    detail = torch.cat([predicted_high_1, predicted_high_2], dim=1).square().mean()
    total = low + high_1 + high_2 + float(lambda_detail) * detail
    return total, {
        "wavelet_low": low.detach(),
        "wavelet_high_1": high_1.detach(),
        "wavelet_high_2": high_2.detach(),
        "wavelet_detail": detail.detach(),
    }


def _horizon_mde(
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    horizons: Iterable[int],
) -> torch.Tensor:
    displacement = torch.linalg.vector_norm(
        prediction_m[..., :3] - target_m[..., :3], dim=-1,
    )
    return torch.stack(
        [displacement[:, : int(horizon)].mean(dim=1) for horizon in horizons],
        dim=1,
    )


def lop_supervision_loss(
    full_prediction_m: torch.Tensor,
    counterfactual_prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    preference_confidence: torch.Tensor,
    lop_valid_mask: torch.Tensor,
    lambda_benefit: float,
    lambda_confidence: float,
    tolerance_m: float,
    temperature_m: float,
    horizons: Tuple[int, ...] = (6, 12, 18, 30),
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    full_error = _horizon_mde(full_prediction_m, target_m, horizons)
    counterfactual_error = _horizon_mde(
        counterfactual_prediction_m, target_m, horizons,
    )
    sample_active = (lop_valid_mask.sum(dim=(1, 2)) > 0).to(full_error.dtype)
    denominator = sample_active.sum().clamp_min(epsilon)
    benefit_per_horizon = (
        F.relu(
            full_error
            - counterfactual_error.detach()
            - float(tolerance_m)
        )
        * sample_active[:, None]
    ).sum(dim=0) / denominator
    benefit = benefit_per_horizon.mean()
    valid_count = lop_valid_mask.sum(dim=(1, 2)).clamp_min(epsilon)
    mean_confidence = (
        preference_confidence * lop_valid_mask
    ).sum(dim=(1, 2)) / valid_count
    mean_full_error = full_error.mean(dim=1)
    mean_counterfactual_error = counterfactual_error.mean(dim=1)
    confidence_target = torch.sigmoid(
        (mean_counterfactual_error - mean_full_error)
        / max(float(temperature_m), epsilon)
    ).detach()
    confidence = F.binary_cross_entropy(
        mean_confidence.clamp(epsilon, 1.0 - epsilon),
        confidence_target,
        reduction="none",
    )
    confidence = (confidence * sample_active).sum() / denominator
    total = float(lambda_benefit) * benefit + float(lambda_confidence) * confidence
    return total, {
        "lop_benefit": benefit.detach(),
        "lop_confidence": confidence.detach(),
        "lop_active_ratio": sample_active.mean().detach(),
        "lop_mean_gate": (
            (mean_confidence * sample_active).sum() / denominator
        ).detach(),
    }


def dri_obstacle_proximity_loss(
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    obstacle_xy_m: torch.Tensor,
    obstacle_height_m: torch.Tensor,
    obstacle_valid: torch.Tensor,
    altitude_reference_m: torch.Tensor,
    horizontal_safe_m: float = 25.0,
    vertical_margin_m: float = 20.0,
    vertical_xy_gate_m: float = 50.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    valid = obstacle_valid.to(dtype=torch.bool)
    valid_time = valid.any(dim=-1)
    large = torch.finfo(prediction_m.dtype).max
    predicted_distance = torch.linalg.vector_norm(
        prediction_m[:, :, None, :2] - obstacle_xy_m, dim=-1,
    )
    target_distance = torch.linalg.vector_norm(
        target_m[:, :, None, :2] - obstacle_xy_m, dim=-1,
    )
    predicted_min = predicted_distance.masked_fill(~valid, large).min(dim=-1).values
    target_min = target_distance.masked_fill(~valid, large).min(dim=-1).values
    predicted_horizontal = F.relu(float(horizontal_safe_m) - predicted_min)
    target_horizontal = F.relu(float(horizontal_safe_m) - target_min)
    altitude_reference_m = altitude_reference_m.to(
        device=prediction_m.device, dtype=prediction_m.dtype,
    ).reshape(-1, 1, 1)
    predicted_altitude_m = prediction_m[:, :, None, 2] + altitude_reference_m
    target_altitude_m = target_m[:, :, None, 2] + altitude_reference_m
    predicted_vertical_candidates = F.relu(
        obstacle_height_m + float(vertical_margin_m)
        - predicted_altitude_m
    )
    target_vertical_candidates = F.relu(
        obstacle_height_m + float(vertical_margin_m)
        - target_altitude_m
    )
    predicted_vertical = torch.where(
        valid & (predicted_distance < float(vertical_xy_gate_m)),
        predicted_vertical_candidates,
        torch.zeros_like(predicted_vertical_candidates),
    ).max(dim=-1).values
    target_vertical = torch.where(
        valid & (target_distance < float(vertical_xy_gate_m)),
        target_vertical_candidates,
        torch.zeros_like(target_vertical_candidates),
    ).max(dim=-1).values
    horizontal_excess = F.relu(predicted_horizontal - target_horizontal)
    vertical_excess = F.relu(predicted_vertical - target_vertical)
    per_step = (
        horizontal_excess / float(horizontal_safe_m)
    ).square() + (
        vertical_excess / float(vertical_margin_m)
    ).square()
    denominator = valid_time.sum().clamp_min(1)
    total = (per_step * valid_time).sum() / denominator
    return total, {
        "dri_horizontal_excess": (
            horizontal_excess * valid_time
        ).sum().detach() / denominator,
        "dri_vertical_excess": (
            vertical_excess * valid_time
        ).sum().detach() / denominator,
        "dri_valid_step_ratio": valid_time.float().mean().detach(),
    }


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_configuration(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required_sections = {"data", "model", "optim", "loss", "output"}
    if set(config) != required_sections:
        raise ValueError(
            f"configuration sections must be exactly {sorted(required_sections)}"
        )
    model = config["model"]
    for name in ("use_lop", "use_dri"):
        if type(model.get(name, True)) is not bool:
            raise ValueError(f"model.{name} must be a boolean")
    for key in ("root", "train_split", "dev_split", "test_split", "data_range_path"):
        if not config["data"].get(key):
            raise ValueError(f"data.{key} is required")
    fixed = {
        "history_length": 15,
        "prediction_length": 30,
        "wavelet": "haar",
        "boundary_mode": "symmetric",
        "wavelet_levels": 2,
        "u2u_radius_m": 500.0,
        "u2o_radius_m": 200.0,
        "max_u2u": 8,
        "building_k": 6,
    }
    combined = {**config["data"], **model}
    for key, expected in fixed.items():
        if combined.get(key) != expected:
            raise ValueError(f"{key} must equal {expected!r}")
    if config["optim"].get("epochs") != 100:
        raise ValueError("epochs must equal 100")
    if config["optim"].get("min_lr") != 1e-5:
        raise ValueError("min_lr must equal 1e-5")
    return config


def prepare_normalization(config_path: Path) -> Dict[str, int]:
    data = load_configuration(config_path)["data"]
    root = _resolve(data["root"])
    split = root / data["train_split"]
    if split.resolve() in {(root / data[key]).resolve() for key in ("dev_split", "test_split")}:
        raise ValueError("training split must differ from validation and test splits")
    print(f"Fitting normalisation on training split only: {split}", flush=True)
    records = [_read_record(path) for path in sorted(split.glob("*.txt"))]
    ranges, summary = fit_data_range(records)
    range_path = _resolve(data["data_range_path"])
    range_path.parent.mkdir(parents=True, exist_ok=True)
    with range_path.open("wb") as handle:
        np.save(handle, ranges)
    print(json.dumps({**summary, "output": str(range_path)}, indent=2), flush=True)
    return summary


def build_model(config: Dict[str, Any], training_dataset=None) -> DROPModel:
    data = config["data"]
    model = config["model"]
    data_range = np.load(
        _resolve(data["data_range_path"]), allow_pickle=True,
    ).item()
    channels = ("lon", "lat", "alt", "spdx", "spdy", "spdz")
    trajectory_min = tuple(float(data_range[name]["min"]) for name in channels)
    trajectory_max = tuple(float(data_range[name]["max"]) for name in channels)
    cfg = DROPConfig(
        use_lop=model.get("use_lop", True),
        use_dri=model.get("use_dri", True),
        input_dim=model["input_dim"],
        output_dim=model["output_dim"],
        lop_dim=model["lop_dim"],
        dri_dim=model["dri_dim"],
        d_model=model["d_model"],
        n_head=model["n_head"],
        encoder_layers=model["encoder_layers"],
        decoder_layers=model["decoder_layers"],
        dropout=model["dropout"],
        wavelet=model["wavelet"],
        boundary_mode=model["boundary_mode"],
        wavelet_levels=model["wavelet_levels"],
        fusion_total_residual_cap=model["fusion_total_residual_cap"],
        sample_interval_s=data["sample_dt_s"],
        dri_hidden_dim=model["dri_hidden_dim"],
        trajectory_min=trajectory_min,
        trajectory_max=trajectory_max,
    )
    dri_encoder = DynamicRiskInteractionEncoder(
        hidden_dim=model["dri_hidden_dim"],
        output_dim=model["dri_dim"],
        dt=data["sample_dt_s"],
        dropout=model["dropout"],
        trajectory_min=trajectory_min,
        trajectory_max=trajectory_max,
    ) if cfg.use_dri else None
    result = DROPModel(cfg, dri_encoder=dri_encoder)
    if training_dataset is not None and cfg.use_lop:
        count = result.lop_encoder.fit_normalization(
            training_dataset, data["history_length"], data["sample_dt_s"],
        )
    return result


def build_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Dict[str, Dict[str, float]]]:
    data = config["data"]
    data_root = _resolve(data["root"])
    range_path = _resolve(data["data_range_path"])
    building_paths = [str(_resolve(path)) for path in data["building_csv_paths"]]
    generator = DataGenerator(
        str(data_root),
        data["history_length"] + data["prediction_length"],
        train=True,
        dev=True,
        test=False,
        train_path=data["train_split"],
        dev_path=data["dev_split"],
        data_range_path=str(range_path),
        gcn_building_k=config["model"]["building_k"],
        building_csv_paths=building_paths,
        train_stride=data["train_stride"],
        eval_stride=data["eval_stride"],
        u2u_radius_m=config["model"]["u2u_radius_m"],
        u2u_max_neighbors=config["model"]["max_u2u"],
    )
    workers = int(data["num_workers"])
    train_loader = DataLoader(
        generator.train_set,
        batch_size=data["batch_size"],
        shuffle=True,
        num_workers=workers,
        collate_fn=generator.collate,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        generator.dev_set,
        batch_size=data["eval_batch_size"],
        shuffle=False,
        num_workers=workers,
        collate_fn=generator.collate,
        pin_memory=torch.cuda.is_available(),
    )
    data_range = np.load(range_path, allow_pickle=True).item()
    return train_loader, dev_loader, data_range


def _prepare_batch(
    batch: Any,
    device: torch.device,
    data_range: Dict[str, Dict[str, float]],
    history_length: int,
) -> Dict[str, Any]:
    trajectory, _, environment, building_knn, validity, u2u = move_batch(batch, device)
    observed_scaled = trajectory[:, :history_length]
    future_scaled = trajectory[:, history_length:]
    unscaled = unscale_trajectory(trajectory, data_range)
    physical, reference = trajectory_to_local_metres(unscaled)
    observed_enu = physical[:, :history_length]
    future_m = physical[:, history_length:]
    u2o = make_u2o_inputs(
        building_knn, observed_enu, reference, history_length,
    )
    return {
        "observed_scaled": observed_scaled,
        "future_scaled": future_scaled,
        "environment": environment[:, :history_length],
        "environment_valid": validity["environment_valid"][:, :history_length],
        "observed_enu": observed_enu,
        "future_m": future_m,
        "reference": reference,
        "building_knn": building_knn,
        "u2o": u2o,
        "u2u": slice_u2u(u2u, history_length),
    }


def compute_objective(
    model: DROPModel,
    codec: HaarWaveletCodec,
    prepared: Dict[str, Any],
    data_range: Dict[str, Dict[str, float]],
    loss_config: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    coefficients, auxiliary = model(
        prepared["observed_scaled"],
        prepared["observed_enu"],
        prepared["environment"],
        u2u=prepared["u2u"],
        u2o=prepared["u2o"],
        return_lop_counterfactual=model.cfg.use_lop,
        environment_valid=prepared["environment_valid"],
    )
    target_difference = future_differences(
        prepared["observed_scaled"], prepared["future_scaled"],
    )
    target_coefficients = codec.encode(target_difference)
    wavelet, diagnostics = wavelet_coefficient_loss(
        coefficients,
        target_coefficients,
        loss_config["lambda_detail"],
    )

    predicted_scaled = inverse_differences(
        prepared["observed_scaled"], codec.decode(coefficients),
    )
    predicted_unscaled = unscale_trajectory(predicted_scaled, data_range)
    predicted_m, _ = trajectory_to_local_metres(
        predicted_unscaled, prepared["reference"],
    )
    lop = predicted_m.new_zeros(())
    lop_diagnostics = dict.fromkeys(
        ("lop_benefit", "lop_confidence", "lop_active_ratio", "lop_mean_gate"), lop,
    )
    if model.cfg.use_lop:
        counterfactual_scaled = inverse_differences(
            prepared["observed_scaled"],
            codec.decode(auxiliary["lop_counterfactual_coefficients"]),
        )
        counterfactual_unscaled = unscale_trajectory(counterfactual_scaled, data_range)
        counterfactual_m, _ = trajectory_to_local_metres(
            counterfactual_unscaled, prepared["reference"],
        )
        lop, lop_diagnostics = lop_supervision_loss(
            predicted_m,
            counterfactual_m,
            prepared["future_m"],
            auxiliary["preference_confidence"],
            auxiliary["lop_valid_mask"],
            lambda_benefit=loss_config["lambda_lop_benefit"],
            lambda_confidence=loss_config["lambda_lop_confidence"],
            tolerance_m=loss_config["lop_tolerance_m"],
            temperature_m=loss_config["lop_confidence_temperature_m"],
            epsilon=loss_config["epsilon"],
        )

    future_obstacles = make_future_obstacles(
        prepared["building_knn"], prepared["reference"],
        prepared["observed_scaled"].size(1),
    ) if model.cfg.use_dri else None
    if future_obstacles is None:
        dri = predicted_m.sum() * 0.0
        dri_diagnostics = {
            "dri_horizontal_excess": dri.detach(),
            "dri_vertical_excess": dri.detach(),
            "dri_valid_step_ratio": dri.detach(),
        }
    else:
        dri, dri_diagnostics = dri_obstacle_proximity_loss(
            predicted_m,
            prepared["future_m"],
            *future_obstacles,
            altitude_reference_m=prepared["reference"][:, 2] * 10.0,
            horizontal_safe_m=loss_config["dri_horizontal_safe_m"],
            vertical_margin_m=loss_config["dri_vertical_margin_m"],
            vertical_xy_gate_m=loss_config["dri_vertical_xy_gate_m"],
        )
    total = wavelet + lop + dri
    displacement = torch.linalg.vector_norm(
        predicted_m[..., :3] - prepared["future_m"][..., :3], dim=-1,
    ).mean()
    diagnostics.update(lop_diagnostics)
    diagnostics.update(dri_diagnostics)
    diagnostics.update(
        {
            "loss_total": total.detach(),
            "loss_wavelet": wavelet.detach(),
            "loss_high": (diagnostics["wavelet_high_1"] + diagnostics["wavelet_high_2"]).detach(),
            "loss_l2": (float(loss_config["lambda_detail"]) * diagnostics["wavelet_detail"]).detach(),
            "loss_lop": lop.detach(),
            "loss_dri": dri.detach(),
            "mde_m": displacement.detach(),
        }
    )
    return total, diagnostics, predicted_unscaled.detach()


def _validation_accumulator(dataset, loss_config: Dict[str, float]):
    from evaluate import MetricAccumulator, PolygonSafety
    return MetricAccumulator(
        PolygonSafety(
            dataset.building_index,
            vertical_margin_m=loss_config["dri_vertical_margin_m"],
        )
    )


def run_epoch(
    model: DROPModel,
    loader: DataLoader,
    codec: HaarWaveletCodec,
    device: torch.device,
    data_range: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer],
    metric_accumulator=None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    batch_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            prepared = _prepare_batch(
                batch,
                device,
                data_range,
                config["data"]["history_length"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, diagnostics, prediction = compute_objective(
                model, codec, prepared, data_range, config["loss"],
            )
            if metric_accumulator is not None:
                predicted = prediction.cpu().numpy().copy()
                target = batch[1][:, config["data"]["history_length"]:].detach().cpu().double().numpy().copy()
                observed_last = batch[1][:, config["data"]["history_length"] - 1].detach().cpu().double().numpy().copy()
                predicted[..., 2] *= 10.0
                target[..., 2] *= 10.0
                observed_last[..., 2] *= 10.0
                metric_accumulator.update(predicted, target, observed_last)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["optim"]["max_grad_norm"],
                )
                optimizer.step()
            for key, value in diagnostics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batch_count += 1
    return {key: value / max(batch_count, 1) for key, value in totals.items()}


def train(config_path: Path) -> None:
    config = load_configuration(config_path)
    seed_everything(config["optim"]["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config["optim"]["cpu"] else "cpu"
    )
    output_dir = _resolve(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").unlink(missing_ok=True)
    log = TrainingLog(output_dir / "train.log")
    writer = SummaryWriter(log_dir=str(output_dir / "Tensorboard"))
    try:
        train_loader, dev_loader, data_range = build_loaders(config)
        model = build_model(config, train_loader.dataset).to(device)
        codec = HaarWaveletCodec().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["optim"]["lr"],
            weight_decay=config["optim"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["optim"]["epochs"],
            eta_min=config["optim"]["min_lr"],
        )
        obstacle_index = train_loader.dataset.building_index
        log.write(
            f"ObstacleIndex: N={len(obstacle_index.height)}, "
            f"k={obstacle_index.k_neighbors}, max_dist={obstacle_index.max_distance_m:.1f} m"
        )
        log.write("--------------------TRAIN DETAILS--------------------")
        log.write(f"device: {device}")
        log.write(f"parameter_count: {sum(parameter.numel() for parameter in model.parameters())}")
        for line in json.dumps(config, ensure_ascii=False, indent=2).splitlines():
            log.write(line)
        best = math.inf
        epochs = config["optim"]["epochs"]
        for epoch in range(epochs):
            log.write(f"--------------------Epoch {epoch}/{epochs - 1}--------------------")
            train_metrics = run_epoch(
                model, train_loader, codec, device, data_range, config, optimizer,
            )
            validation = _validation_accumulator(dev_loader.dataset, config["loss"])
            dev_metrics = run_epoch(
                model, dev_loader, codec, device, data_range, config, None,
                metric_accumulator=validation,
            )
            rows = validation.rows()
            scheduler.step()
            for stage, values in (("Train", train_metrics), ("Validation", dev_metrics)):
                log.write(
                    f"{stage} Loss: Total={values['loss_total']:.7f}, "
                    f"Low-Freq={values['wavelet_low']:.7f}, "
                    f"High-Freq={values['loss_high']:.7f}, L2={values['loss_l2']:.7e}, "
                    f"L_lop={values['loss_lop']:.7f}, L_dri={values['loss_dri']:.7f}"
                )
            log.write(f"learning_rate: {optimizer.param_groups[0]['lr']:.8f}")
            _log_evaluation(log, rows, epoch)
            _write_tensorboard(writer, train_metrics, dev_metrics, rows, epoch)
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "configuration": config,
                "validation_metrics": rows,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if dev_metrics["loss_total"] < best:
                best = dev_metrics["loss_total"]
                torch.save(checkpoint, output_dir / "best.pt")
    finally:
        writer.close()
        log.close()


def smoke_test(config_path: Path) -> None:
    config = load_configuration(config_path)
    model = build_model(config)
    codec = HaarWaveletCodec()
    batch = 2
    observed = torch.rand(batch, 15, 6)
    future = torch.rand(batch, 30, 6)
    observed_enu = torch.randn(batch, 15, 6)
    environment = torch.randn(batch, 15, 15)
    environment[..., 14] = 1.0
    u2o = {
        "rel_xy_m": torch.randn(batch, 15, 6, 2) * 40.0,
        "height_m": torch.rand(batch, 15, 6) * 80.0,
        "dist_m": torch.rand(batch, 15, 6) * 200.0,
        "valid_mask": torch.ones(batch, 15, 6, dtype=torch.bool),
    }
    coefficients, auxiliary = model(
        observed,
        observed_enu,
        environment,
        u2o=u2o,
        return_lop_counterfactual=True,
    )
    target_coefficients = codec.encode(future_differences(observed, future))
    wavelet, _ = wavelet_coefficient_loss(
        coefficients, target_coefficients, config["loss"]["lambda_detail"],
    )
    full_m = codec.decode(coefficients)
    target_m = codec.decode(target_coefficients)
    lop = dri = full_m.new_zeros(())
    if model.cfg.use_lop:
        counterfactual_m = codec.decode(auxiliary["lop_counterfactual_coefficients"])
        lop, _ = lop_supervision_loss(
            full_m,
            counterfactual_m,
            target_m,
            auxiliary["preference_confidence"],
            auxiliary["lop_valid_mask"],
            config["loss"]["lambda_lop_benefit"],
            config["loss"]["lambda_lop_confidence"],
            config["loss"]["lop_tolerance_m"],
            config["loss"]["lop_confidence_temperature_m"],
        )
    if model.cfg.use_dri:
        obstacle_xy = torch.randn(batch, 30, 6, 2) * 40.0
        obstacle_height = torch.rand(batch, 30, 6) * 80.0
        obstacle_valid = torch.ones(batch, 30, 6, dtype=torch.bool)
        dri, _ = dri_obstacle_proximity_loss(
            full_m,
            target_m,
            obstacle_xy,
            obstacle_height,
            obstacle_valid,
            altitude_reference_m=torch.zeros(batch),
        )
    total = wavelet + lop + dri
    total.backward()
    if coefficients.shape != (batch, 31, 6):
        raise AssertionError("unexpected coefficient shape")
    if model.cfg.use_lop and not auxiliary["preference_confidence"].requires_grad:
        raise AssertionError("preference confidence is not differentiable")
    if model.cfg.use_lop and model.lop_encoder.environment_gate.weight.grad is None:
        raise AssertionError("LOP confidence received no gradient")
    print("smoke test passed")


def smoke_data(config_path: Path) -> None:
    config = load_configuration(config_path)
    config["data"].update(batch_size=2, eval_batch_size=2, num_workers=0)
    seed_everything(config["optim"]["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    train_loader, dev_loader, data_range = build_loaders(config)
    model = build_model(config, train_loader.dataset)
    codec = HaarWaveletCodec()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optim"]["lr"])
    for name, loader in (("train", train_loader), ("dev", dev_loader)):
        dataset = loader.dataset
        def context_score(index):
            record_index, start = dataset.windows[index]
            record = dataset.records[record_index]
            reliable = record.environment[start:start + 15, 14].sum()
            edges = dataset._u2u(record_index, start)["valid_mask"][:15].sum().item()
            return edges, reliable
        selected = sorted(range(len(dataset)), key=context_score, reverse=True)[:2]
        batch = loader.collate_fn([dataset[index] for index in selected])
        print(json.dumps({"smoke_context": name, "windows": selected,
                          "observed_u2u_edges": int(batch[5]["valid_mask"][:, :15].sum())}), flush=True)
        metrics = run_epoch(
            model, [batch], codec, torch.device("cpu"), data_range, config,
            optimizer if name == "train" else None,
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise AssertionError(f"non-finite {name} diagnostics")
        print(json.dumps({"smoke_split": name, **metrics}), flush=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise AssertionError("non-finite parameters after diagnostic step")
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("non-finite gradients after diagnostic step")
    print("data smoke test passed; no checkpoint saved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "optDir" / "opt_DROP.json",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--smoke-data", action="store_true")
    mode.add_argument(
        "--prepare-normalization", action="store_true",
        help="Overwrite data_range using training TXT only, then exit; retrain weights after changing the input scale.",
    )
    arguments = parser.parse_args()
    if arguments.prepare_normalization:
        prepare_normalization(arguments.config)
    elif arguments.smoke_test:
        smoke_test(arguments.config)
    elif arguments.smoke_data:
        smoke_data(arguments.config)
    else:
        train(arguments.config)


if __name__ == "__main__":
    main()
