from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Point
from shapely.ops import nearest_points
from torch.utils.data import DataLoader

from building_obstacle_index import BuildingObstacleIndex
from dataloader import EnvironmentFeatures, LAYERS, TrajectoryWindowDataset
from train import ROOT, _prepare_batch, _resolve, build_model
from utils import HaarWaveletCodec, inverse_differences, unscale_trajectory


HORIZONS = (6, 12, 18, 30)
MAPE_MIN_ABS = np.array([1e-6, 1e-6, 5.0])
STATE_NAMES = ("lon", "lat", "alt", "spdx", "spdy", "spdz")
STATE_UNITS = ("deg", "deg", "m", "mps", "mps", "mps")


def ecef(position: np.ndarray) -> np.ndarray:
    lon, lat = np.deg2rad(position[..., 0]), np.deg2rad(position[..., 1])
    height = position[..., 2]
    eccentricity_squared = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)
    radius = 6378137.0 / np.sqrt(1.0 - eccentricity_squared * np.sin(lat) ** 2)
    return np.stack([
        (radius + height) * np.cos(lat) * np.cos(lon),
        (radius + height) * np.cos(lat) * np.sin(lon),
        (radius * (1.0 - eccentricity_squared) + height) * np.sin(lat),
    ], axis=-1)


def signed_prism_distance(point: Point, height_m: float, polygon, roof_m: float) -> float:
    horizontal = (
        -point.distance(polygon.boundary) if polygon.covers(point)
        else point.distance(polygon)
    )
    vertical = max(-height_m, height_m - roof_m)
    return float(
        np.hypot(max(horizontal, 0.0), max(vertical, 0.0))
        + min(max(horizontal, vertical), 0.0)
    )


class PolygonSafety:
    def __init__(self, index: BuildingObstacleIndex, safe_base_m: float = 35.0,
                 safe_speed_gain_s: float = 2.0, safe_min_m: float = 20.0,
                 safe_max_m: float = 120.0, vertical_margin_m: float = 20.0,
                 height_scale_m: float = 10.0):
        self.index = index
        self.safe_base_m = float(safe_base_m)
        self.safe_speed_gain_s = float(safe_speed_gain_s)
        self.safe_min_m = float(safe_min_m)
        self.safe_max_m = float(safe_max_m)
        self.vertical_margin_m = float(vertical_margin_m)
        self.height_scale_m = float(height_scale_m)
        if min(self.safe_base_m, self.safe_min_m, self.safe_max_m, self.height_scale_m) <= 0:
            raise ValueError("safety-distance parameters and height scale must be positive")
        if self.safe_min_m > self.safe_max_m:
            raise ValueError("safe_min_m cannot exceed safe_max_m")

    def _geometry(self, identifier: int):
        geometry = self.index._projected_geometry(identifier)
        if geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"building {identifier} has no complete polygon geometry")
        return geometry

    def _exposure(self, point: Point, altitude: float, velocity: np.ndarray,
                  identifiers: set[int]):
        delta, height_risk = 0.0, 0.0
        for identifier in identifiers:
            polygon = self._geometry(identifier)
            distance = point.distance(polygon)
            if distance > 0.0:
                anchor = nearest_points(point, polygon)[1]
                direction = np.asarray([anchor.x - point.x, anchor.y - point.y]) / distance
                closing_speed = max(float(np.dot(velocity, direction)), 0.0)
            else:
                closing_speed = 0.0
            separation = np.clip(
                self.safe_base_m + self.safe_speed_gain_s * closing_speed,
                self.safe_min_m, self.safe_max_m,
            )
            delta = max(delta, max(1.0 - distance / separation, 0.0))
            argument = (self.index.height[identifier] + self.vertical_margin_m - altitude) / self.height_scale_m
            height_risk = max(height_risk, float(1.0 / (1.0 + np.exp(-np.clip(argument, -700, 700)))))
        return delta, height_risk

    def pair(self, prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
        shape = prediction.shape[:2]
        result = {
            "delta_error": np.zeros(shape), "height_error": np.zeros(shape),
            "pred_clearance": np.full(shape, np.inf), "true_clearance": np.full(shape, np.inf),
            "active": np.zeros(shape, dtype=bool),
        }
        projected = []
        for states in (prediction, target):
            east, north = self.index.to_metres.transform(states[..., 0], states[..., 1])
            projected.append(np.stack([east, north], axis=-1))
        for sample in range(shape[0]):
            for step in range(shape[1]):
                points = [Point(xy[sample, step]) for xy in projected]
                identifiers = [
                    {item[1] for item in self.index._query_point(point.x, point.y, len(self.index.height))}
                    for point in points
                ]
                exposures = []
                for side, (point, states, own_ids) in enumerate(zip(points, (prediction, target), identifiers)):
                    altitude = float(states[sample, step, 2])
                    exposures.append(self._exposure(
                        point, altitude, states[sample, step, 3:5], own_ids,
                    ))
                result["delta_error"][sample, step] = abs(exposures[0][0] - exposures[1][0])
                result["height_error"][sample, step] = abs(exposures[0][1] - exposures[1][1])
                shared = identifiers[1]
                if not shared:
                    continue
                result["active"][sample, step] = True
                for side, (point, states) in enumerate(zip(points, (prediction, target))):
                    altitude = float(states[sample, step, 2])
                    clearance = min(
                        signed_prism_distance(point, altitude, self._geometry(identifier), self.index.height[identifier])
                        for identifier in shared
                    )
                    result["pred_clearance" if side == 0 else "true_clearance"][sample, step] = clearance
        return result


class MetricAccumulator:
    def __init__(self, safety: PolygonSafety, dt: float = 2.0):
        self.safety, self.dt = safety, float(dt)
        self.totals = {
            horizon: {
                "samples": 0, "points": 0, "absolute": np.zeros(6), "squared": np.zeros(6),
                "percentage": np.zeros(3), "percentage_count": np.zeros(3, dtype=np.int64),
                "displacement": 0.0, "final_displacement": 0.0, "delta": 0.0, "height": 0.0,
                "path_length": 0.0, "clearance": 0.0, "entry": 0.0, "safety_samples": 0,
            } for horizon in HORIZONS
        }

    def update(self, prediction: np.ndarray, target: np.ndarray,
               observed_last: np.ndarray | None = None) -> None:
        prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (30, 6):
            raise ValueError("prediction and target must have matching shape [B,30,6]")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("non-finite predictions or targets cannot be evaluated")
        for states in (prediction, target):
            if (np.abs(states[..., 0]) > 180).any() or (np.abs(states[..., 1]) > 90).any():
                raise ValueError("longitude or latitude is outside its geographic range")
        error = prediction - target
        displacement = np.linalg.norm(ecef(prediction[..., :3]) - ecef(target[..., :3]), axis=-1)
        target_ecef = ecef(target[..., :3])
        if observed_last is None:
            initial_ecef = target_ecef[:, :1]
        else:
            observed_last = np.asarray(observed_last, dtype=np.float64)
            if observed_last.shape != (len(target), 6):
                raise ValueError("observed_last must have shape [B,6]")
            initial_ecef = ecef(observed_last[:, None, :3])
        path_steps = np.linalg.norm(
            np.diff(np.concatenate([initial_ecef, target_ecef], axis=1), axis=1), axis=-1,
        )
        valid_mape = np.abs(target[..., :3]) >= MAPE_MIN_ABS
        relative_error = np.divide(
            np.abs(error[..., :3]), np.abs(target[..., :3]),
            out=np.zeros_like(error[..., :3]), where=valid_mape,
        )
        safety = self.safety.pair(prediction, target)
        for horizon, totals in self.totals.items():
            batch_size = len(prediction)
            totals["samples"] += batch_size
            totals["points"] += batch_size * horizon
            totals["absolute"] += np.abs(error[:, :horizon]).sum(axis=(0, 1))
            totals["squared"] += np.square(error[:, :horizon]).sum(axis=(0, 1))
            totals["percentage"] += relative_error[:, :horizon].sum(axis=(0, 1))
            totals["percentage_count"] += valid_mape[:, :horizon].sum(axis=(0, 1))
            totals["displacement"] += displacement[:, :horizon].sum()
            totals["final_displacement"] += displacement[:, horizon - 1].sum()
            totals["path_length"] += path_steps[:, :horizon].sum(axis=1).sum()
            totals["delta"] += safety["delta_error"][:, :horizon].sum()
            totals["height"] += safety["height_error"][:, :horizon].sum()
            active = safety["active"][:, :horizon].any(axis=1)
            totals["safety_samples"] += int(active.sum())
            if not active.any():
                continue
            clearances = [safety[key][active, :horizon] for key in ("pred_clearance", "true_clearance")]
            totals["clearance"] += np.abs(clearances[0].min(axis=1) - clearances[1].min(axis=1)).sum()
            entry_times = []
            for clearance in clearances:
                entered = clearance <= 0.0
                entry_times.append(np.where(entered.any(axis=1), entered.argmax(axis=1) + 1, horizon + 1) * self.dt)
            totals["entry"] += np.abs(entry_times[0] - entry_times[1]).sum()

    def rows(self) -> list[dict]:
        rows = []
        for horizon, totals in self.totals.items():
            if totals["samples"] == 0:
                raise ValueError("evaluation dataset contains no windows")
            row = {"horizon_steps": horizon, "horizon_seconds": horizon * self.dt,
                   "sample_count": totals["samples"], "safety_sample_count": totals["safety_samples"]}
            for column, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
                row[f"mae_{name}_{unit}"] = float(totals["absolute"][column] / totals["points"])
                row[f"rmse_{name}_{unit}"] = float(np.sqrt(totals["squared"][column] / totals["points"]))
            for column, name in enumerate(("lon", "lat", "alt")):
                count = int(totals["percentage_count"][column])
                ratio = float(totals["percentage"][column] / count) if count else None
                row[f"mape_{name}_percent"] = ratio * 100.0 if ratio is not None else None
                if name in ("lon", "lat"):
                    row[f"mape_{name}_permille"] = ratio * 1000.0 if ratio is not None else None
                row[f"mape_{name}_count"] = count
            mean_path = totals["path_length"] / totals["samples"]
            row.update({
                "mde_m": float(totals["displacement"] / totals["points"]),
                "fde_m": float(totals["final_displacement"] / totals["samples"]),
                "nmde": float((totals["displacement"] / totals["points"]) / mean_path) if mean_path > 0 else None,
                "ground_truth_path_length_m": float(mean_path),
                "delta_b_mae": float(totals["delta"] / totals["points"]),
                "h_r_mae": float(totals["height"] / totals["points"]),
                "msce_m": float(totals["clearance"] / totals["safety_samples"]) if totals["safety_samples"] else None,
                "fpete_s": float(totals["entry"] / totals["safety_samples"]) if totals["safety_samples"] else None,
            })
            rows.append(row)
        return rows


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluate(checkpoint_path: Path, split: str, output: Path, cpu: bool = False,
             batch_size: int | None = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["configuration"]
    data = config["data"]
    if (data["history_length"], data["prediction_length"], data["sample_dt_s"]) != (15, 30, 2.0):
        raise ValueError("formal evaluation requires history=15, prediction=30 and dt=2 s")
    data_range = np.load(_resolve(data["data_range_path"]), allow_pickle=True).item()
    model = build_model(config)
    normalization = model.dri_encoder if model.cfg.use_dri else model
    expected_minimum = normalization.trajectory_min.clone()
    expected_scale = normalization.trajectory_scale.clone()
    model.load_state_dict(checkpoint["model"], strict=True)
    if not torch.equal(expected_minimum, normalization.trajectory_min) or not torch.equal(expected_scale, normalization.trajectory_scale):
        raise ValueError("data_range differs from the checkpoint normalisation")
    if model.cfg.use_lop:
        model.lop_encoder.validate_normalization()
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model.to(device).eval()
    codec = HaarWaveletCodec().to(device)
    print("Loading complete building polygons and fixed evaluation windows...", flush=True)
    building_paths = [_resolve(path) for path in data["building_csv_paths"]]
    index = BuildingObstacleIndex([str(path) for path in building_paths])
    split_name = data[f"{split}_split"]
    split_path = _resolve(data["root"]) / split_name
    environment_features = EnvironmentFeatures(split_path.parent / "maps")
    dataset = TrajectoryWindowDataset(
        split_path, 45, data["eval_stride"], data_range, index,
        building_k=config["model"]["building_k"],
        u2u_radius_m=config["model"]["u2u_radius_m"], max_u2u=config["model"]["max_u2u"],
        environment_features=environment_features,
    )
    if not len(dataset):
        raise ValueError(f"no eligible windows in {split_path}")
    loader = DataLoader(dataset, batch_size=batch_size or data["eval_batch_size"],
                        shuffle=False, num_workers=0)
    loss_config = config["loss"]
    metrics = MetricAccumulator(PolygonSafety(
        index,
        vertical_margin_m=loss_config["dri_vertical_margin_m"],
    ))
    split_files = sorted(split_path.glob("*.txt"))
    files = [dict(file=path.name, sha256=sha256(path), points=len(record.time_seconds))
             for path, record in zip(split_files, dataset.records)]
    if len(files) != len(split_files):
        raise ValueError("evaluation split contains empty trajectory files")
    window_rows = [
        {"window_id": number, "file": files[record]["file"], "start": start, "stop": start + 45}
        for number, (record, start) in enumerate(dataset.windows)
    ]
    window_stream = io.StringIO(newline="")
    window_writer = csv.DictWriter(
        window_stream, fieldnames=("window_id", "file", "start", "stop"), lineterminator="\n",
    )
    window_writer.writeheader()
    window_writer.writerows(window_rows)
    window_bytes = window_stream.getvalue().encode("utf-8-sig")
    manifest = {
        "split": str(split_path), "window_length": 45, "stride": data["eval_stride"],
        "ordering": "sorted filenames; ascending starts from zero", "window_count": len(dataset),
        "window_index_sha256": hashlib.sha256(window_bytes).hexdigest(), "files": files,
        "data_range": {"path": str(_resolve(data["data_range_path"])), "sha256": sha256(_resolve(data["data_range_path"]))},
        "modules": {"use_lop": model.cfg.use_lop, "use_dri": model.cfg.use_dri},
        "lop_statistics": {"source": "checkpoint.model.lop_encoder", **{
            name: getattr(model.lop_encoder, name).detach().cpu().flatten().tolist()
            for name in ("kin_mean", "kin_std", "env_mean", "env_std")
        }} if model.cfg.use_lop else None,
        "time_basis": "TXT absolute timestamps in UTC+08:00; provenance and cross-UAV synchronisation verified before release, not inferred from formatting",
        "environment_sources": [
            {"file": name, "sha256": sha256(split_path.parent / "maps" / name)}
            for name in sorted({"weather_hourly.csv"} | {name for sources in LAYERS.values() for name, _ in sources})
        ],
        "data_validity": dataset.audit_summary,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    with torch.no_grad():
        for number, batch in enumerate(loader, 1):
            prepared = _prepare_batch(batch, device, data_range, 15)
            coefficients, _ = model(prepared["observed_scaled"], prepared["observed_enu"],
                                    prepared["environment"], u2u=prepared["u2u"], u2o=prepared["u2o"],
                                    environment_valid=prepared["environment_valid"])
            scaled = inverse_differences(prepared["observed_scaled"], codec.decode(coefficients))
            prediction = unscale_trajectory(scaled.double(), data_range).cpu().numpy()
            target = batch[1][:, 15:].double().numpy().copy()
            observed_last = batch[1][:, 14].double().numpy().copy()
            prediction[..., 2] *= 10.0
            target[..., 2] *= 10.0
            observed_last[..., 2] *= 10.0
            metrics.update(prediction, target, observed_last)
            if number == 1 or number % 20 == 0 or number == len(loader):
                print(f"evaluated {min(number * loader.batch_size, len(dataset))}/{len(dataset)} windows", flush=True)
    rows = metrics.rows()
    payload = {
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
        "epoch": checkpoint["epoch"], "configuration": config,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "building_sources": [{"path": str(path), "sha256": sha256(path)} for path in building_paths],
        "definitions": {
            "coordinates": "longitude/latitude in degrees; UAV and building heights in metres above local ground (AGL)",
            "mde_fde": "WGS84 ECEF distance with recorded AGL as the height argument; no terrain/ellipsoid datum conversion",
            "nmde": "MDE divided by the mean ground-truth future path length over the same horizon",
            "state_errors": "MAE/RMSE cover lon, lat, alt, spdx, spdy and spdz; MAPE covers lon, lat and alt",
            "mape_min_abs": MAPE_MIN_ABS.tolist(),
            "mape_units": "all three coordinates include 100*ratio (percent); longitude and latitude also retain 1000*ratio (permille)",
            "point_safety": "MAE of independently max-pooled exposure scores over all future points; empty neighbourhood score=0",
            "horizontal_separation": {
                "formula": "clip(35 m + 2 s * non-negative closing speed, 20 m, 120 m)",
                "safe_base_m": metrics.safety.safe_base_m,
                "safe_speed_gain_s": metrics.safety.safe_speed_gain_s,
                "safe_min_m": metrics.safety.safe_min_m,
                "safe_max_m": metrics.safety.safe_max_m,
            },
            "vertical_margin_m": metrics.safety.vertical_margin_m, "height_scale_m": metrics.safety.height_scale_m,
            "solid": "complete UTM EPSG:32648 building polygon extruded from 0 to mapped AGL height under a local-flat-ground approximation; no terrain correction or dilation",
            "point_safety_candidates": "all buildings within 200 m of each evaluated point; no six-building cap",
            "trajectory_safety_candidates": "all buildings within 200 m of the true point; the same complete solids are used for predicted and true trajectories",
            "msce": "MAE of per-trajectory minimum signed 3D distance over the horizon",
            "fpete": "MAE of first sampled solid entry time; boundary included, no-entry=(horizon+1)*2 s",
            "trajectory_safety_samples": "at least one valid building in the ground-truth search neighbourhood within the horizon",
            "empty_denominator": "null for MAPE/MSCE/FPETE when no eligible observations exist",
        },
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "window_manifest.json").write_bytes(manifest_bytes)
    (output / "window_index.csv").write_bytes(window_bytes)
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    payload = evaluate(args.checkpoint, args.split, args.output, args.cpu, args.batch_size)
    for row in payload["rows"]:
        print(json.dumps(row, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
