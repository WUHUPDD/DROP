from __future__ import annotations

import ast
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from pyproj import Transformer
from shapely import STRtree, is_valid, make_valid, transform, wkt
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import unary_union

from building_obstacle_index import BuildingObstacleIndex


csv.field_size_limit(2**31 - 1)


LAYERS = {
    "water": (("水系（见山）.csv", "geometry"),),
    "green": (("绿地（见山）.csv", "geometry"),),
    "road": (("道路+铁路（见山）.csv", "geometry"),),
    "building": (("障碍物（见山）.csv", "geometry"),
                 ("建筑物补充.csv", "geometry"),
                 ("障碍物补充后数据.csv", "geometry_expand")),
}
WEATHER_FIELDS = ("weather_code", "wind_speed_10m", "wind_direction_10m",
                  "wind_speed_100m", "wind_direction_100m", "temperature_2m")


def _geometry(text):
    if not text.strip():
        raise ValueError("empty geographic geometry")
    if text[0].isalpha():
        return wkt.loads(text)
    if text[0] == "{":
        return shape(json.loads(text))
    coords = ast.literal_eval(text)
    depth, value = 0, coords
    while isinstance(value, (list, tuple)) and value:
        depth += 1
        value = value[0]
    if depth == 2:
        return Polygon(coords)
    if depth == 3:
        polygons = [Polygon(ring) for ring in coords]
    elif depth == 4:
        polygons = [Polygon(rings[0], rings[1:]) for rings in coords]
    else:
        raise ValueError("unsupported geographic geometry")
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


class EnvironmentFeatures:
    def __init__(self, map_directory: Path):
        self.root = Path(map_directory)
        self.project = Transformer.from_crs(4326, 32648, always_xy=True).transform
        self.layers = {}
        self.geographic_cache = {}
        self.weather = {}
        with (self.root / "weather_hourly.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (round(float(row["longitude"]) / 0.25),
                       round(float(row["latitude"]) / 0.25), row["time"][:13])
                self.weather[key] = np.asarray([float(row[k]) for k in WEATHER_FIELDS], dtype=np.float32)

    def _load_layers(self):
        for name, sources in LAYERS.items():
            geometries = []
            for filename, column in sources:
                with (self.root / filename).open(encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        geom = _geometry(row[column].strip())
                        if not geom.is_empty:
                            geometries.append(geom)
            geometries = transform(geometries, self.project, interleaved=False)
            invalid = ~is_valid(geometries)
            geometries[invalid] = make_valid(geometries[invalid])
            self.layers[name] = STRtree(geometries)

    def land_cover(self, longitude, latitude, radii=(100,)):
        if not self.layers:
            self._load_layers()
        values = []
        for lon, lat in zip(longitude, latitude):
            key = (round(float(lon) * 1000), round(float(lat) * 1000), tuple(radii))
            if key not in self.geographic_cache:
                point = Point(*self.project(key[0] / 1000, key[1] / 1000))
                search = point.buffer(max(radii))
                unions = []
                for tree in self.layers.values():
                    hits = tree.query(search, predicate="intersects")
                    unions.append(unary_union(tree.geometries[hits]))
                proportions = []
                for radius in radii:
                    disk = point.buffer(radius)
                    areas = np.clip([geom.intersection(disk).area / (np.pi * radius**2)
                                     for geom in unions], 0, 1)
                    total = areas.sum()
                    if total > 1:
                        areas /= total
                    water, green, road, building = areas
                    proportions.append([max(0., 1. - total), road, green, water, building])
                self.geographic_cache[key] = np.round(proportions, 4).astype(np.float32)
            values.append(self.geographic_cache[key])
        return np.asarray(values)

    def lookup(self, states_m, times):
        raw = []
        for state, time in zip(states_m, times):
            key = (round(float(state[0]) / 0.25), round(float(state[1]) / 0.25),
                   time.strftime("%Y-%m-%dT%H"))
            if key not in self.weather:
                raise ValueError(f"missing hourly weather for grid/time {key} in {self.root}")
            raw.append(self.weather[key])
        code, v10, d10, v100, d100, temperature = np.asarray(raw, dtype=np.float32).T
        z = np.clip(states_m[:, 2].astype(float), 10., 1000.)
        alpha = np.full(len(z), 0.2)
        valid = (v10 > 0.1) & (v100 > 0.1)
        alpha[valid] = np.log(v100[valid] / v10[valid]) / np.log(10.)
        wind = v10 * (z / 10.) ** np.clip(alpha, 0.05, 0.4)
        angle = np.radians((d10 + ((d100 - d10 + 180.) % 360. - 180.)
                            * np.log(z / 10.) / np.log(10.)) % 360.)
        temperature = temperature - 0.0065 * (z - 2.)
        categories = np.zeros((len(z), 5), dtype=np.float32)
        for i, value in enumerate(code):
            if not np.isfinite(value):
                continue
            value = int(value)
            category = (0 if value == 0 else 2 if value in {45, 48} else
                        3 if value in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82} else
                        4 if value in {71, 73, 75, 77, 85, 86} else 1)
            categories[i, category] = 1.
        geography = self.land_cover(states_m[:, 0], states_m[:, 1])[:, 0]
        features = np.column_stack([categories, wind, np.sin(angle), np.cos(angle), temperature, geography])
        valid = np.isfinite(features).all(axis=1) & categories.any(axis=1)
        return np.where(np.isfinite(features), features, 0.).astype(np.float32), valid[:, None]


TRAJECTORY_FIELDS = ("lon", "lat", "alt", "spdx", "spdy", "spdz")
LOCAL_TIMEZONE = timezone(timedelta(hours=8))


@dataclass
class FlightRecord:
    identifier: str
    date: str
    time_seconds: np.ndarray
    trajectory_unscaled: np.ndarray
    environment: np.ndarray
    physical_id: str
    timestamp_us: np.ndarray
    environment_valid: np.ndarray


def _read_record(
    path: Path,
    environment_features: Optional[EnvironmentFeatures] = None,
) -> FlightRecord:
    """Read TXT timestamps assumed synchronised and verified before release (UTC+08:00)."""
    states, times = [], []
    identifier = ""
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("|")
            if len(fields) != 9:
                raise ValueError(f"expected exactly nine pipe-separated fields: {path}:{line_number}")
            try:
                time = datetime.strptime(fields[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TIMEZONE)
                values = np.asarray(fields[2:], dtype=np.float64)
            except ValueError as error:
                raise ValueError(f"invalid timestamp or numeric field: {path}:{line_number}") from error
            identity = fields[1]
            if not re.fullmatch(r"[A-Za-z0-9]+", identity) or (identifier and identity != identifier):
                raise ValueError(f"invalid or mixed UAV IDs: {path}:{line_number}")
            if (not np.isfinite(values).all()
                    or not (-180 <= values[0] <= 180 and -90 <= values[1] <= 90)
                    or not np.isclose(values[6], np.linalg.norm(values[3:6]), rtol=1e-5, atol=1e-5)):
                raise ValueError(f"invalid state or speed magnitude: {path}:{line_number}")
            if times and (time - times[-1]).total_seconds() != 2.0:
                raise ValueError(f"expected complete timestamps at 2 s intervals: {path}:{line_number}")
            identifier = identity
            states.append(values[:6])
            times.append(time)
    if not states:
        raise ValueError(f"empty trajectory: {path}")
    states = np.asarray(states, dtype=np.float64)
    environment = np.zeros((len(states), 15), dtype=np.float32)
    environment_valid = np.zeros((len(states), 1), dtype=np.bool_)
    if environment_features is not None:
        environment[:, :14], environment_valid = environment_features.lookup(states, times)
    environment[:, 14] = 1.
    states[:, 2] /= 10.
    return FlightRecord(
        identifier=identifier,
        date=times[0].strftime("%Y-%m-%d"),
        time_seconds=np.asarray([t.hour * 3600 + t.minute * 60 + t.second for t in times]),
        trajectory_unscaled=states,
        environment=environment,
        physical_id=identifier,
        timestamp_us=np.asarray([round(t.timestamp() * 1e6) for t in times], dtype=np.int64),
        environment_valid=environment_valid,
    )


def fit_data_range(records):
    if not records:
        raise ValueError("training split must contain trajectories")
    states = np.concatenate([record.trajectory_unscaled for record in records])
    if not np.isfinite(states).all():
        raise ValueError("training states must be finite")
    data_range = {}
    for index, name in enumerate(TRAJECTORY_FIELDS):
        values = states[:, index]
        if values.max() <= values.min():
            raise ValueError(f"training attribute {name} has zero range")
        data_range[name] = dict(min=float(values.min()), max=float(values.max()),
                               mu=float(values.mean()), sigma=float(values.std(ddof=0)))
    return data_range, {
        "training_files": len(records), "training_points": len(states),
        "height_divisor": 10, "standard_deviation_ddof": 0,
    }


class TrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        split_path: Path,
        window_length: int,
        stride: int,
        data_range: Dict[str, Dict[str, float]],
        building_index: BuildingObstacleIndex,
        building_k: int = 6,
        u2u_radius_m: float = 500.0,
        max_u2u: int = 8,
        environment_features: Optional[EnvironmentFeatures] = None,
    ) -> None:
        if not split_path.is_dir():
            raise FileNotFoundError(split_path)
        self.window_length = int(window_length)
        self.data_range = data_range
        self.building_index = building_index
        self.building_k = int(building_k)
        self.u2u_radius_m = float(u2u_radius_m)
        self.max_u2u = int(max_u2u)
        self.records = [
            _read_record(path, environment_features)
            for path in sorted(split_path.glob("*.txt"))
        ]
        self.windows: List[Tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            last_start = len(record.time_seconds) - self.window_length
            self.windows.extend(
                (record_index, start)
                for start in range(0, last_start + 1, max(1, int(stride)))
            )
        self.physical_id_codes = {
            identity: code for code, identity in enumerate(sorted({
                record.physical_id for record in self.records if record.physical_id
            }))
        }
        self.time_index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for record_index, record in enumerate(self.records):
            for row_index, timestamp in enumerate(record.timestamp_us):
                self.time_index[int(timestamp)].append((record_index, row_index))
        self.audit_summary = {
            "split": split_path.name,
            "records": len(self.records),
            "windows": len(self.windows),
            "points": sum(len(record.time_seconds) for record in self.records),
            "timestamp_points": sum(len(record.timestamp_us) for record in self.records),
            "valid_environment_points": sum(int(record.environment_valid.sum()) for record in self.records),
        }

    def __len__(self) -> int:
        return len(self.windows)

    def _scale(self, trajectory: np.ndarray) -> np.ndarray:
        scaled = trajectory.copy()
        for index, name in enumerate(TRAJECTORY_FIELDS):
            minimum = float(self.data_range[name]["min"])
            maximum = float(self.data_range[name]["max"])
            scaled[:, index] = (scaled[:, index] - minimum) / (maximum - minimum)
        return scaled.astype(np.float32)

    def _u2u(self, record_index: int, start: int) -> Dict[str, torch.Tensor]:
        record = self.records[record_index]
        relative_position = np.zeros(
            (self.window_length, self.max_u2u, 3), dtype=np.float32,
        )
        neighbour_velocity = np.zeros_like(relative_position)
        valid = np.zeros((self.window_length, self.max_u2u), dtype=np.bool_)
        identifiers = np.full(
            (self.window_length, self.max_u2u), -1, dtype=np.int64,
        )
        for offset in range(self.window_length):
            row = start + offset
            ego = record.trajectory_unscaled[row]
            candidates = []
            for other_record_index, other_row in self.time_index.get(
                int(record.timestamp_us[row]), []
            ):
                if other_record_index == record_index:
                    continue
                other_record = self.records[other_record_index]
                if other_record.physical_id == record.physical_id:
                    continue
                other = other_record.trajectory_unscaled[other_row]
                east = (
                    (other[0] - ego[0])
                    * 111320.0
                    * np.cos(np.deg2rad(ego[1]))
                )
                north = (other[1] - ego[1]) * 110540.0
                up = (other[2] - ego[2]) * 10.0
                distance = float(np.sqrt(east * east + north * north + up * up))
                if distance <= self.u2u_radius_m:
                    candidates.append(
                        (
                            distance,
                            np.asarray([east, north, up], dtype=np.float32),
                            other[3:6].astype(np.float32),
                            self.physical_id_codes[other_record.physical_id],
                        )
                    )
            candidates.sort(key=lambda item: item[0])
            unique_candidates = {}
            for candidate in candidates:
                unique_candidates.setdefault(candidate[-1], candidate)
            for neighbour, (_, rel, velocity, identifier) in enumerate(
                list(unique_candidates.values())[: self.max_u2u]
            ):
                relative_position[offset, neighbour] = rel
                neighbour_velocity[offset, neighbour] = velocity
                valid[offset, neighbour] = True
                identifiers[offset, neighbour] = identifier
        return {
            "rel_pos_m": torch.from_numpy(relative_position),
            "velocity_mps": torch.from_numpy(neighbour_velocity),
            "valid_mask": torch.from_numpy(valid),
            "id_codes": torch.from_numpy(identifiers),
        }

    def __getitem__(self, index: int):
        record_index, start = self.windows[index]
        record = self.records[record_index]
        stop = start + self.window_length
        unscaled = record.trajectory_unscaled[start:stop]
        trajectory = torch.from_numpy(self._scale(unscaled))
        environment = torch.from_numpy(record.environment[start:stop])
        longitude = unscaled[:, 0][None]
        latitude = unscaled[:, 1][None]
        building = self.building_index.query_batch(
            longitude, latitude, k_neighbors=self.building_k,
        )
        building_tensors = {
            key: torch.from_numpy(value[0]) for key, value in building.items()
        }
        return (
            trajectory,
            torch.from_numpy(unscaled),
            environment,
            building_tensors,
            {"environment_valid": torch.from_numpy(record.environment_valid[start:stop])},
            self._u2u(record_index, start),
        )


class DataGenerator:
    def __init__(
        self,
        data_path: str,
        minibatch_len: int,
        train: bool = True,
        dev: bool = True,
        test: bool = False,
        train_path: str = "train",
        dev_path: str = "dev",
        test_path: str = "test",
        data_range_path: Optional[str] = None,
        train_stride: int = 1,
        eval_stride: int = 1,
        gcn_building_k: int = 6,
        building_csv_paths: Optional[Sequence[str]] = None,
        building_entity_merge_radius_m: float = 2.0,
        u2u_radius_m: float = 500.0,
        u2u_max_neighbors: int = 8,
    ) -> None:
        root = Path(data_path)
        range_path = Path(data_range_path or root / "data_range.npy")
        self.data_range = np.load(range_path, allow_pickle=True).item()
        building_index = BuildingObstacleIndex(
            csv_paths=building_csv_paths,
            k_neighbors=gcn_building_k,
            entity_merge_radius_m=building_entity_merge_radius_m,
        )
        environment_features = EnvironmentFeatures(root / "maps")

        def dataset(split: str, stride: int) -> TrajectoryWindowDataset:
            result = TrajectoryWindowDataset(
                root / split,
                minibatch_len,
                stride,
                self.data_range,
                building_index,
                building_k=gcn_building_k,
                u2u_radius_m=u2u_radius_m,
                max_u2u=u2u_max_neighbors,
                environment_features=environment_features,
            )
            if not len(result):
                raise ValueError(f"no eligible {minibatch_len}-point windows in {root / split}")
            return result

        if train:
            self.train_set = dataset(train_path, train_stride)
        if dev:
            self.dev_set = dataset(dev_path, eval_stride)
        if test:
            self.test_set = dataset(test_path, eval_stride)

    @staticmethod
    def collate(batch):
        return default_collate(batch)


__all__ = ["DataGenerator", "TrajectoryWindowDataset", "fit_data_range"]
