from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import STRtree, make_valid, wkt
from shapely.geometry import MultiPolygon, Point, Polygon, box, shape
from shapely.ops import nearest_points, transform


DEFAULT_BUILDING_CSV_FILES = tuple(
    Path(__file__).resolve().parent / "example_data" / "maps" / name
    for name in ("建筑物补充.csv", "障碍物（见山）.csv", "障碍物补充后数据.csv")
)
COORDINATE_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[, ]\s*"
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


@dataclass(frozen=True)
class _Source:
    path: Path
    columns: dict[str, int]


def _finite_float(value: object, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _coordinates(text: object) -> np.ndarray:
    pairs = COORDINATE_PATTERN.findall(str(text or ""))
    return np.asarray(pairs, dtype=np.float64).reshape(-1, 2)


def _nested_depth(value: object) -> int:
    if not isinstance(value, (list, tuple)):
        return 0
    if not value:
        return 1
    return 1 + _nested_depth(value[0])


def _parse_geometry(text: object, longitude: float, latitude: float):
    raw = str(text or "").strip()
    geometry = None
    if raw:
        try:
            if raw[0] in "[{":
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value = ast.literal_eval(raw)
                if isinstance(value, dict):
                    geometry = shape(value)
                elif _nested_depth(value) == 3:
                    geometry = Polygon(value[0], value[1:])
                elif _nested_depth(value) == 4:
                    geometry = MultiPolygon(
                        [(polygon[0], polygon[1:]) for polygon in value]
                    )
            else:
                geometry = wkt.loads(raw)
        except (SyntaxError, TypeError, ValueError):
            geometry = None
    if geometry is None or geometry.is_empty:
        geometry = Point(longitude, latitude)
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    return geometry


def _scan_sources(paths: Sequence[str], transformer: Transformer):
    sources = []
    longitude = []
    latitude = []
    height = []
    bounds = []
    source_index = []
    offsets = []
    for path_value in paths:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            header = next(csv.reader([handle.readline().decode("utf-8-sig")]))
            columns = {
                str(name).strip().lower(): index
                for index, name in enumerate(header)
            }
            current_source = len(sources)
            sources.append(_Source(path, columns))
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                row = next(csv.reader([raw_line.decode("utf-8", errors="replace")]))

                def get(name: str) -> str:
                    column = columns.get(name)
                    return row[column] if column is not None and column < len(row) else ""

                obstacle_height = _finite_float(get("height"), 0.0)
                if obstacle_height <= 0.0:
                    continue
                geometry_text = get("geometry") or get("geometry_expand")
                coordinates = _coordinates(geometry_text)
                if coordinates.size:
                    lon = float(coordinates[:, 0].mean())
                    lat = float(coordinates[:, 1].mean())
                    projected_x, projected_y = transformer.transform(
                        coordinates[:, 0], coordinates[:, 1],
                    )
                else:
                    lon = _finite_float(get("longitude") or get("lon") or get("lng"))
                    lat = _finite_float(get("latitude") or get("lat"))
                    projected_x, projected_y = transformer.transform([lon], [lat])
                if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                    continue
                longitude.append(lon)
                latitude.append(lat)
                height.append(obstacle_height)
                bounds.append((
                    float(np.min(projected_x)), float(np.min(projected_y)),
                    float(np.max(projected_x)), float(np.max(projected_y)),
                ))
                source_index.append(current_source)
                offsets.append(offset)
    if not longitude:
        raise RuntimeError("no valid building entities were loaded")
    return (
        sources,
        np.asarray(longitude, dtype=np.float64),
        np.asarray(latitude, dtype=np.float64),
        np.asarray(height, dtype=np.float64),
        np.asarray(bounds, dtype=np.float64),
        np.asarray(source_index, dtype=np.int32),
        np.asarray(offsets, dtype=np.int64),
    )


def _representatives(
    longitude: np.ndarray,
    latitude: np.ndarray,
    height: np.ndarray,
    radius_m: float,
    transformer: Transformer,
) -> np.ndarray:
    if radius_m <= 0.0 or longitude.size < 2:
        return np.arange(longitude.size)
    x, y = transformer.transform(longitude, latitude)
    parent = np.arange(longitude.size)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in cKDTree(np.column_stack([x, y])).query_pairs(radius_m):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    groups = {}
    for index in range(longitude.size):
        groups.setdefault(find(index), []).append(index)
    return np.asarray(
        [
            max(group, key=lambda index: (height[index], -index))
            for group in groups.values()
        ],
        dtype=np.int64,
    )


class BuildingObstacleIndex:
    def __init__(
        self,
        csv_paths: Optional[Sequence[str]] = None,
        k_neighbors: int = 6,
        max_distance_m: float = 200.0,
        entity_merge_radius_m: float = 2.0,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.max_distance_m = float(max_distance_m)
        if self.k_neighbors < 1 or self.max_distance_m <= 0.0:
            raise ValueError("k_neighbors and max_distance_m must be positive")
        self.to_metres = Transformer.from_crs(
            "EPSG:4326", "EPSG:32648", always_xy=True,
        )
        self.to_wgs84 = Transformer.from_crs(
            "EPSG:32648", "EPSG:4326", always_xy=True,
        )
        (
            self.sources,
            longitude,
            latitude,
            height,
            bounds,
            source_index,
            offsets,
        ) = _scan_sources(tuple(csv_paths or DEFAULT_BUILDING_CSV_FILES), self.to_metres)
        selected = _representatives(
            longitude,
            latitude,
            height,
            float(entity_merge_radius_m),
            self.to_metres,
        )
        self.longitude = longitude[selected]
        self.latitude = latitude[selected]
        self.height = height[selected]
        selected_bounds = bounds[selected]
        self.source_index = source_index[selected]
        self.offsets = offsets[selected]
        boxes = [box(*extent) for extent in selected_bounds]
        self.tree = STRtree(boxes)

    def _row(self, index: int) -> tuple[list[str], _Source]:
        source = self.sources[int(self.source_index[index])]
        with source.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            row = next(
                csv.reader([handle.readline().decode("utf-8", errors="replace")])
            )
        return row, source

    @lru_cache(maxsize=32768)
    def _projected_geometry(self, index: int):
        row, source = self._row(index)

        def get(name: str) -> str:
            column = source.columns.get(name)
            return row[column] if column is not None and column < len(row) else ""

        raw = get("geometry") or get("geometry_expand")
        geometry = _parse_geometry(
            raw, self.longitude[index], self.latitude[index],
        )
        return transform(self.to_metres.transform, geometry)

    def _query_point(self, x: float, y: float, count: int):
        point = Point(x, y)
        search = box(
            x - self.max_distance_m,
            y - self.max_distance_m,
            x + self.max_distance_m,
            y + self.max_distance_m,
        )
        matches = []
        for index in self.tree.query(search):
            index = int(index)
            geometry = self._projected_geometry(index)
            distance = point.distance(geometry)
            if distance > self.max_distance_m + 1e-7:
                continue
            outline = geometry.boundary
            contour = nearest_points(
                point, outline if not outline.is_empty else geometry,
            )[1]
            matches.append((distance, index, contour.x, contour.y))
        matches.sort(key=lambda item: (item[0], item[1]))
        return matches[:count]

    def query_batch(
        self,
        longitude: np.ndarray,
        latitude: np.ndarray,
        k_neighbors: Optional[int] = None,
    ):
        longitude = np.asarray(longitude, dtype=np.float64)
        latitude = np.asarray(latitude, dtype=np.float64)
        if longitude.shape != latitude.shape:
            raise ValueError("longitude and latitude must have matching shapes")
        count = self.k_neighbors if k_neighbors is None else int(k_neighbors)
        if count < 1:
            raise ValueError("k_neighbors must be positive")
        original_shape = longitude.shape
        lon_flat, lat_flat = longitude.reshape(-1), latitude.reshape(-1)
        finite = np.isfinite(lon_flat) & np.isfinite(lat_flat)
        x = np.full_like(lon_flat, np.nan)
        y = np.full_like(lat_flat, np.nan)
        x[finite], y[finite] = self.to_metres.transform(
            lon_flat[finite].tolist(), lat_flat[finite].tolist(),
        )
        contour_lon = np.repeat(lon_flat[:, None], count, axis=1)
        contour_lat = np.repeat(lat_flat[:, None], count, axis=1)
        obstacle_height = np.zeros((len(lon_flat), count), dtype=np.float64)
        distance = np.full((len(lon_flat), count), np.inf, dtype=np.float64)
        indices = np.full((len(lon_flat), count), -1, dtype=np.int64)
        for row in np.flatnonzero(finite):
            matches = self._query_point(float(x[row]), float(y[row]), count)
            if not matches:
                continue
            size = len(matches)
            distance[row, :size] = [item[0] for item in matches]
            indices[row, :size] = [item[1] for item in matches]
            outline_x = [item[2] for item in matches]
            outline_y = [item[3] for item in matches]
            contour_lon[row, :size], contour_lat[row, :size] = (
                self.to_wgs84.transform(outline_x, outline_y)
            )
            obstacle_height[row, :size] = self.height[indices[row, :size]]
        valid = indices >= 0
        result_shape = original_shape + (count,)
        contour_xyz = np.stack(
            [contour_lon, contour_lat, obstacle_height], axis=-1,
        ).reshape(result_shape + (3,)).astype(np.float32)
        return {
            "contour_xyz": contour_xyz,
            "dist_m": distance.reshape(result_shape).astype(np.float32),
            "valid_mask": valid.reshape(result_shape),
            "indices": indices.reshape(result_shape),
        }

__all__ = ["BuildingObstacleIndex", "DEFAULT_BUILDING_CSV_FILES"]
