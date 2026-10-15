from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pyproj import Transformer
from shapely.geometry import Point
from torch.utils.data import default_collate

from building_obstacle_index import BuildingObstacleIndex
from dataloader import EnvironmentFeatures, LAYERS, WEATHER_FIELDS, TrajectoryWindowDataset, _read_record, fit_data_range
from DRI import U2U_FEATURES, _previous_by_id
from DROP_model import DROPConfig, EmbeddingFusion
from evaluate import HORIZONS, MetricAccumulator, PolygonSafety, evaluate, signed_prism_distance
from LOP import LOPEncoder
from train import (
    _prepare_batch,
    build_model,
    build_loaders,
    compute_objective,
    dri_obstacle_proximity_loss,
    load_configuration,
    main as training_main,
    prepare_normalization,
)
from utils import HaarWaveletCodec


ROOT = Path(__file__).resolve().parent
LON, LAT = 104.0, 30.6
TO_METRES = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
TO_WGS84 = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)


def write_buildings(path, offsets):
    origin_x, origin_y = TO_METRES.transform(LON, LAT)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lon", "lat", "height"])
        for east, north in offsets:
            lon, lat = TO_WGS84.transform(origin_x + east, origin_y + north)
            writer.writerow([lon, lat, 30])
    return BuildingObstacleIndex([str(path)], entity_merge_radius_m=0)


def polygon_index(path: Path) -> BuildingObstacleIndex:
    centre_x, centre_y = TO_METRES.transform(LON, LAT)
    corners = [
        (centre_x - 10, centre_y - 10), (centre_x + 10, centre_y - 10),
        (centre_x + 10, centre_y + 10), (centre_x - 10, centre_y + 10),
        (centre_x - 10, centre_y - 10),
    ]
    coordinates = [TO_WGS84.transform(x, y) for x, y in corners]
    geometry = "POLYGON ((" + ", ".join(f"{lon} {lat}" for lon, lat in coordinates) + "))"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["geometry", "height"])
        writer.writerow([geometry, 30])
    return BuildingObstacleIndex([str(path)], entity_merge_radius_m=0)


def write_record(path, identity, times, east=0):
    rows = [f"{time}|{identity}|{LON + east / 95700}|{LAT}|25|1|0.5|0.1|{np.sqrt(1.26)}"
            for time in times]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_environment(root):
    maps = root / "maps"
    maps.mkdir(exist_ok=True)
    polygon = "POLYGON ((104 30.6,104.0001 30.6,104.0001 30.6001,104 30.6001,104 30.6))"
    for sources in LAYERS.values():
        for name, column in sources:
            with (maps / name).open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([column])
                writer.writerow([polygon])
    with (maps / "weather_hourly.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["longitude", "latitude", "time", *WEATHER_FIELDS])
        for hour in range(24):
            writer.writerow([104, 30.5, f"2026-04-26T{hour:02d}:00", 0, 10, 350, 20, 10, 22])
    return EnvironmentFeatures(maps)


def full_configuration():
    config = load_configuration(ROOT / "optDir/opt_DROP.json")
    config["model"].update(use_lop=True, use_dri=True)
    return config


class FusionTests(unittest.TestCase):
    def test_mlp_transformer_encoder_and_two_level_haar_decoder(self):
        config = full_configuration()
        config["model"].update(d_model=16, encoder_layers=1, decoder_layers=1, dropout=0)
        model = build_model(config)
        self.assertFalse(hasattr(model.trajectory_encoder, "dwt"))
        self.assertEqual(model.trajectory_encoder.input_mlp[0].in_features, 6)
        self.assertEqual(model.trajectory_encoder.input_mlp[0].out_features, 8)
        self.assertEqual(model.trajectory_encoder.input_mlp[2].out_features, 16)
        self.assertEqual(len(model.trajectory_encoder.encoder.layers), 1)
        self.assertTrue(model.trajectory_encoder.encoder.layers[0].norm_first)
        self.assertFalse(model.trajectory_encoder.encoder.enable_nested_tensor)
        self.assertEqual(model.trajectory_encoder.encoder.layers[0].self_attn.num_heads, 4)
        self.assertEqual(model.decoder.layers[0].dwt.J, 2)
        observed = torch.rand(2, 15, 6, requires_grad=True)
        encoded = model.trajectory_encoder(observed)
        self.assertEqual(encoded.shape, (2, 15, 16))
        encoded.square().mean().backward()
        self.assertTrue(torch.isfinite(observed.grad).all())

    def test_cap_zero_tiny_and_gradcheck(self):
        fusion = EmbeddingFusion(DROPConfig(d_model=8, lop_dim=4, dri_dim=4)).double()
        for magnitude in (0.0, 1e-12, 1e-8, 0.1, 100.0):
            residual = (torch.randn(1, 2, 3, dtype=torch.double) * magnitude).requires_grad_()
            reference = torch.randn_like(residual).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(lambda r, h: fusion._cap_residual(r, h)[0], (residual, reference)))
        zero = torch.zeros(1, 2, 3, dtype=torch.double, requires_grad=True)
        fusion._cap_residual(zero, zero)[0].sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_cap_nonzero_formula_unchanged(self):
        fusion = EmbeddingFusion(DROPConfig())
        r, h = torch.randn(3, 15, 192), torch.randn(3, 15, 192)
        ratio = r.norm(dim=(1, 2), keepdim=True) / (0.1 * h.norm(dim=(1, 2), keepdim=True) + fusion.eps)
        actual, _ = fusion._cap_residual(r, h)
        torch.testing.assert_close(actual, r * ratio.tanh() / ratio)

    def test_masked_and_mixed_fusion_gradients(self):
        for mixed in (False, True):
            fusion = EmbeddingFusion(DROPConfig(d_model=8, lop_dim=4, dri_dim=4))
            h = torch.randn(2, 15, 8, requires_grad=True)
            p = torch.randn(2, 15, 4, requires_grad=True)
            r = torch.randn(2, 15, 4, requires_grad=True)
            confidence = torch.rand(2, 15, 1, requires_grad=True)
            mask = torch.zeros(2, 15, 1)
            if mixed:
                mask[1, 5:] = 1
            memory, _, _, _ = fusion(h, p, confidence, r, mask, mask)
            torch.testing.assert_close(memory[0], fusion.output_norm(h)[0])
            (memory * torch.randn_like(memory)).sum().backward()
            for tensor in [h, p, r, confidence, *fusion.parameters()]:
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_full_model_inactive_backward(self):
        config = full_configuration()
        config["model"].update(d_model=16, encoder_layers=1, decoder_layers=1, dropout=0)
        model = build_model(config)
        observed = torch.rand(2, 15, 6, requires_grad=True)
        environment = torch.rand(2, 15, 15)
        environment[..., 14] = 0
        prediction, aux = model(observed, torch.randn(2, 15, 10), environment, return_lop_counterfactual=True)
        self.assertFalse(aux["lop_valid_mask"].any())
        self.assertFalse(aux["dri_active_mask"].any())
        torch.testing.assert_close(prediction, aux["lop_counterfactual_coefficients"])
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(observed.grad).all())
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_hovering_input_gradients_are_finite(self):
        config = full_configuration()
        config["model"].update(d_model=16, encoder_layers=1, decoder_layers=1, dropout=0)
        model = build_model(config)
        minimum = model.dri_encoder.trajectory_min
        scale = model.dri_encoder.trajectory_scale
        physical = (minimum + 0.5 * scale).expand(2, 15, 6).clone()
        physical[..., 3:] = 0.0
        observed = ((physical - minimum) / scale).requires_grad_()
        kinematics = torch.zeros(2, 15, 6, requires_grad=True)
        environment = torch.rand(2, 15, 15)
        environment[..., 14] = 1.0
        coefficients, _ = model(observed, kinematics, environment)
        coefficients.square().mean().backward()
        for tensor in (observed, kinematics):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class BuildingTests(unittest.TestCase):
    def test_projected_polygon_bounds_cover_all_true_extrema(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "polygon.csv"
            polygon = "POLYGON ((104 30.6, 104.002 30.6, 104.002 30.61, 104 30.61, 104 30.6))"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["geometry", "height"])
                writer.writerow([polygon, 30])
            index = BuildingObstacleIndex([str(path)], entity_merge_radius_m=0)
            geometry = index._projected_geometry(0)
            np.testing.assert_allclose(index.tree.geometries[0].bounds, geometry.bounds, atol=1e-7)
            vertices = np.asarray(geometry.exterior.coords)
            for axis in (0, 1):
                for side in (-1, 1):
                    row = vertices[:, axis].argmin() if side < 0 else vertices[:, axis].argmax()
                    query = vertices[row].copy()
                    query[axis] += 199.5 * side
                    lon, lat = index.to_wgs84.transform(*query)
                    result = index.query_batch(np.array([[lon]]), np.array([[lat]]))
                    self.assertTrue(result["valid_mask"].any())
                    self.assertAlmostEqual(float(result["dist_m"][0, 0, 0]), 199.5, places=3)

    def test_east_190_boundary_and_top_six(self):
        with tempfile.TemporaryDirectory() as temp:
            index = write_buildings(Path(temp) / "b.csv", [(190, 0), (201, 0), (0, 180), (0, 199), (-199, 0), (120, 160), (141, 141), (0, 210)])
            result = index.query_batch(np.array([[LON]]), np.array([[LAT]]))
            self.assertEqual(result["indices"][0, 0, :2].tolist(), [2, 0])
            self.assertEqual(set(result["indices"][0, 0, 2:4]), {3, 4})
            self.assertEqual(result["indices"][0, 0, 4:].tolist(), [6, 5])
            self.assertTrue(result["valid_mask"].all())
            np.testing.assert_allclose(result["dist_m"][0, 0], [180, 190, 199, 199, 141 * np.sqrt(2), 200], atol=2e-3)
            empty = index.query_batch(np.array([[LON + 1]]), np.array([[LAT]]))
            self.assertFalse(empty["valid_mask"].any())
            self.assertTrue(np.isinf(empty["dist_m"]).all())

    def test_all_directions_and_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                index = write_buildings(Path(temp) / "b.csv", [(199 * np.cos(angle), 199 * np.sin(angle)), (201 * np.cos(angle), 201 * np.sin(angle))])
                result = index.query_batch(np.array([[LON]]), np.array([[LAT]]))
                self.assertEqual(result["valid_mask"].shape, (1, 1, 6))
                self.assertEqual(result["valid_mask"].sum(), 1)
                self.assertEqual(result["indices"][0, 0, 0], 0)

    def test_against_brute_force(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as temp:
            index = write_buildings(Path(temp) / "b.csv", rng.uniform(-400, 400, (300, 2)))
            longitude = LON + rng.uniform(-0.002, 0.002, (3, 8))
            latitude = LAT + rng.uniform(-0.002, 0.002, (3, 8))
            result = index.query_batch(longitude, latitude)
            for row in np.ndindex(longitude.shape):
                building_x, building_y = TO_METRES.transform(index.longitude, index.latitude)
                query_x, query_y = TO_METRES.transform(longitude[row], latitude[row])
                distance = np.hypot(building_x - query_x, building_y - query_y)
                valid = np.flatnonzero(distance <= 200 + 1e-7)
                expected = valid[np.lexsort((valid, distance[valid]))][:6]
                np.testing.assert_array_equal(result["indices"][row][result["valid_mask"][row]], expected)

    def test_polygon_contour_not_centroid(self):
        with tempfile.TemporaryDirectory() as temp:
            origin_x, origin_y = TO_METRES.transform(LON, LAT)
            corners_xy = [
                (origin_x + 90, origin_y - 10),
                (origin_x + 110, origin_y - 10),
                (origin_x + 110, origin_y + 10),
                (origin_x + 90, origin_y + 10),
                (origin_x + 90, origin_y - 10),
            ]
            corners = [TO_WGS84.transform(x, y) for x, y in corners_xy]
            polygon = "POLYGON ((" + ", ".join(
                f"{lon} {lat}" for lon, lat in corners
            ) + "))"
            path = Path(temp) / "polygon.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["height", "geometry"])
                writer.writerow([35, polygon])
            index = BuildingObstacleIndex([str(path)], entity_merge_radius_m=0)
            result = index.query_batch(np.array([[LON]]), np.array([[LAT]]))
            self.assertTrue(result["valid_mask"][0, 0, 0])
            self.assertAlmostEqual(float(result["dist_m"][0, 0, 0]), 90.0, places=3)
            contour = result["contour_xyz"][0, 0, 0]
            contour_x, contour_y = TO_METRES.transform(contour[0], contour[1])
            self.assertAlmostEqual(contour_x - origin_x, 90.0, delta=0.5)
            self.assertAlmostEqual(contour_y - origin_y, 0.0, delta=0.5)
            self.assertEqual(float(contour[2]), 35.0)

    def test_dri_loss_uses_shared_contour_anchor(self):
        prediction = torch.tensor(
            [[[15.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], requires_grad=True,
        )
        target = torch.zeros(1, 1, 6)
        contour_anchor = torch.tensor([[[[20.0, 0.0]]]])
        obstacle_height = torch.zeros(1, 1, 1)
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        loss, _ = dri_obstacle_proximity_loss(
            prediction,
            target,
            contour_anchor,
            obstacle_height,
            valid,
            altitude_reference_m=torch.tensor([100.0]),
        )
        self.assertAlmostEqual(float(loss.detach()), 0.36, places=6)
        loss.backward()
        self.assertAlmostEqual(float(prediction.grad[0, 0, 0]), 0.048, places=6)


class DataTests(unittest.TestCase):
    def test_nine_column_state_and_absolute_time(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "UAV1.txt"
            lines = [f"2026-04-26 12:00:{s:02d}|UAV1|104.0|30.6|100|3|4|0|5" for s in (0, 2)]
            path.write_text("\n".join(lines)+"\n", encoding="utf-8")
            record = _read_record(path)
            self.assertEqual(record.physical_id, "UAV1")
            self.assertEqual(record.date, "2026-04-26")
            np.testing.assert_array_equal(record.trajectory_unscaled[:, 2], [10, 10])
            np.testing.assert_array_equal(record.trajectory_unscaled[:, 3:], [[3,4,0],[3,4,0]])
            self.assertTrue((record.timestamp_us >= 0).all())
            self.assertEqual(record.timestamp_us[1]-record.timestamp_us[0], 2_000_000)
            self.assertTrue((record.environment[:, 14] == 1).all())
            self.assertFalse(record.environment_valid.any())

    def test_nine_column_invalid_rows_are_not_silently_skipped(self):
        row = "2026-04-26 12:00:00|UAV1|104.0|30.6|100|3|4|0|5"
        invalid = [row.replace("|", "\t"), row+"|extra", row.replace("2026-04-26 ", ""), row.rsplit("|",1)[0], row[:-1]+"6", row.replace("104.0","nan"),
                   row+"\n"+row, row+"\n"+row.replace("12:00:00","12:00:03"),
                   row+"\n"+row.replace("12:00:00","12:00:02").replace("UAV1","UAV2")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "UAV1.txt"
            for text in invalid:
                path.write_text(text+"\n",encoding="utf-8")
                with self.assertRaises(ValueError):
                    _read_record(path)

    def test_txt_alone_enables_u2u_matching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "train"
            split.mkdir()
            for identity in ("UAV1", "UAV2"):
                write_record(split / f"{identity}.txt", identity,
                             [f"2026-04-26 12:00:{s:02d}" for s in (0, 2)],
                             east=0 if identity == "UAV1" else 10)
            index = write_buildings(root / "building.csv",[(100,0)])
            ranges = np.load(ROOT / "data_range.npy",allow_pickle=True).item()
            dataset = TrajectoryWindowDataset(split,2,1,ranges,index)
            graph = dataset._u2u(0,0)
            self.assertEqual(graph["valid_mask"].sum().item(),2)
            self.assertTrue((graph["id_codes"][:,0]==dataset.physical_id_codes["UAV2"]).all())
            self.assertFalse(dataset.records[0].environment_valid.any())

    def test_u2u_horizontal_distance_is_an_independent_input(self):
        config = full_configuration()
        encoder = build_model(config).dri_encoder.eval()
        captured = {}
        hook = encoder.u2u_projection.register_forward_pre_hook(
            lambda _, args: captured.update(features=args[0].detach())
        )
        observed = torch.zeros(1, 15, 6)
        u2u = {
            "rel_pos_m": torch.tensor([3.,4.,12.]).expand(1,15,1,3),
            "velocity_mps": torch.zeros(1,15,1,3),
            "valid_mask": torch.ones(1,15,1,dtype=torch.bool),
            "id_codes": torch.ones(1,15,1,dtype=torch.long),
        }
        risk = encoder(observed, u2u=u2u)
        hook.remove()
        self.assertEqual(encoder.u2u_projection.in_features, 19)
        horizontal = U2U_FEATURES.index("horizontal_distance")
        distance = U2U_FEATURES.index("distance")
        torch.testing.assert_close(captured["features"][0,:,0,horizontal], torch.full((15,),5./500))
        torch.testing.assert_close(captured["features"][0,:,0,distance], torch.full((15,),13./500))
        self.assertFalse(captured["features"][:,:,1:].any())
        risk.square().mean().backward()
        gradient = encoder.u2u_projection.weight.grad[:,horizontal]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_default_splits_and_compact_training_loader(self):
        config = full_configuration()
        self.assertEqual(config["data"]["dev_split"], "dev")
        self.assertTrue(all(not Path(p).is_absolute() for p in config["data"]["building_csv_paths"]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path = root / "building.csv"
            write_buildings(index_path, [(100,0)])
            for split in ("train", "dev"):
                folder = root / split
                folder.mkdir()
                lines = [f"2026-04-26 12:{s//60:02d}:{s%60:02d}|UAV1|104|30.6|100|3|4|0|5"
                         for s in range(0,90,2)]
                (folder / "UAV1.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
            config["data"].update(root=str(root), building_csv_paths=[str(index_path)],
                                 batch_size=2,eval_batch_size=2)
            write_environment(root)
            train_loader, dev_loader, ranges = build_loaders(config)
            self.assertEqual(len(train_loader.dataset),1)
            self.assertEqual(len(dev_loader.dataset),1)
            batch = next(iter(train_loader))
            prepared = _prepare_batch(batch,torch.device("cpu"),ranges,15)
            self.assertEqual(prepared["observed_scaled"].shape,(1,15,6))
            self.assertEqual(prepared["future_scaled"].shape,(1,30,6))

    def test_u2u_derivatives_follow_physical_identity_after_rank_swaps(self):
        values = torch.tensor([[[[10.0], [20.0]], [[21.0], [11.0]], [[12.0], [22.0]]]])
        identifiers = torch.tensor([[[1, 2], [2, 1], [1, 2]]])
        valid = torch.ones_like(identifiers, dtype=torch.bool)
        previous, matched = _previous_by_id(values, valid, identifiers)
        torch.testing.assert_close(previous[0, 1:, :, 0], torch.tensor([[20.0, 10.0], [11.0, 21.0]]))
        self.assertTrue(matched[0, 1:].all())

    def test_u2o_closing_speed_direction(self):
        config = full_configuration()
        model = build_model(config).eval()
        environment = torch.ones(1, 15, 15)
        u2o = {
            "rel_xy_m": torch.zeros(1, 15, 6, 2),
            "height_m": torch.zeros(1, 15, 6),
            "dist_m": torch.full((1, 15, 6), 40.0),
            "valid_mask": torch.ones(1, 15, 6, dtype=torch.bool),
        }
        u2o["rel_xy_m"][..., 0] = 40.0

        def risk(velocity: float):
            physical = torch.tensor(
                [104.0, 30.6, 10.0, velocity, 0.0, 0.0]
            ).expand(1, 15, 6)
            observed = (
                physical - model.dri_encoder.trajectory_min
            ) / model.dri_encoder.trajectory_scale
            with torch.no_grad():
                _, auxiliary = model(
                    observed,
                    torch.randn(1, 15, 10),
                    environment,
                    u2o=u2o,
                )
            return auxiliary["dri"]

        approaching = risk(10.0)
        departing = risk(-10.0)
        self.assertTrue((approaching["u2o_closing_speed"] == 10.0).all())
        self.assertTrue((departing["u2o_closing_speed"] == 0.0).all())
        self.assertGreater(
            float(approaching["u2o_risk_bias"].mean()),
            float(departing["u2o_risk_bias"].mean()),
        )

    def test_u2o_selection_uses_utm_distance_without_enu_refiltering(self):
        config = full_configuration()
        encoder = build_model(config).dri_encoder
        u2o = {
            "rel_xy_m": torch.zeros(1, 1, 6, 2),
            "height_m": torch.ones(1, 1, 6),
            "dist_m": torch.full((1, 1, 6), float("inf")),
            "valid_mask": torch.zeros(1, 1, 6, dtype=torch.bool),
        }
        u2o["rel_xy_m"][0, 0, 0, 0] = 250.0
        u2o["dist_m"][0, 0, 0] = 190.0
        u2o["valid_mask"][0, 0, 0] = True
        relative, _, valid = encoder._select_u2o(u2o, torch.zeros(1, 1, 3))
        self.assertTrue(valid[0, 0, 0])
        self.assertEqual(float(relative[0, 0, 0, 0]), 250.0)

    def test_dri_vertical_loss_restores_absolute_altitude(self):
        prediction = torch.zeros(1, 1, 6)
        target = torch.zeros(1, 1, 6)
        target[..., 2] = 20.0
        obstacle_xy = torch.zeros(1, 1, 1, 2)
        obstacle_height = torch.full((1, 1, 1), 50.0)
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        loss, diagnostics = dri_obstacle_proximity_loss(
            prediction,
            target,
            obstacle_xy,
            obstacle_height,
            valid,
            altitude_reference_m=torch.tensor([100.0]),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(diagnostics["dri_vertical_excess"]), 0.0)

    def test_fixed_training_schedule(self):
        config = full_configuration()
        self.assertEqual(config["optim"]["epochs"], 100)
        self.assertNotIn("early_stopping_patience", config["optim"])
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=config["optim"]["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["optim"]["epochs"],
            eta_min=config["optim"]["min_lr"],
        )
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"], config["optim"]["min_lr"], places=12,
        )

    def test_time_reliability_reaches_dri_risk_bias(self):
        config = full_configuration()
        model = build_model(config).eval()
        physical = torch.tensor([104.0, 30.6, 2.5, 5.0, 5.0, 0.0]).expand(1, 15, 6)
        observed = (physical - model.dri_encoder.trajectory_min) / model.dri_encoder.trajectory_scale
        environment = torch.rand(1, 15, 15)
        kinematics = torch.randn(1, 15, 10)
        u2o = {
            "rel_xy_m": torch.ones(1, 15, 6, 2) * 25,
            "height_m": torch.ones(1, 15, 6) * 30,
            "dist_m": torch.ones(1, 15, 6) * np.sqrt(2 * 25 ** 2),
            "valid_mask": torch.ones(1, 15, 6, dtype=torch.bool),
        }
        with torch.no_grad():
            environment[..., 14] = 1
            _, reliable = model(observed, kinematics, environment, u2o=u2o)
            environment[..., 14] = 0
            _, unreliable = model(observed, kinematics, environment, u2o=u2o)
        difference = reliable["dri"]["u2o_risk_bias"] - unreliable["dri"]["u2o_risk_bias"]
        self.assertTrue((difference > 0).all())
        self.assertFalse(unreliable["lop_valid_mask"].any())
        self.assertTrue(unreliable["dri_active_mask"].all())

    def test_txt_time_and_derived_environment_activate_lop(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.txt"
            provider = write_environment(Path(temp))
            write_record(path, "UAV1", ["2026-04-26 12:00:00", "2026-04-26 12:00:02"])
            record = _read_record(path, provider)
            self.assertTrue((record.environment[:, 14] == 1).all())
            self.assertTrue(record.environment_valid.all())
            self.assertTrue(np.isfinite(record.environment).all())
            self.assertEqual(record.environment[0,0], 1.)
            self.assertAlmostEqual(float(record.environment[0,8]), 22 - .0065*23, places=5)
            np.testing.assert_allclose(record.environment[:,9:14].sum(axis=1),1,atol=.0002)
            encoder = LOPEncoder()
            _, _, valid = encoder(torch.zeros(1, 2, 10),
                                  torch.from_numpy(record.environment)[None],
                                  torch.from_numpy(record.environment_valid)[None])
            self.assertTrue(valid.all())

    def test_environment_missing_weather_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            provider = write_environment(Path(temp))
            with self.assertRaisesRegex(ValueError, "missing hourly weather"):
                provider.lookup(np.array([[104.,30.6,100.,0,0,0]]), [datetime(2026,4,27)])

    def test_wind_height_circular_direction_and_all_land_cover_radii(self):
        with tempfile.TemporaryDirectory() as temp:
            provider = write_environment(Path(temp))
            states = np.array([[104.,30.6,z,0,0,0] for z in (10,100,1000)])
            features, valid = provider.lookup(states,[datetime(2026,4,26,12)]*3)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(features[:,5],[10,20,40],rtol=1e-6)
            np.testing.assert_allclose(features[:,6],np.sin(np.deg2rad([350,10,30])),atol=1e-6)
            np.testing.assert_allclose(features[:,8],22-.0065*(states[:,2]-2),atol=1e-6)
            cover = provider.land_cover(states[:,0],states[:,1],(100,200,300,400,500))
            self.assertEqual(cover.shape,(3,5,5))
            np.testing.assert_allclose(cover.sum(axis=-1),1,atol=.0003)
            self.assertTrue(((cover>=0)&(cover<=1)).all())
            self.assertTrue((cover[0,1:,4]<=cover[0,:-1,4]).all())

    def test_missing_weather_values_cannot_activate_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            provider = write_environment(Path(temp))
            provider.weather[(416,122,"2026-04-26T12")][1] = np.nan
            features, valid = provider.lookup(np.array([[104.,30.6,100,0,0,0]]),[datetime(2026,4,26,12)])
            self.assertFalse(valid.any())
            self.assertTrue(np.isfinite(features).all())

    def test_missing_environment_suppresses_lop(self):
        model = LOPEncoder()
        environment = torch.rand(2, 15, 15)
        environment[..., 14] = 1
        valid = torch.ones(2, 15, 1, dtype=torch.bool)
        valid[0] = False
        preference, confidence, mask = model(torch.randn(2, 15, 10), environment, valid)
        self.assertFalse(preference[0].any())
        self.assertFalse(confidence[0].any())
        self.assertFalse(mask[0].any())
        self.assertTrue(mask[1].all())
        confidence.sum().backward()
        self.assertTrue(torch.isfinite(model.environment_gate.weight.grad).all())

    def test_midnight_and_timestamp_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.txt"
            write_record(path, "UAV1", ["2026-04-26 23:59:59", "2026-04-27 00:00:01"])
            record = _read_record(path)
            self.assertEqual(record.timestamp_us[1] - record.timestamp_us[0], 2_000_000)
            write_record(path, "UAV1", ["2026-04-26 12:00:02", "2026-04-26 12:00:00"])
            with self.assertRaises(ValueError):
                _read_record(path)

    def test_u2u_physical_identity_date_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "train"
            split.mkdir()
            specs = [
                ("ego", "UAV1", 26, 0, 0),
                ("self_overlap", "UAV1", 26, 0, 0),
                ("neighbour", "UAV2", 26, 100, 0),
                ("neighbour_duplicate", "UAV2", 26, 100, 0),
                ("next_day", "UAV3", 27, 0, 0),
                ("other_time", "UAV4", 26, 0, 1),
                ("outside", "UAV6", 26, 501, 0),
            ]
            for name, identity, day, east, minute in specs:
                path = split / f"{name}.txt"
                times = [f"2026-04-{day} 12:{minute:02d}:{s:02d}" for s in (0,2)]
                write_record(path, identity, times, east)
            index = write_buildings(root / "b.csv", [(100, 0)])
            ranges = {name: {"min": 0, "max": 200} for name in ("lon", "lat", "alt", "spdx", "spdy", "spdz")}
            dataset = TrajectoryWindowDataset(split, 2, 1, ranges, index)
            graph = dataset._u2u(0, 0)
            self.assertEqual(graph["valid_mask"].sum().item(), 2)
            self.assertTrue((graph["id_codes"][:, 0] == dataset.physical_id_codes["UAV2"]).all())
            for i, record in enumerate(dataset.records):
                if record.identifier in {"UAV3", "UAV4"}:
                    self.assertFalse(dataset._u2u(i, 0)["valid_mask"].any())

    def test_training_batch_validity_and_losses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "train"
            split.mkdir()
            times = [f"2026-04-26 12:{2 * i // 60:02d}:{2 * i % 60:02d}" for i in range(45)]
            path = split / "sample.txt"
            write_record(path, "UAV1", times)
            index = write_buildings(root / "b.csv", [(100, 0)])
            ranges = np.load(ROOT / "data_range.npy", allow_pickle=True).item()
            dataset = TrajectoryWindowDataset(split, 45, 1, ranges, index)
            prepared = _prepare_batch(default_collate([dataset[0]]), torch.device("cpu"), ranges, 15)
            self.assertFalse(prepared["environment_valid"].any())
            config = full_configuration()
            model = build_model(config)
            captured = {}
            hook = model.lop_encoder.register_forward_hook(lambda _, args, output: captured.update(mask=output[2].detach()))
            loss, _, _ = compute_objective(
                model,
                HaarWaveletCodec(),
                prepared,
                ranges,
                config["loss"],
            )
            hook.remove()
            self.assertFalse(captured["mask"].any())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class NormalizationTests(unittest.TestCase):
    def test_preparation_uses_training_txt_only_and_converts_height_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            states = np.array([[104., 30.6, 100., 1., 2., 3.],
                               [104.001, 30.602, 200., 4., 5., 6.]])
            for split in ("train", "dev", "test"):
                (root / split).mkdir()
            rows = [f"2026-04-26 12:00:{2 * i:02d}|UAV1|"
                    + "|".join(str(value) for value in state)
                    + f"|{np.linalg.norm(state[3:])}" for i, state in enumerate(states)]
            (root / "train/UAV1.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            for split in ("dev", "test"):
                (root / split / "unread.txt").write_text("must not be read", encoding="utf-8")
            config = full_configuration()
            output = root / "normalization/data_range.npy"
            config["data"].update(root=str(root), data_range_path=str(output))
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("train.build_loaders", side_effect=AssertionError("preparation must not load dev/test or maps")):
                summary = prepare_normalization(config_path)
            self.assertEqual(summary["training_files"], 1)
            self.assertEqual(summary["training_points"], 2)
            self.assertEqual(summary["height_divisor"], 10)
            ranges = np.load(output, allow_pickle=True).item()
            names = ("lon", "lat", "alt", "spdx", "spdy", "spdz")
            self.assertEqual(tuple(ranges), names)
            states[:, 2] /= 10.
            for index, name in enumerate(names):
                values = states[:, index]
                for key, expected in (("min", values.min()), ("max", values.max()),
                                      ("mu", values.mean()), ("sigma", values.std(ddof=0))):
                    self.assertEqual(ranges[name][key], expected)

    def test_preparation_rejects_shared_split_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "data_range.npy"
            output.write_bytes(b"unchanged")
            config_path = root / "config.json"
            for shared in ("dev_split", "test_split"):
                config = full_configuration()
                config["data"].update(root=str(root), data_range_path=str(output))
                config["data"][shared] = "train"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "training split must differ"):
                    prepare_normalization(config_path)
                self.assertEqual(output.read_bytes(), b"unchanged")

    def test_data_range_rejects_empty_constant_and_nonfinite_training_data(self):
        with self.assertRaisesRegex(ValueError, "must contain trajectories"):
            fit_data_range([])
        for states, message in ((np.ones((2, 6)), "zero range"),
                                (np.full((2, 6), np.nan), "must be finite")):
            with self.assertRaisesRegex(ValueError, message):
                fit_data_range([SimpleNamespace(trajectory_unscaled=states)])

    def test_normalization_cli_exits_without_training(self):
        config_path = ROOT / "optDir/opt_DROP.json"
        with patch("sys.argv", ["train.py", "--prepare-normalization", "--config", str(config_path)]), \
                patch("train.prepare_normalization") as prepare, \
                patch("train.train", side_effect=AssertionError("must not train")):
            training_main()
            prepare.assert_called_once_with(config_path)
        with patch("sys.argv", ["train.py"]), patch("train.train") as training, \
                patch("train.prepare_normalization", side_effect=AssertionError("must be explicit")):
            training_main()
            training.assert_called_once_with(config_path)

    def test_lop_fit_uses_observed_points_only_and_handles_constant_features(self):
        states = np.arange(270, dtype=np.float64).reshape(45, 6) / 100.
        states[:, :2] += [LON, LAT]
        environment = np.zeros((45, 15), dtype=np.float32)
        environment[:, [0, 14]] = 1.
        record = SimpleNamespace(trajectory_unscaled=states, environment=environment,
                                 environment_valid=np.ones((45, 1), dtype=bool))
        dataset = SimpleNamespace(records=[record], windows=[(0, 0)])
        ranges, summary = fit_data_range(dataset.records)
        encoder = LOPEncoder()
        self.assertEqual(encoder.fit_normalization(dataset), 15)
        names = ("kin_mean", "kin_std", "env_mean", "env_std")
        statistics = {name: getattr(encoder, name).clone() for name in names}
        self.assertEqual(summary["training_points"], 45)
        self.assertEqual(ranges["alt"]["max"], states[:, 2].max())
        torch.testing.assert_close(statistics["env_std"], torch.ones(1, 1, 15))
        states[15:] += 20.
        environment[15:] += 30.
        new_ranges, _ = fit_data_range(dataset.records)
        self.assertEqual(encoder.fit_normalization(dataset), 15)
        self.assertNotEqual(ranges["lon"]["max"], new_ranges["lon"]["max"])
        for name in statistics:
            torch.testing.assert_close(statistics[name], getattr(encoder, name), rtol=0, atol=0)
        states[:15, 3] += 2.
        encoder.fit_normalization(dataset)
        self.assertFalse(torch.equal(encoder.kin_mean, statistics["kin_mean"]))
        expected = {name: getattr(encoder, name).clone() for name in names}
        for mode in (True, False):
            encoder.train(mode)
            encoder(torch.zeros(1, 15, 6), torch.from_numpy(environment[:15])[None])
            for name in names:
                torch.testing.assert_close(getattr(encoder, name), expected[name], rtol=0, atol=0)
        record.environment_valid[:] = False
        with self.assertRaisesRegex(ValueError, "valid training observations"):
            encoder.fit_normalization(dataset)

    def test_model_fits_only_when_training_dataset_is_supplied(self):
        config = full_configuration()
        self.assertNotIn("lop_stats_path", config["data"])
        dataset = object()
        with patch.object(LOPEncoder, "fit_normalization", return_value=360) as fit:
            model = build_model(config, dataset)
            fit.assert_called_once_with(dataset, 15, 2.)
            fit.reset_mock()
            build_model(config)
            fit.assert_not_called()
            for name in ("kin_mean", "kin_std", "env_mean", "env_std"):
                self.assertNotIn("lop_" + name, config["data"])
                self.assertIn("lop_encoder." + name, model.state_dict())

    def test_evaluation_rejects_mismatched_range_and_invalid_lop_statistics(self):
        config = full_configuration()
        config["model"].update(d_model=16, encoder_layers=1, decoder_layers=1)
        state = build_model(config).state_dict()
        for name, value, message in (
            ("dri_encoder.trajectory_min", state["dri_encoder.trajectory_min"] + 0.1, "data_range differs"),
            ("lop_encoder.kin_mean", torch.full_like(state["lop_encoder.kin_mean"], float("nan")), "invalid LOP"),
            ("lop_encoder.env_std", torch.zeros_like(state["lop_encoder.env_std"]), "invalid LOP"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                changed = dict(state)
                changed[name] = value
                checkpoint = Path(temp) / "mismatch.pt"
                torch.save({"configuration": config, "model": changed, "epoch": 0}, checkpoint)
                output = Path(temp) / "evaluation"
                with self.assertRaisesRegex(ValueError, message):
                    evaluate(checkpoint, "test", output, cpu=True)
                self.assertFalse(output.exists())


class ModuleSwitchTests(unittest.TestCase):
    def setUp(self):
        self.config = full_configuration()
        self.config["model"].update(d_model=16, encoder_layers=1, decoder_layers=1, dropout=0)

    def test_switches_require_booleans_and_default_to_full(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for name in ("use_lop", "use_dri"):
                for value in (None, 0, 1, "true", "false"):
                    with self.subTest(name=name, value=value):
                        self.config["model"][name] = value
                        path.write_text(json.dumps(self.config), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, "must be a boolean"):
                            load_configuration(path)
                        with self.assertRaisesRegex(ValueError, "must be a boolean"):
                            build_model(self.config)
                        with self.assertRaisesRegex(ValueError, "must be a boolean"):
                            DROPConfig(**{name: value})
                self.config["model"][name] = True
            full = build_model(self.config)
            for name in ("use_lop", "use_dri"):
                del self.config["model"][name]
            previous = build_model(self.config)
            previous.load_state_dict(full.state_dict(), strict=True)
            self.assertTrue(previous.cfg.use_lop and previous.cfg.use_dri)

    def test_disabled_modules_are_not_constructed_or_fitted(self):
        for use_lop, use_dri in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(use_lop=use_lop, use_dri=use_dri), ExitStack() as stack:
                self.config["model"].update(use_lop=use_lop, use_dri=use_dri)
                if not use_lop:
                    stack.enter_context(patch("DROP_model.LOPEncoder", side_effect=AssertionError("LOP constructed")))
                if not use_dri:
                    for target in ("train.DynamicRiskInteractionEncoder", "DROP_model.DynamicRiskInteractionEncoder"):
                        stack.enter_context(patch(target, side_effect=AssertionError("DRI constructed")))
                fit = stack.enter_context(patch.object(LOPEncoder, "fit_normalization", return_value=30))
                model = build_model(self.config, object())
                self.assertEqual(fit.call_count, int(use_lop))
                self.assertEqual(model.lop_encoder is not None, use_lop)
                self.assertEqual(model.dri_encoder is not None, use_dri)
                state = model.state_dict()
                for name, enabled in (("lop", use_lop), ("dri", use_dri)):
                    self.assertEqual(any(key.startswith(name + "_encoder.") for key in state), enabled)
                    self.assertEqual(any(key.startswith("fusion." + name + "_") for key in state), enabled)
                self.assertEqual(model.fusion.fusion_mlp is not None, use_lop or use_dri)

    def test_disabled_inputs_are_ignored_and_counterfactual_is_skipped(self):
        observed = torch.rand(2, 15, 6)
        kinematics = torch.randn(2, 15, 6)
        environment = torch.rand(2, 15, 15)
        environment[..., 14] = 1
        u2o = {
            "rel_xy_m": torch.full((2, 15, 1, 2), 40.),
            "height_m": torch.full((2, 15, 1), 30.),
            "dist_m": torch.full((2, 15, 1), 50.),
            "valid_mask": torch.ones(2, 15, 1, dtype=torch.bool),
        }
        for use_lop, use_dri in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(use_lop=use_lop, use_dri=use_dri):
                self.config["model"].update(use_lop=use_lop, use_dri=use_dri)
                model = build_model(self.config).eval()
                with torch.no_grad(), patch.object(model.decoder, "forward", wraps=model.decoder.forward) as decode:
                    prediction, aux = model(observed, kinematics, environment, u2o=u2o, return_lop_counterfactual=True)
                    self.assertEqual(decode.call_count, 1 + int(use_lop))
                self.assertEqual(prediction.shape, (2, 31, 6))
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertEqual("lop_counterfactual_coefficients" in aux, use_lop)
                if not use_lop:
                    self.assertFalse(aux["lop_valid_mask"].any())
                    self.assertFalse(aux["preference_projection"].any())
                if not use_dri:
                    self.assertFalse(aux["dri_active_mask"].any())
                    self.assertFalse(aux["risk_projection"].any())
                changed_environment = environment.clone()
                if not use_lop:
                    changed_environment[..., :14] = float("nan")
                with torch.no_grad():
                    changed, _ = model(
                        observed, kinematics if use_lop else torch.full_like(kinematics, float("nan")),
                        changed_environment if use_lop or use_dri else None,
                        u2o=u2o if use_dri else {"invalid": None},
                        u2u=None if use_dri else {"invalid": None},
                    )
                torch.testing.assert_close(prediction, changed, rtol=0, atol=0)

    def test_four_variants_train_reload_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "train"
            split.mkdir()
            times = [f"2026-04-26 00:{second // 60:02d}:{second % 60:02d}" for second in range(0, 90, 2)]
            for number in range(2):
                write_record(split / f"UAV{number}.txt", f"UAV{number}", times, east=number * 50)
            environment = write_environment(root)
            index = polygon_index(root / "building.csv")
            ranges = np.load(ROOT / "data_range.npy", allow_pickle=True).item()
            dataset = TrajectoryWindowDataset(split, 45, 1, ranges, index, environment_features=environment)
            prepared = _prepare_batch(default_collate([dataset[0], dataset[1]]), torch.device("cpu"), ranges, 15)
            self.assertTrue(prepared["environment_valid"].all())
            self.assertTrue(prepared["u2u"]["valid_mask"].any())
            self.assertTrue(prepared["u2o"]["valid_mask"].any())
            self.config["data"].update(root=str(root), test_split="train", building_csv_paths=[str(root / "building.csv")])
            for use_lop, use_dri in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(use_lop=use_lop, use_dri=use_dri), ExitStack() as stack:
                    self.config["model"].update(use_lop=use_lop, use_dri=use_dri)
                    if not use_lop:
                        stack.enter_context(patch("train.lop_supervision_loss", side_effect=AssertionError("LOP loss called")))
                    if not use_dri:
                        stack.enter_context(patch("train.make_future_obstacles", side_effect=AssertionError("DRI targets computed")))
                        stack.enter_context(patch("train.dri_obstacle_proximity_loss", side_effect=AssertionError("DRI loss called")))
                    model = build_model(self.config, dataset)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                    loss, diagnostics, _ = compute_objective(model, HaarWaveletCodec(), prepared, ranges, self.config["loss"])
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    for name, enabled in (("lop", use_lop), ("dri", use_dri)):
                        if not enabled:
                            self.assertEqual(float(diagnostics["loss_" + name]), 0.)
                        else:
                            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                                for p in getattr(model, name + "_encoder").parameters()))
                    optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        expected, _ = model(prepared["observed_scaled"], prepared["observed_enu"], prepared["environment"],
                                            u2u=prepared["u2u"], u2o=prepared["u2o"], environment_valid=prepared["environment_valid"])
                    checkpoint = root / f"checkpoint_{int(use_lop)}_{int(use_dri)}.pt"
                    torch.save({"model": model.state_dict(), "configuration": self.config, "epoch": 1}, checkpoint)
                    saved = torch.load(checkpoint, weights_only=True)
                    restored = build_model(saved["configuration"]).eval()
                    restored.load_state_dict(saved["model"], strict=True)
                    with torch.no_grad():
                        actual, _ = restored(prepared["observed_scaled"], prepared["observed_enu"], prepared["environment"],
                                             u2u=prepared["u2u"], u2o=prepared["u2o"], environment_valid=prepared["environment_valid"])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    output = root / checkpoint.stem
                    with patch.object(LOPEncoder, "fit_normalization", side_effect=AssertionError("evaluation fitted statistics")):
                        payload = evaluate(checkpoint, "test", output, cpu=True)
                    self.assertEqual(payload["rows"][0]["sample_count"], 2)
                    self.assertEqual(payload["rows"][0]["safety_sample_count"], 2)
                    manifest = json.loads((output / "window_manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["modules"], {"use_lop": use_lop, "use_dri": use_dri})
                    self.assertEqual(manifest["lop_statistics"] is not None, use_lop)

    def test_normalization_guard_remains_active_without_dri(self):
        for use_lop in (False, True):
            with self.subTest(use_lop=use_lop), tempfile.TemporaryDirectory() as temp:
                self.config["model"].update(use_lop=use_lop, use_dri=False)
                state = build_model(self.config).state_dict()
                state["trajectory_min"] = state["trajectory_min"] + .1
                checkpoint = Path(temp) / "mismatch.pt"
                torch.save({"configuration": self.config, "model": state, "epoch": 0}, checkpoint)
                with self.assertRaisesRegex(ValueError, "data_range differs"):
                    evaluate(checkpoint, "test", Path(temp) / "evaluation", cpu=True)


class HeightTests(unittest.TestCase):
    def test_agl_obstacle_height_is_not_replaced_by_ground_or_roof_elevation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "building.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["lon", "lat", "height", "altitude", "PeakHeight"])
                writer.writerow([LON, LAT, 30., 500., 530.])
            index = BuildingObstacleIndex([str(path)])
            self.assertEqual(float(index.height[0]), 30.)
            result = index.query_batch(np.array([[LON]]), np.array([[LAT]]))
            self.assertEqual(float(result["contour_xyz"][0, 0, 0, 2]), 30.)

    def test_dri_vertical_features_use_agl_metres(self):
        config = full_configuration()
        encoder = build_model(config).dri_encoder.eval()
        physical = torch.tensor([LON, LAT, 4., 0., 0., 0.]).expand(1, 15, 6)
        observed = (physical - encoder.trajectory_min) / encoder.trajectory_scale
        u2o = {
            "rel_xy_m": torch.tensor([50., 0.]).expand(1, 15, 1, 2),
            "height_m": torch.full((1, 15, 1), 30.),
            "dist_m": torch.full((1, 15, 1), 50.),
            "valid_mask": torch.ones(1, 15, 1, dtype=torch.bool),
        }
        captured = {}
        hook = encoder.u2o_projection.register_forward_pre_hook(
            lambda _, args: captured.update(features=args[0].detach())
        )
        with torch.no_grad():
            encoder(observed, u2o=u2o)
        hook.remove()
        torch.testing.assert_close(captured["features"][0, :, 0, 2], torch.full((15,), -0.1))
        torch.testing.assert_close(captured["features"][0, :, 0, 3], torch.full((15,), 0.3))


class EvaluationTests(unittest.TestCase):
    def test_signed_distance_to_complete_building_solid(self):
        with tempfile.TemporaryDirectory() as temp:
            index = polygon_index(Path(temp) / "building.csv")
            polygon = index._projected_geometry(0)
            centre = polygon.centroid
            self.assertAlmostEqual(signed_prism_distance(centre, 15, polygon, 30), -10.0, places=3)
            outside = Point(polygon.bounds[2] + 5, centre.y)
            self.assertAlmostEqual(signed_prism_distance(outside, 35, polygon, 30), np.sqrt(50), places=3)

    def test_trajectory_safety_set_depends_only_on_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            index = polygon_index(Path(temp) / "building.csv")
            centre_x, centre_y = TO_METRES.transform(LON, LAT)
            prediction = np.zeros((1, 30, 6), dtype=np.float64)
            target = np.zeros_like(prediction)
            pred_lon, pred_lat = TO_WGS84.transform(centre_x, centre_y)
            true_lon, true_lat = TO_WGS84.transform(centre_x + 300, centre_y)
            prediction[..., :3] = [pred_lon, pred_lat, 15]
            target[..., :3] = [true_lon, true_lat, 15]
            result = PolygonSafety(index).pair(prediction, target)
            self.assertFalse(result["active"].any())
            self.assertGreater(float(result["delta_error"].mean()), 0.0)
            prediction, target = target, prediction
            result = PolygonSafety(index).pair(prediction, target)
            self.assertTrue(result["active"].all())
            self.assertTrue(np.isfinite(result["pred_clearance"]).all())
            self.assertTrue(np.isfinite(result["true_clearance"]).all())

    def test_nine_metric_types_and_four_horizons(self):
        class EmptySafety:
            def pair(self, prediction, target):
                shape = prediction.shape[:2]
                return {
                    "delta_error": np.zeros(shape), "height_error": np.zeros(shape),
                    "pred_clearance": np.full(shape, np.inf),
                    "true_clearance": np.full(shape, np.inf),
                    "active": np.zeros(shape, dtype=bool),
                }

        target = np.zeros((2, 30, 6), dtype=np.float64)
        target[..., :3] = [104.0, 30.6, 100.0]
        prediction = target.copy()
        prediction[..., 2] += 10.0
        accumulator = MetricAccumulator(EmptySafety())
        accumulator.update(prediction[:1], target[:1])
        accumulator.update(prediction[1:], target[1:])
        rows = accumulator.rows()
        self.assertEqual([row["horizon_steps"] for row in rows], list(HORIZONS))
        required = {
            "mae_lon_deg", "rmse_lon_deg", "mae_spdx_mps", "rmse_spdz_mps",
            "mape_lon_permille", "mape_lon_percent", "mape_lat_percent",
            "mde_m", "fde_m", "nmde", "ground_truth_path_length_m",
            "delta_b_mae", "h_r_mae", "msce_m", "fpete_s",
        }
        for row in rows:
            self.assertTrue(required.issubset(row))
            self.assertAlmostEqual(row["mae_alt_m"], 10.0)
            self.assertAlmostEqual(row["rmse_alt_m"], 10.0)
            self.assertAlmostEqual(row["mape_alt_percent"], 10.0)
            self.assertAlmostEqual(row["mde_m"], 10.0, places=5)
            self.assertAlmostEqual(row["fde_m"], 10.0, places=5)
            self.assertAlmostEqual(row["mae_spdx_mps"], 0.0)
            self.assertIsNone(row["nmde"])
            self.assertIsNone(row["msce_m"])
            self.assertIsNone(row["fpete_s"])

    def test_formal_evaluation_entry_writes_reproducible_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "test_01"
            split.mkdir()
            times = [f"2026-04-26 00:{second // 60:02d}:{second % 60:02d}" for second in range(0, 90, 2)]
            write_record(split / "trajectory.txt", "UAV1", times)
            write_environment(root)
            building_path = root / "building.csv"
            polygon_index(building_path)
            config = full_configuration()
            config["data"].update(
                root=str(root), data_range_path=str(ROOT / "data_range.npy"),
                building_csv_paths=[str(building_path)], eval_batch_size=1,
                test_split="test_01", dev_split="test_01",
            )
            config["model"].update(d_model=16, encoder_layers=1, decoder_layers=1, dropout=0)
            model = build_model(config)
            model.lop_encoder.kin_mean.fill_(0.25)
            model.lop_encoder.env_std.fill_(2.)
            checkpoint = root / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "configuration": config, "epoch": 0}, checkpoint)
            output = root / "evaluation"
            range_path = ROOT / "data_range.npy"
            range_before = range_path.read_bytes()
            with patch.object(LOPEncoder, "fit_normalization", side_effect=AssertionError("evaluation must not fit statistics")), \
                    patch("train.prepare_normalization", side_effect=AssertionError("evaluation must not prepare statistics")), \
                    patch("train.fit_data_range", side_effect=AssertionError("evaluation must not fit ranges")), \
                    patch("dataloader.fit_data_range", side_effect=AssertionError("evaluation must not fit ranges")):
                payload = evaluate(checkpoint, "test", output, cpu=True, batch_size=1)
                dev_payload = evaluate(checkpoint, "dev", root / "dev_evaluation", cpu=True, batch_size=1)
            self.assertEqual(range_path.read_bytes(), range_before)
            self.assertEqual(dev_payload["rows"], payload["rows"])
            self.assertEqual(len(payload["rows"]), 4)
            self.assertEqual(payload["rows"][0]["sample_count"], 1)
            for name in ("metrics.json", "metrics.csv", "window_manifest.json", "window_index.csv"):
                self.assertTrue((output / name).is_file())
            manifest = json.loads((output / "window_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["window_count"], 1)
            self.assertEqual(manifest["files"][0]["file"], "trajectory.txt")
            self.assertEqual(len(manifest["data_range"]["sha256"]), 64)
            self.assertEqual(manifest["lop_statistics"]["source"], "checkpoint.model.lop_encoder")
            self.assertEqual(manifest["lop_statistics"]["kin_mean"], [0.25] * 10)
            self.assertEqual(manifest["lop_statistics"]["env_std"], [2.] * 15)
            self.assertEqual(len(manifest["window_index_sha256"]), 64)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
