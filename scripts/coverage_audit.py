"""Passive 2D audit: floor cells actually returned by LiDAR, within known reachable free space.
This is a conservative floor-observation metric, not a 3D volumetric coverage certificate.
"""

import json
from pathlib import Path
import threading
import numpy as np


class CoverageAudit:
    def __init__(self, grid):
        data = np.load(grid)
        self.reachable = data["reachable"]
        self.occupied = data["occupied"]
        self.clearance = data["clearance"]
        self.resolution = float(data["resolution"])
        self.observed = np.zeros_like(self.reachable)
        self.lock = threading.Lock()
        self.last_stamp = -float("inf")
        self.frames = 0
        self.minimum_clearance = float("inf")
        self.near_obstacle_samples = 0
        self.outside_samples = 0

    def odometry(self, position):
        x, y, z = position
        ix = int(np.floor(x / self.resolution))
        iy = int(np.floor(y / self.resolution))
        with self.lock:
            if 0 <= iy < self.clearance.shape[0] and 0 <= ix < self.clearance.shape[1]:
                clearance = float(self.clearance[iy, ix])
                self.minimum_clearance = min(self.minimum_clearance, clearance)
                if clearance < 0.5 and z < 2.8:
                    self.near_obstacle_samples += 1
            else:
                self.outside_samples += 1

    def cloud(self, msg):
        stamp = msg.header.stamp.to_sec()
        if stamp - self.last_stamp < 0.2:
            return
        if msg.header.frame_id.lstrip("/") != "world":
            raise ValueError("Coverage requires world-frame points")
        fields = {f.name: f for f in msg.fields}
        if any(fields[k].datatype != 7 for k in ["x", "y", "z"]):
            raise ValueError("Expected float32 XYZ cloud")
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [(">" if msg.is_bigendian else "<") + "f4"] * 3,
                "offsets": [fields[k].offset for k in ["x", "y", "z"]],
                "itemsize": msg.point_step,
            }
        )
        points = np.ndarray(
            (msg.height, msg.width),
            dtype=dtype,
            buffer=msg.data,
            strides=(msg.row_step, msg.point_step),
        ).reshape(-1)
        mask = (
            np.isfinite(points["x"])
            & np.isfinite(points["y"])
            & (points["z"] >= -0.15)
            & (points["z"] <= 0.2)
        )
        ix = np.floor(points["x"][mask] / self.resolution).astype(int)
        iy = np.floor(points["y"][mask] / self.resolution).astype(int)
        valid = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < self.reachable.shape[1])
            & (iy < self.reachable.shape[0])
        )
        with self.lock:
            self.observed[iy[valid], ix[valid]] = True
            self.frames += 1
            self.last_stamp = stamp

    def save(self, directory):
        with self.lock:
            count = int((self.observed & self.reachable).sum())
            total = int(self.reachable.sum())
            result = {
                "metric": "reachable_free_floor_observed_by_lidar",
                "resolution_m": self.resolution,
                "observed_cells": count,
                "reachable_cells": total,
                "observed_fraction": count / total if total else None,
                "observed_area_m2": count * self.resolution**2,
                "cloud_frames": self.frames,
                "minimum_projected_clearance_m": self.minimum_clearance
                if np.isfinite(self.minimum_clearance)
                else None,
                "near_obstacle_odometry_samples": self.near_obstacle_samples,
                "outside_map_odometry_samples": self.outside_samples,
            }
            directory = Path(directory)
            np.savez_compressed(
                directory / "coverage.npz",
                observed=self.observed,
                reachable=self.reachable,
                occupied=self.occupied,
                resolution=self.resolution,
            )
            (directory / "coverage.json").write_text(json.dumps(result, indent=2))
        return result
