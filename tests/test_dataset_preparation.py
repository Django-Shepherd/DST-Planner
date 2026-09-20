"""Exercise strict admission and map-disjoint splits using small complete episodes."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learning"))
from dst_planner.data import admit_episode, build_dataset, episode
from dst_planner.features import FEATURE_CONFIG


class DatasetPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()

    def write_json(self, path, value):
        path.write_text(json.dumps(value))

    def make_episode(self, name, map_name, kind="forest"):
        scene = self.root / "maps" / map_name
        scene.mkdir(parents=True, exist_ok=True)
        self.write_json(scene / "scene.json", {"kind": kind})
        path = self.runs / name
        path.mkdir()
        self.write_json(
            path / "metadata.json",
            {
                "schema_version": 1,
                "baseline": False,
                "learning_features": True,
                "policy_mode": "epic",
                "coverage_grid": str(scene / "coverage_grid.npz"),
                "map": str(scene / "map.pcd"),
                "map_sha256": hashlib.sha256(map_name.encode()).hexdigest(),
            },
        )
        self.write_json(
            path / "summary.json",
            {
                "errors": [],
                "reason": "epic_finish",
                "terminated": True,
                "truncated": False,
                "distance_m": 2.0,
                "odometry_count": 2,
                "trajectory_count": 1,
                "coverage": {
                    "observed_fraction": 0.98,
                    "near_obstacle_odometry_samples": 0,
                    "outside_map_odometry_samples": 0,
                },
            },
        )
        decision = {
            "schema_version": 1,
            "decision_id": 0,
            "stamp": 1.0,
            "position": [0.0, 0.0, 1.0],
            "velocity": [0.0, 0.0, 0.0],
            "planning_anchor": [0.0, 0.0, 1.0],
            "yaw": 0.0,
            "capture_ms": 0.1,
            "nodes": [
                {"id": 0, "position": [0.0, 0.0, 1.0], "is_current": True},
                {"id": 1, "position": [2.0, 0.0, 1.0], "is_current": False},
            ],
            "current_node_id": 0,
            "edges": [[0, 1, 2.0], [1, 0, 2.0]],
            "candidates": [
                {
                    "id": 0,
                    "node_id": 1,
                    "position": [2.0, 0.0, 1.0],
                    "yaw": 0.0,
                    "odom_cost": 2.0,
                },
            ],
            "result": "single_candidate",
            "expert_route": [0, 1],
            "matrix_dimension": 0,
            "cost_matrix": [],
            "selected_candidate_id": 0,
            "feature_version": FEATURE_CONFIG["version"],
            "gain_definition": FEATURE_CONFIG["gain_definition"],
            "information_gain_m3": [10.0],
            "reachability_dimension": 2,
            "reachable": [False, True, True, False],
        }
        (path / "decisions.jsonl").write_text(json.dumps(decision) + "\n")
        self.write_json(
            path / "decisions.jsonl.status.json",
            {
                "schema_version": 1,
                "failed": False,
                "enqueued": 1,
                "written": 1,
            },
        )
        events = [
            {"event": "trigger", "stamp": 0.0},
            {
                "event": "odometry",
                "stamp": 1.0,
                "position": [0.0, 0.0, 1.0],
                "velocity": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "event": "trajectory",
                "stamp": 1.0,
                "start_time": 1.0,
                "traj_id": 0,
                "duration": [2.0],
                "order": 1,
                "coef_x": [0.0, 1.0],
                "coef_y": [0.0, 0.0],
                "coef_z": [1.0, 0.0],
            },
            {
                "event": "execution",
                "stamp": 1.0,
                "start_time": 1.0,
                "kind": "trajectory_applied",
                "decision_id": 0,
                "traj_id": 0,
                "goal": [2.0, 0.0, 1.0],
            },
            {
                "event": "odometry",
                "stamp": 3.0,
                "position": [2.0, 0.0, 1.0],
                "velocity": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            {"event": "fsm", "stamp": 3.0, "state": "FINISH"},
            {"event": "episode_end", "stamp": 3.0, "reason": "epic_finish"},
        ]
        (path / "events.jsonl").write_text(
            "".join(
                json.dumps(dict(event, wall_time=event["stamp"])) + "\n"
                for event in events
            )
        )
        return path

    def load_dataset(self, output):
        return torch.load(output / "dataset.pt", map_location="cpu", weights_only=False)

    def test_real_episode_keeps_raw_distance_reward_and_expert_labels(self):
        path = self.make_episode("run", "map")
        samples, labels, audit = episode(path)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["reward"], 98.0)
        self.assertEqual(samples[0]["distance_m"], 2.0)
        self.assertEqual(samples[0]["duration_s"], 2.0)
        self.assertEqual(samples[0]["expert"], [0])
        self.assertTrue(samples[0]["terminated"])
        self.assertFalse(samples[0]["truncated"])
        self.assertEqual(labels[0]["expert"], [0])
        self.assertEqual(audit["finish_rewards"], 1)

    def test_nonexpert_and_missing_coverage_are_rejected_before_labeling(self):
        for name, change in (
            ("policy", lambda meta, summary: meta.update(policy_mode="dst")),
            ("grid", lambda meta, summary: meta.pop("coverage_grid")),
            ("coverage", lambda meta, summary: summary.pop("coverage")),
        ):
            with self.subTest(name=name):
                path = self.make_episode(name, name)
                meta = json.loads((path / "metadata.json").read_text())
                summary = json.loads((path / "summary.json").read_text())
                change(meta, summary)
                self.write_json(path / "metadata.json", meta)
                self.write_json(path / "summary.json", summary)
                with self.assertRaisesRegex(ValueError, "expert|Coverage|coverage"):
                    admit_episode(path)

    def test_invalid_coverage_or_safety_audit_is_rejected(self):
        for index, changes in enumerate(
            (
                {"observed_fraction": 0.949},
                {"observed_fraction": float("nan")},
                {"observed_fraction": None},
                {"near_obstacle_odometry_samples": 1},
                {"outside_map_odometry_samples": 1},
            )
        ):
            with self.subTest(changes=changes):
                path = self.make_episode(f"run-{index}", f"map-{index}")
                summary = json.loads((path / "summary.json").read_text())
                summary["coverage"].update(changes)
                self.write_json(path / "summary.json", summary)
                with self.assertRaises(ValueError):
                    admit_episode(path)

    def test_duplicate_roots_and_starts_preserve_stratified_map_disjoint_splits(self):
        map_kinds = {}
        for kind in ("forest", "partition", "dungeon"):
            for index in range(5):
                map_name = f"{kind}-{index}"
                map_kinds[hashlib.sha256(map_name.encode()).hexdigest()] = kind
                for start in range(2):
                    self.make_episode(f"{map_name}-start-{start}", map_name, kind)
        output = self.root / "dataset"
        report = build_dataset([self.runs, self.runs / "."], output)
        data = self.load_dataset(output)
        self.assertEqual(
            set(data),
            {
                "train",
                "validation",
                "bc_train",
                "bc_validation",
                "feature_config",
            },
        )
        self.assertEqual(data["feature_config"], FEATURE_CONFIG)
        self.assertEqual(len(report["episodes"]), 30)
        self.assertEqual(set(report["episodes_per_map"].values()), {2})
        self.assertEqual(
            report["counts"],
            {
                "train": 24,
                "validation": 6,
                "bc_train": 24,
                "bc_validation": 6,
            },
        )
        training = set(report["training_maps"])
        validation = set(report["validation_maps"])
        self.assertFalse(training & validation)
        self.assertEqual(training | validation, set(map_kinds))
        self.assertEqual(
            sorted(map_kinds[value] for value in validation),
            ["dungeon", "forest", "partition"],
        )
        for split, maps in (("train", training), ("validation", validation)):
            self.assertEqual({sample["map_sha256"] for sample in data[split]}, maps)
            self.assertEqual(
                {sample["map_sha256"] for sample in data["bc_" + split]}, maps
            )
        self.assertEqual(report["rejected"], [])
        self.assertEqual(json.loads((output / "manifest.json").read_text()), report)
        self.assertEqual(
            report["dataset_sha256"],
            hashlib.sha256((output / "dataset.pt").read_bytes()).hexdigest(),
        )

    def test_count_failure_keeps_admission_rejections_without_dataset(self):
        self.make_episode("accepted", "forest-a")
        rejected = self.make_episode("rejected", "forest-b")
        metadata = json.loads((rejected / "metadata.json").read_text())
        metadata["policy_mode"] = "dst"
        self.write_json(rejected / "metadata.json", metadata)
        for name, minimums, expected in (
            ("episodes", {"min_episodes": 2, "min_maps": 1}, "accepted episodes"),
            ("maps", {"min_episodes": 1, "min_maps": 2}, "accepted maps"),
        ):
            with self.subTest(name=name):
                output = self.root / f"dataset-{name}"
                with self.assertRaisesRegex(ValueError, expected):
                    build_dataset([self.runs], output, **minimums)
                admission = json.loads((output / "admission.json").read_text())
                self.assertEqual(len(admission["episodes"]), 1)
                self.assertEqual(len(admission["rejected"]), 1)
                self.assertIn("Only EPIC expert", admission["rejected"][0]["reason"])
                self.assertFalse((output / "dataset.pt").exists())
                self.assertFalse((output / "manifest.json").exists())

    def test_split_failure_keeps_admission_report(self):
        self.make_episode("forest", "forest-a")
        self.make_episode("dungeon", "dungeon-a", "dungeon")
        output = self.root / "dataset"
        with self.assertRaisesRegex(ValueError, "at least two maps per scene type"):
            build_dataset([self.runs], output, min_maps=2, min_episodes=2)
        self.assertTrue((output / "admission.json").exists())
        self.assertFalse((output / "dataset.pt").exists())

    def test_cli_builds_dataset_and_rejects_legacy_entry_arguments(self):
        for index in range(3):
            self.make_episode(f"run-{index}", f"map-{index}")
        # Running from learning keeps this test independent of the caller's PYTHONPATH.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dst_planner.data",
                "--roots",
                str(self.runs),
                "--output",
                str(self.root / "dataset"),
            ],
            cwd=ROOT / "learning",
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "train": 2,
                "validation": 1,
                "bc_train": 2,
                "bc_validation": 1,
            },
        )
        legacy = subprocess.run(
            [
                sys.executable,
                "-m",
                "dst_planner.data",
                "--episodes",
                str(self.runs),
                "--output",
                str(self.root / "legacy"),
            ],
            cwd=ROOT / "learning",
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(legacy.returncode, 2)
        self.assertFalse((self.root / "legacy").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
