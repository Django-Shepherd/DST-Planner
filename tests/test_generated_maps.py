"""Regression checks for migrated generation and the passive coverage metric."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "maps"), str(ROOT / "scripts")]
from forest_map_generator import ForestMapGenerator
from coverage_audit import CoverageAudit


class GeneratedMapsTest(unittest.TestCase):
    def test_python_batch_seed_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(directory)
            generated = []
            for output in [root / "first", root / "repeat"]:
                generator = ForestMapGenerator(
                    room_size=8,
                    num_columns=4,
                    min_radius=0.3,
                    max_radius=0.5,
                    output_dir=str(output),
                )
                result = generator.generate_all(prefix="scene", batch_size=2, seed=42)
                self.assertEqual(generator.output_dir, str(output))
                snapshots = {
                    Path(path).relative_to(output): Path(path).read_bytes()
                    for paths in result.values()
                    for path in paths
                }
                generated.append(snapshots)
                with self.assertRaises(FileExistsError):
                    generator.generate_all(prefix="scene", batch_size=2, seed=43)
                self.assertEqual(generator.output_dir, str(output))
                for path, content in snapshots.items():
                    self.assertEqual((output / path).read_bytes(), content)
            self.assertEqual(generated[0], generated[1])

    def test_legacy_batch_counts_and_prefix(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(
            io.StringIO()
        ):
            g = ForestMapGenerator(
                room_size=8, num_columns=4, min_radius=0.3, max_radius=0.5, output_dir=d
            )
            result = g.generate_all_batch(batch_size=1)
            self.assertEqual(
                {k: len(v) for k, v in result.items()}, {"mesh": 1, "pcd": 1, "png": 1}
            )
            for fmt in ["mesh", "pcd", "png"]:
                files = getattr(g, "generate_" + fmt + "_batch")(
                    prefix="check_" + fmt, batch_size=2
                )
                self.assertEqual(len(files), 2)
                self.assertTrue(
                    all(
                        Path(f).is_file() and Path(f).name.startswith("check_" + fmt)
                        for f in files
                    )
                )

    def test_cli_seed_and_shared_format_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common = [
                sys.executable,
                str(ROOT / "maps/generate.py"),
                "custom",
                "--map-type",
                "forest",
                "--room-size",
                "8",
                "--num-columns",
                "4",
                "--min-radius",
                ".3",
                "--max-radius",
                ".5",
                "--seed",
                "42",
            ]
            for name, formats in [("a", "pcd,png"), ("b", "all")]:
                cmd = common + [
                    "--output-dir",
                    str(root / name),
                    "--output-types",
                    formats,
                    "--batch-size",
                    "2",
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            for i in [1, 2]:
                for extension in ["pcd.pcd", "map.png"]:
                    file = (
                        Path("forest")
                        / ("batch_" + str(i))
                        / ("forest_" + str(i - 1) + "_" + extension)
                    )
                    self.assertEqual(
                        (root / "a" / file).read_bytes(),
                        (root / "b" / file).read_bytes(),
                    )
            self.assertNotEqual(
                (root / "a/forest/batch_1/forest_0_map.png").read_bytes(),
                (root / "a/forest/batch_2/forest_1_map.png").read_bytes(),
            )
            refused = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn(b"Output exists", refused.stderr)

    def test_floor_metric_padding_filter_and_reachability(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reachable = np.zeros((3, 3), dtype=bool)
            reachable[0, :2] = True
            np.savez(
                root / "grid.npz",
                reachable=reachable,
                occupied=~reachable,
                resolution=1.0,
                clearance=np.ones((3, 3)),
            )
            audit = CoverageAudit(root / "grid.npz")
            # Two rows with 8 bytes padding: floor, non-floor, unreachable, second reachable.
            points = np.array(
                [[0.2, 0.2, 0], [1.2, 0.2, 1], [2.2, 2.2, 0], [1.2, 0.2, 0]],
                dtype="<f4",
            )
            msg = NS(
                header=NS(frame_id="world", stamp=NS(to_sec=lambda: 1.0)),
                fields=[
                    NS(name=n, datatype=7, offset=i * 4) for i, n in enumerate("xyz")
                ],
                height=2,
                width=2,
                point_step=12,
                row_step=32,
                is_bigendian=False,
                data=points[:2].tobytes()
                + b"\x00" * 8
                + points[2:].tobytes()
                + b"\x00" * 8,
            )
            audit.cloud(msg)
            result = audit.save(root)
            self.assertEqual(result["observed_cells"], 2)
            self.assertEqual(result["reachable_cells"], 2)
            self.assertEqual(int(audit.observed.sum()), 3)
            msg.header.stamp.to_sec = lambda: 2.0
            msg.header.frame_id = "body"
            with self.assertRaises(ValueError):
                audit.cloud(msg)


if __name__ == "__main__":
    unittest.main()
