"""Check scene dispatch and collection manifests without ROS or map libraries."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "map_generation", ROOT / "maps/generate.py"
)
generate_scenes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_scenes)


class SceneGenerationCLITest(unittest.TestCase):
    def test_custom_defaults_and_geometry_options(self):
        defaults = {
            "forest": dict(
                room_size=80, num_columns=240, min_radius=0.5, max_radius=2.0
            ),
            "partition": dict(
                room_size=80, num_columns=240, num_cubes=81, half_width=0.5
            ),
            "dungeon": dict(room_size=80, max_rooms=30, corridor_width=1),
            "hybrid": dict(room_size=320, num_regions=16, num_dense_regions=8),
        }
        with tempfile.TemporaryDirectory() as directory:
            for kind, expected in defaults.items():
                with self.subTest(kind=kind), mock.patch.object(
                    generate_scenes, "create_generator"
                ) as create, contextlib.redirect_stdout(io.StringIO()):
                    generate_scenes.main(
                        [
                            "custom",
                            "--map-type",
                            kind,
                            "--output-dir",
                            directory,
                        ]
                    )
                    self.assertEqual(create.call_args.args, (kind,))
                    parameters = create.call_args.kwargs
                    self.assertEqual(
                        parameters["output_dir"], str(Path(directory) / kind)
                    )
                    self.assertEqual(parameters["wall_height"], 3.0)
                    self.assertEqual(parameters["floor_thickness"], 0.1)
                    for name, value in expected.items():
                        self.assertEqual(parameters[name], value)
                    create.return_value.generate_batch.assert_called_once_with(
                        ["mesh", "pcd", "png"],
                        prefix=None,
                        batch_size=1,
                        seed=None,
                        map_name=kind,
                        visualize=False,
                    )

    def test_custom_export_settings_reach_shared_batch_path(self):
        with mock.patch.object(
            generate_scenes, "create_generator"
        ) as create, contextlib.redirect_stdout(io.StringIO()):
            generate_scenes.main(
                [
                    "custom",
                    "--map-type",
                    "hybrid",
                    "--room-size",
                    "40",
                    "--seed",
                    "42",
                    "--output-types",
                    "png,pcd",
                    "--prefix",
                    "sample",
                    "--batch-size",
                    "2",
                    "--visualize",
                ]
            )
        self.assertEqual(create.call_args.kwargs["room_size"], 40)
        create.return_value.generate_batch.assert_called_once_with(
            ["png", "pcd"],
            prefix="sample",
            batch_size=2,
            seed=42,
            map_name="hybrid",
            visualize=True,
        )

    def test_invalid_generation_arguments_do_not_start_generation(self):
        cases = [
            ["custom", "--batch-size", "0"],
            ["custom", "--output-types", "pcd,pcd"],
            ["custom", "--output-types", "invalid"],
            ["custom", "--prefix", "../escape"],
            ["test", "--output", "unused", "--size", "0"],
            ["train", "--output", "unused", "--maps-per-type", "0"],
        ]
        for options in cases:
            with self.subTest(options=options), mock.patch.object(
                generate_scenes, "create_generator"
            ) as create, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    generate_scenes.main(options)
                self.assertEqual(raised.exception.code, 2)
                create.assert_not_called()

    def test_test_scene_defaults_and_overrides(self):
        cases = [([], 40, 20260916), (["--size", "20", "--seed", "45"], 20, 45)]
        for options, size, seed in cases:
            with self.subTest(
                options=options
            ), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "scenes"
                with mock.patch.object(generate_scenes, "generate") as generate:
                    generate_scenes.main(["test", "--output", str(output), *options])
                self.assertTrue(output.is_dir())
                self.assertEqual(
                    generate.call_args_list,
                    [
                        mock.call(output.resolve(), "forest", seed, size),
                        mock.call(output.resolve(), "partition", seed + 1, size),
                        mock.call(output.resolve(), "dungeon", seed + 2, size),
                    ],
                )

    def test_training_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scenes"
            with mock.patch.object(
                generate_scenes, "generate_training_scenes"
            ) as generate:
                generate_scenes.main(["train", "--output", str(output)])
            generate.assert_called_once_with(output.resolve(), 10, 917000)

    def test_training_manifest_order_starts_and_yaws(self):
        initial = [3.125, 3.875, 1.5]
        interior = [29.875, 30.125, 1.5]

        def write_scene(parent, kind, seed, size):
            scene = parent / kind
            scene.mkdir()
            (scene / "scene.json").write_text(json.dumps({"initial_position": initial}))
            for name in (f"{kind}.pcd", "epic.yaml", "coverage_grid.npz"):
                (scene / name).touch()

        def fixture_starts(scene, start):
            self.assertTrue((scene / "coverage_grid.npz").is_file())
            self.assertEqual(start, initial)
            return [start, interior]

        with tempfile.TemporaryDirectory() as directory:
            output = (Path(directory) / "scenes").resolve()
            with mock.patch.object(
                generate_scenes, "generate", side_effect=write_scene
            ) as generate, mock.patch.object(
                generate_scenes, "_training_starts", side_effect=fixture_starts
            ), contextlib.redirect_stdout(io.StringIO()):
                generate_scenes.main(
                    [
                        "train",
                        "--output",
                        str(output),
                        "--maps-per-type",
                        "2",
                        "--seed",
                        "45",
                    ]
                )
            self.assertEqual(
                generate.call_args_list,
                [
                    mock.call(output / "map_00", "forest", 45, 40),
                    mock.call(output / "map_00", "partition", 46, 40),
                    mock.call(output / "map_01", "forest", 47, 40),
                    mock.call(output / "map_01", "partition", 48, 40),
                ],
            )
            scenes = json.loads((output / "scenes.json").read_text())
            self.assertEqual(
                [scene["name"] for scene in scenes],
                [
                    "forest_00_start0",
                    "forest_00_start1",
                    "partition_00_start0",
                    "partition_00_start1",
                    "forest_01_start0",
                    "forest_01_start1",
                    "partition_01_start0",
                    "partition_01_start1",
                ],
            )
            for index, scene in enumerate(scenes):
                self.assertEqual(
                    set(scene),
                    {
                        "name",
                        "map",
                        "config",
                        "coverage_grid",
                        "initial_position",
                        "initial_yaw",
                    },
                )
                for field in ("map", "config", "coverage_grid"):
                    self.assertTrue(Path(scene[field]).is_file())
                self.assertEqual(
                    scene["initial_position"], interior if index % 2 else initial
                )
                self.assertEqual(
                    scene["initial_yaw"],
                    math.atan2(20 - interior[1], 20 - interior[0])
                    if index % 2
                    else 0.0,
                )
            self.assertEqual(
                json.loads((output / "generation_complete.json").read_text()),
                {
                    "complete": True,
                    "maps": 4,
                    "episodes": 8,
                },
            )

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(generate_scenes, "generate") as generate:
                with self.assertRaises(FileExistsError):
                    generate_scenes.main(["test", "--output", directory])
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
