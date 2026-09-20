"""Check batch failure reporting without requiring a running ROS environment."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CollectionDispatchTest(unittest.TestCase):
    def stub_workspace(self, root):
        scripts = root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(ROOT / "scripts/collect.py", scripts / "collect.py")
        (scripts / "run_episode.py").write_text("""import json
from pathlib import Path
import sys

args = sys.argv[1:]
output = Path(args[args.index('--output') + 1])
output.mkdir()
(output / 'runner_args.json').write_text(json.dumps(args))
(output / 'summary.json').write_text(json.dumps({'reason': 'epic_finish', 'distance_m': 12}))
""")
        (scripts / "validate_episode.py").write_text("""import json
from pathlib import Path
import sys

(Path(sys.argv[1]) / 'validator_args.json').write_text(json.dumps(sys.argv[2:]))
print(json.dumps({'valid': True}))
""")
        return scripts / "collect.py"

    def run_batch(self, root, scenes, *args):
        manifest = root / "scenes.json"
        manifest.write_text(json.dumps(scenes))
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/collect.py"),
                "--scenes",
                str(manifest),
                "--output",
                str(root / "runs"),
                "--workers",
                "1",
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_startup_failures_are_reported_for_every_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = [
                dict(
                    name=name,
                    map=str(root / "missing.pcd"),
                    config=str(root / "missing.yaml"),
                    initial_position=[0, 0, 1],
                )
                for name in ("first", "second")
            ]
            result = self.run_batch(root, scenes)
            self.assertEqual(result.returncode, 1, result.stderr)
            records = json.loads((root / "runs/manifest.json").read_text())
            self.assertEqual({x["name"] for x in records}, {"first", "second"})
            self.assertTrue(
                all(
                    not x["complete"] and x["reason"] == "runner_failed"
                    for x in records
                )
            )
            for name in ("first", "second"):
                self.assertTrue((root / "runs" / name / "validation.json").is_file())
                self.assertIn(
                    "Map does not exist", (root / "runs" / (name + ".log")).read_text()
                )
            completion = json.loads((root / "runs/completion.json").read_text())
            self.assertEqual(completion, dict(finished=True, total=2, complete=0))

    def test_invalid_input_does_not_create_run_directory(self):
        for scenes, args in [
            ([], []),
            (
                [
                    dict(
                        name="../escape",
                        map="x",
                        config="x",
                        initial_position=[0, 0, 1],
                    )
                ],
                [],
            ),
            ([], ["--workers", "0"]),
        ]:
            with self.subTest(
                scenes=scenes, args=args
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_batch(root, scenes, *args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((root / "runs").exists())

    def test_single_map_defaults_and_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.stub_workspace(root)
            command = [
                sys.executable,
                str(script),
                "--map",
                "garage.pcd",
                "--output",
                str(root / "runs"),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = json.loads((root / "runs/manifest.json").read_text())
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertTrue(record["accepted"])
                self.assertEqual(record["exit_code"], 0)
                episode = Path(record["episode"])
                args = json.loads((episode / "runner_args.json").read_text())
                self.assertEqual(float(args[args.index("--duration") + 1]), 180)
                position = args[args.index("--init") + 1 : args.index("--init") + 4]
                self.assertEqual(list(map(float, position)), [5, 0, 2])
                self.assertNotIn("--learning-features", args)
                self.assertEqual(
                    json.loads((episode / "validator_args.json").read_text()), []
                )

            # A failed episode must prevent further single-map launches.
            (script.parent / "run_episode.py").write_text("raise SystemExit(1)\n")
            result = subprocess.run(
                command[:-1] + [str(root / "failed"), "--count", "3"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            records = json.loads((root / "failed/manifest.json").read_text())
            self.assertEqual(len(records), 1)
            self.assertFalse(records[0]["accepted"])
            self.assertFalse((root / "failed/episode-0001.log").exists())

    def test_scene_and_single_map_option_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.stub_workspace(root)
            scene = dict(
                name="forest",
                map="forest.pcd",
                config="forest.yaml",
                coverage_grid="coverage.npz",
                initial_position=[3, 4, 1.5],
                initial_yaw=0.7,
            )
            (root / "scenes.json").write_text(json.dumps([scene]))
            modes = [
                (["--scenes", str(root / "scenes.json")], 1200, "forest"),
                (
                    [
                        "--map",
                        scene["map"],
                        "--count",
                        "1",
                        "--config",
                        scene["config"],
                        "--coverage-grid",
                        scene["coverage_grid"],
                        "--init",
                        "3",
                        "4",
                        "1.5",
                        "--init-yaw",
                        "0.7",
                        "--duration",
                        "75",
                        "--learning-features",
                        "--require-complete",
                    ],
                    75,
                    "episode-0000",
                ),
            ]
            for index, (mode, duration, name) in enumerate(modes):
                output = root / f"mode-{index}"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        *mode,
                        "--output",
                        str(output),
                        "--cpu-render",
                        "--startup-timeout",
                        "45",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                episode = output / name
                args = json.loads((episode / "runner_args.json").read_text())
                for option, expected in [
                    ("--map", scene["map"]),
                    ("--config", scene["config"]),
                    ("--coverage-grid", scene["coverage_grid"]),
                ]:
                    self.assertEqual(args[args.index(option) + 1], expected)
                for option, expected in [
                    ("--duration", duration),
                    ("--startup-timeout", 45),
                    ("--init-yaw", 0.7),
                ]:
                    self.assertEqual(float(args[args.index(option) + 1]), expected)
                position = args[args.index("--init") + 1 : args.index("--init") + 4]
                self.assertEqual(list(map(float, position)), scene["initial_position"])
                self.assertIn("--learning-features", args)
                self.assertIn("--cpu-render", args)
                self.assertIn(
                    "--require-complete",
                    json.loads((episode / "validator_args.json").read_text()),
                )

    def test_interrupt_stops_runner_and_skips_queued_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.stub_workspace(root)
            (script.parent / "run_episode.py").write_text("""import json
import os
from pathlib import Path
import signal
import sys
import time

output = Path(sys.argv[sys.argv.index('--output') + 1])
output.mkdir()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(output / 'ready.json').write_text(json.dumps({'pid': os.getpid()}))
while True:
    time.sleep(1)
""")
            scenes = [
                dict(
                    name=name,
                    map="forest.pcd",
                    config="forest.yaml",
                    initial_position=[3, 4, 1.5],
                )
                for name in ("active", "queued")
            ]
            (root / "scenes.json").write_text(json.dumps(scenes))
            # Shorten only the cleanup grace period so an unresponsive runner is testable.
            bootstrap = (
                "import collect; collect.CLEANUP_TIMEOUT = 0.1; "
                "raise SystemExit(collect.main())"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    "--scenes",
                    str(root / "scenes.json"),
                    "--output",
                    str(root / "runs"),
                    "--workers",
                    "1",
                ],
                cwd=script.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            child_pid = None
            try:
                ready = root / "runs/active/ready.json"
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(process.communicate()[1])
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), "Runner did not start")
                child_pid = json.loads(ready.read_text())["pid"]
                process.terminate()
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 128 + signal.SIGTERM, stderr)
                records = {
                    record["name"]: record
                    for record in json.loads((root / "runs/manifest.json").read_text())
                }
                self.assertEqual(records["queued"]["reason"], "dispatcher_stopped")
                self.assertFalse(records["active"]["complete"])
                self.assertFalse((root / "runs/queued/ready.json").exists())
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
