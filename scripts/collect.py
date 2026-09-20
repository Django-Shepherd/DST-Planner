#!/usr/bin/env python3
"""Collect EPIC expert episodes from a scene list or a repeated single map."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
CLEANUP_TIMEOUT = 60


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenes", help="JSON scene list for parallel collection")
    source.add_argument("--map", help="Point cloud for sequential repeated episodes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, help="Scene-list workers (default: 3)")
    parser.add_argument("--count", type=int, default=2, help="Episode count for --map")
    parser.add_argument(
        "--duration",
        type=float,
        help="Seconds per episode (scene list: 1200; map: 180)",
    )
    parser.add_argument("--startup-timeout", type=float, default=90)
    parser.add_argument("--config", help="EPIC configuration for --map")
    parser.add_argument("--coverage-grid", help="Coverage grid for --map")
    parser.add_argument("--init", nargs=3, type=float, default=[5.0, 0.0, 2.0])
    parser.add_argument("--init-yaw", type=float, default=0.0)
    parser.add_argument("--cpu-render", action="store_true")
    parser.add_argument(
        "--learning-features", action="store_true", help="Always enabled for --scenes"
    )
    parser.add_argument(
        "--require-complete", action="store_true", help="Always enabled for --scenes"
    )
    args = parser.parse_args(argv)
    args.workers = (
        args.workers if args.workers is not None else (3 if args.scenes else 1)
    )
    args.duration = (
        args.duration if args.duration is not None else (1200 if args.scenes else 180)
    )
    if args.workers < 1 or args.count < 1:
        parser.error("workers and count must be positive")
    if any(
        not math.isfinite(x) or x <= 0 for x in (args.duration, args.startup_timeout)
    ):
        parser.error("duration and startup-timeout must be positive and finite")
    if args.map and args.workers != 1:
        parser.error(
            "--map collects sequentially; use --scenes for parallel collection"
        )
    try:
        scenes = load_scenes(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return args, scenes


def load_scenes(args):
    if args.scenes:
        scenes = json.loads(Path(args.scenes).read_text())
    else:
        scenes = [
            dict(
                name=f"episode-{i:04d}",
                map=args.map,
                config=args.config,
                coverage_grid=args.coverage_grid,
                initial_position=args.init,
                initial_yaw=args.init_yaw,
            )
            for i in range(args.count)
        ]
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a nonempty JSON list")
    for scene in scenes:
        if not isinstance(scene, dict) or any(
            k not in scene for k in ("name", "map", "initial_position")
        ):
            raise ValueError("Each scene requires name, map and initial_position")
        name = scene["name"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise ValueError("Invalid scene name")
        if not isinstance(scene["map"], str) or not scene["map"]:
            raise ValueError("Each scene requires a nonempty map path")
        if args.scenes and not scene.get("config"):
            raise ValueError("Each scene-list entry requires an EPIC config path")
        position = scene["initial_position"]
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("initial_position must contain three coordinates")
        if not all(
            math.isfinite(float(x)) for x in [*position, scene.get("initial_yaw", 0.0)]
        ):
            raise ValueError("Start coordinates and yaw must be finite")
    if len({scene["name"] for scene in scenes}) != len(scenes):
        raise ValueError("Duplicate scene names")
    return scenes


class Collector:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.stopped = threading.Event()
        self.children = set()
        self.lock = threading.RLock()
        self.stop_signal = None

    def interrupt(self, signum, _frame):
        self.stop_signal = signum
        self.stopped.set()
        with self.lock:
            for child in self.children:
                if child.poll() is None:
                    try:
                        child.terminate()
                    except ProcessLookupError:
                        pass

    def command(self, scene, episode):
        args = self.args
        command = [
            sys.executable,
            str(ROOT / "scripts/run_episode.py"),
            "--output",
            str(episode),
            "--map",
            scene["map"],
            "--duration",
            str(args.duration),
            "--startup-timeout",
            str(args.startup_timeout),
            "--init",
            *map(str, scene["initial_position"]),
            "--init-yaw",
            str(scene.get("initial_yaw", 0.0)),
        ]
        for option, key in [
            ("--config", "config"),
            ("--coverage-grid", "coverage_grid"),
        ]:
            if scene.get(key):
                command.extend([option, scene[key]])
        if args.scenes or args.learning_features:
            command.append("--learning-features")
        if args.cpu_render:
            command.append("--cpu-render")
        return command

    def wait(self, child):
        while True:
            try:
                return child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if self.stopped.is_set():
                    # Let the runner clean up its ROS processes before forcing exit.
                    try:
                        return child.wait(timeout=CLEANUP_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        return child.wait()

    def run(self, scene):
        name = scene["name"]
        episode = self.output / name
        record = dict(name=name, output=str(episode), complete=False)
        child = None
        try:
            command = self.command(scene, episode)
            with (self.output / (name + ".log")).open("w") as log:
                with self.lock:
                    if self.stopped.is_set() or (self.output / "STOP").exists():
                        record["reason"] = "dispatcher_stopped"
                        return record
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.children.add(child)
                (self.output / (name + ".process.json")).write_text(
                    json.dumps(
                        dict(pid=child.pid, command=command, started_at=time.time())
                    )
                )
                code = self.wait(child)
            record["returncode"] = code
            summary_file = episode / "summary.json"
            summary = (
                json.loads(summary_file.read_text()) if summary_file.exists() else {}
            )
            check_command = [
                sys.executable,
                str(ROOT / "scripts/validate_episode.py"),
                str(episode),
            ]
            if self.args.scenes or self.args.require_complete:
                check_command.append("--require-complete")
            check = subprocess.run(check_command, capture_output=True, text=True)
            validation = (
                json.loads(check.stdout)
                if check.stdout
                else dict(valid=False, error=check.stderr)
            )
            # A startup failure may occur before the runner creates its directory.
            episode.mkdir(exist_ok=True)
            (episode / "validation.json").write_text(json.dumps(validation, indent=2))
            record.update(
                complete=code == 0 and check.returncode == 0,
                reason=summary.get("reason")
                or ("runner_failed" if code else "validation_failed"),
                distance_m=summary.get("distance_m"),
                validation=validation,
            )
        except Exception as exc:
            record.update(reason="collection_error", error=str(exc))
        finally:
            if child is not None:
                if child.poll() is None:
                    try:
                        child.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=CLEANUP_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                with self.lock:
                    self.children.discard(child)
            if not self.args.scenes:
                # Keep single-map manifest fields compatible with existing runs.
                record.update(
                    episode=str(episode),
                    exit_code=record.get("returncode"),
                    accepted=record["complete"],
                )
        return record

    def collect(self, scenes):
        records = []

        def save(record):
            records.append(record)
            (self.output / "manifest.json").write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)

        if self.args.scenes:
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                futures = [pool.submit(self.run, scene) for scene in scenes]
                for future in as_completed(futures):
                    save(future.result())
        else:
            for scene in scenes:
                record = self.run(scene)
                save(record)
                if not record["complete"]:
                    break
        (self.output / "completion.json").write_text(
            json.dumps(
                dict(
                    finished=True,
                    total=len(scenes),
                    complete=sum(x["complete"] for x in records),
                )
            )
        )
        if self.stop_signal is not None:
            return 128 + self.stop_signal
        return (
            0
            if len(records) == len(scenes) and all(x["complete"] for x in records)
            else 1
        )


def main(argv=None):
    args, scenes = parse_args(argv)
    collector = Collector(args)
    previous = {
        sig: signal.signal(sig, collector.interrupt)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return collector.collect(scenes)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
