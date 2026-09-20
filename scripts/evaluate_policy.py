#!/usr/bin/env python3
"""Run strict Actor evaluation with isolated, owned services and ROS episodes.
Each scene gets its own service log, request latency summary and complete audit.
Requires the ROS environment to be sourced; learning Python may be a separate env.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenes", help="Scene list in the collection schema")
    source.add_argument(
        "--scene-dir",
        help="Directory containing generated kind/scene.json subdirectories",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--learning-python", required=True)
    p.add_argument("--ros-python", default="/usr/bin/python3")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--duration", type=float, default=1200)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    actor = Path(a.actor).resolve()
    if a.scenes:
        scenes = json.loads(Path(a.scenes).read_text())
    else:
        scenes = []
        for path in sorted(Path(a.scene_dir).resolve().glob("*/scene.json")):
            meta = json.loads(path.read_text())
            kind = meta["kind"]
            scene = path.parent
            scenes.append(
                dict(
                    name=kind,
                    map=str(scene / (kind + ".pcd")),
                    config=str(scene / "epic.yaml"),
                    coverage_grid=str(scene / "coverage_grid.npz"),
                    initial_position=meta["initial_position"],
                )
            )
    if not scenes:
        raise ValueError("No evaluation scenes")
    if len({s["name"] for s in scenes}) != len(scenes):
        raise ValueError("Duplicate scene names")
    if any(
        Path(s["name"]).name != s["name"] or s["name"] in [".", ".."] for s in scenes
    ):
        raise ValueError("Invalid scene name")
    (out / "config.json").write_text(
        json.dumps(
            dict(
                arguments=vars(a),
                actor_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),
                scenes=scenes,
            ),
            indent=2,
        )
    )
    children = []
    lock = threading.Lock()
    stop = threading.Event()

    def launch(cmd, log, env=None):
        if stop.is_set():
            raise InterruptedError("Evaluation interrupted")
        child = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with lock:
            children.append(child)
        return child

    def interrupt(*_):
        stop.set()
        with lock:
            for child in children:
                if child.poll() is None:
                    child.terminate()

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def run(scene):
        name = scene["name"]
        episode = out / name
        service = None
        runner = None
        try:
            with tempfile.TemporaryDirectory(prefix="dst-eval-") as tmp, (
                out / (name + ".policy.log")
            ).open("w") as plog, (out / (name + ".runner.log")).open("w") as rlog:
                address = str(Path(tmp) / "policy.sock")
                cmd = [
                    a.learning_python,
                    "-m",
                    "dst_planner.predict",
                    "serve",
                    "--actor",
                    str(actor),
                    "--socket",
                    address,
                    "--device",
                    a.device,
                    "--threads",
                    "2",
                ]
                service = launch(
                    cmd, plog, dict(os.environ, PYTHONPATH=str(root / "learning"))
                )
                (out / (name + ".policy.process.json")).write_text(
                    json.dumps(
                        dict(pid=service.pid, command=cmd, started_at=time.time())
                    )
                )
                deadline = time.monotonic() + 45
                while not Path(address).exists():
                    if service.poll() is not None:
                        raise RuntimeError("Policy service exited before readiness")
                    if stop.is_set() or time.monotonic() > deadline:
                        raise TimeoutError("Policy service readiness timeout")
                    time.sleep(0.1)
                cmd = [
                    a.ros_python,
                    str(root / "scripts/run_episode.py"),
                    "--learning-features",
                    "--policy-mode",
                    "strict",
                    "--policy-socket",
                    address,
                    "--map",
                    scene["map"],
                    "--config",
                    scene["config"],
                    "--init",
                    *map(str, scene["initial_position"]),
                    "--init-yaw",
                    str(scene.get("initial_yaw", 0.0)),
                    "--output",
                    str(episode),
                    "--duration",
                    str(a.duration),
                    "--startup-timeout",
                    "90",
                ]
                if scene.get("coverage_grid"):
                    cmd += ["--coverage-grid", scene["coverage_grid"]]
                runner = launch(cmd, rlog)
                (out / (name + ".runner.process.json")).write_text(
                    json.dumps(
                        dict(pid=runner.pid, command=cmd, started_at=time.time())
                    )
                )
                code = runner.wait()
                service.terminate()
                service.wait(timeout=10)
                if service.returncode != 0:
                    raise RuntimeError("Policy service did not shut down cleanly")
            check = subprocess.run(
                [
                    a.ros_python,
                    str(root / "scripts/validate_episode.py"),
                    str(episode),
                    "--require-complete",
                ],
                capture_output=True,
                text=True,
                cwd=root,
            )
            (episode / "validation.json").write_text(
                check.stdout or json.dumps(dict(valid=False, error=check.stderr))
            )
            summary = json.loads((episode / "summary.json").read_text())
            counts = Counter()
            results = Counter()
            for line in (episode / "events.jsonl").open():
                e = json.loads(line)
                if e["event"] == "execution":
                    counts[e["kind"]] += 1
            for line in (episode / "decisions.jsonl").open():
                results[json.loads(line)["result"]] += 1
            calls = [
                json.loads(line)
                for line in (out / (name + ".policy.log")).read_text().splitlines()
                if line.startswith("{")
            ]
            calls = [x for x in calls if x.get("event") == "inference"]
            latency = sorted(x["inference_ms"] for x in calls if x.get("ok"))
            coverage = summary.get("coverage", {})
            successful = (
                code == 0
                and check.returncode == 0
                and counts["policy_success"] > 0
                and counts["policy_fallback"] == 0
                and results["lkh"] == 0
                and all(x.get("ok") for x in calls)
                and coverage.get("observed_fraction", 0) >= 0.95
                and not coverage.get("near_obstacle_odometry_samples", 1)
                and not coverage.get("outside_map_odometry_samples", 1)
            )
            report = dict(
                name=name,
                passed=successful,
                summary=summary,
                execution_counts=dict(counts),
                decision_kinds=dict(results),
                inference_requests=len(calls),
                inference_errors=sum(not x.get("ok") for x in calls),
                inference_ms_p50=latency[len(latency) // 2] if latency else None,
                inference_ms_p95=latency[
                    min(len(latency) - 1, int(0.95 * len(latency)))
                ]
                if latency
                else None,
            )
            print(json.dumps(report), flush=True)
            return report
        except Exception as exc:
            return dict(name=name, passed=False, error=str(exc))
        finally:
            for child in [runner, service]:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        reports = list(pool.map(run, scenes))
    result = dict(
        finished=True, passed=all(x["passed"] for x in reports), scenes=reports
    )
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
