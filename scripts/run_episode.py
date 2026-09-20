#!/usr/bin/env python3
"""Run one EPIC episode in an owned ROS master, preserving all other ROS jobs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import threading
import time
import xmlrpc.client


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--config")
    p.add_argument(
        "--coverage-grid", help="Optional generated-scene reachable-floor audit grid"
    )
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--startup-timeout", type=float, default=45)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--learning-features", action="store_true")
    p.add_argument(
        "--policy-mode",
        choices=["epic", "shadow", "strict", "fallback_epic"],
        default="epic",
    )
    p.add_argument("--policy-socket", default="")
    p.add_argument(
        "--cpu-render",
        action="store_true",
        help="Explicit fallback; official default is GPU",
    )
    p.add_argument("--init", nargs=3, type=float, default=[5.0, 0.0, 2.0])
    p.add_argument("--init-yaw", type=float, default=0.0, help="Initial yaw in radians")
    a = p.parse_args()
    out = Path(a.output).resolve()
    scene = Path(a.map).resolve()
    if not scene.is_file():
        p.error("Map does not exist")
    if a.duration <= 0 or a.startup_timeout <= 0:
        p.error("Timeouts must be positive")
    out.mkdir(parents=True, exist_ok=False)
    (out / "lkh").mkdir()
    (out / "roslog").mkdir()
    root = Path(__file__).resolve().parents[1]
    config = (
        Path(a.config).resolve()
        if a.config
        else root / "src/global_planner/exploration_manager/config/garage.yaml"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(
        ROS_MASTER_URI=f"http://127.0.0.1:{port}",
        ROS_IP="127.0.0.1",
        ROS_HOSTNAME="127.0.0.1",
        ROS_LOG_DIR=str(out / "roslog"),
    )
    os.environ["OMP_NUM_THREADS"] = "4"
    # Headless OpenGL swap synchronization otherwise throttles this host to ~1 Hz.
    os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")
    meta = {
        "schema_version": 1,
        "episode_id": out.name,
        "baseline": a.baseline,
        "map": str(scene),
        "map_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "config": str(config),
        "config_text": config.read_text(),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for base in ["src/global_planner/exploration_manager", "scripts"]
            for f in sorted((root / base).rglob("*"))
            if f.is_file() and "__pycache__" not in f.parts
        },
        "diff": subprocess.check_output(["git", "-C", str(root), "diff"], text=True),
        "policy_mode": a.policy_mode,
        "policy_socket": a.policy_socket,
        "learning_features": a.learning_features or a.policy_mode != "epic",
        "initial_position": a.init,
        "initial_yaw": a.init_yaw,
        "renderer": "cpu" if a.cpu_render else "gpu",
        "renderer_sync_to_vblank": os.environ["__GL_SYNC_TO_VBLANK"],
        "sensor_range": 60.0,
        "lidar_pitch": 40.0,
        "sensor_resolution": 0.2,
        "duration_limit": a.duration,
        "ros_master_uri": os.environ["ROS_MASTER_URI"],
        "started_at": time.time(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    coverage = None
    if a.coverage_grid:
        from coverage_audit import CoverageAudit

        coverage = CoverageAudit(a.coverage_grid)
        meta["coverage_grid"] = str(Path(a.coverage_grid).resolve())
        meta["coverage_grid_sha256"] = hashlib.sha256(
            Path(a.coverage_grid).read_bytes()
        ).hexdigest()
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    children = []
    streams = []
    stop = threading.Event()
    failure = []
    owned_descendants = {}

    def remember_children():
        # roslaunch starts nodes in their own process groups; retain their birth IDs.
        processes = {}
        for proc in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = proc.read_text().rsplit(")", 1)[1].split()
                processes[int(proc.parent.name)] = (int(fields[1]), fields[19])
            except (OSError, ValueError, IndexError):
                continue
        parents = {p.pid for p in children if p.poll() is None}
        parents.update(
            pid
            for pid, birth in owned_descendants.items()
            if pid in processes and processes[pid][1] == birth
        )
        while True:
            found = {
                pid
                for pid, (ppid, _) in processes.items()
                if ppid in parents and pid not in parents
            }
            if not found:
                break
            parents.update(found)
        for pid in parents:
            if pid in processes:
                owned_descendants[pid] = processes[pid][1]

    def alive_owned():
        alive = []
        for pid, birth in owned_descendants.items():
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[19] == birth and fields[0] != "Z":
                    alive.append(pid)
            except (OSError, IndexError):
                pass
        return alive

    events = queue.Queue(maxsize=20000)
    state = {
        "fsm": None,
        "last_odom": None,
        "distance": 0.0,
        "odometry_count": 0,
        "trajectory_count": 0,
    }

    def emit(kind, data):
        try:
            events.put_nowait({"event": kind, "wall_time": time.time(), **data})
        except queue.Full:
            failure.append("telemetry_queue_overflow")
            stop.set()

    def write_events():
        try:
            with (out / "events.jsonl").open("x") as f:
                while True:
                    item = events.get()
                    if item is None:
                        break
                    f.write(json.dumps(item, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            failure.append("telemetry_write_error: " + str(exc))
            stop.set()

    writer = threading.Thread(target=write_events)
    writer.start()

    def spawn(cmd, name):
        log = (out / (name + ".log")).open("w")
        streams.append(log)
        child = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        children.append(child)
        return child

    reason = "startup_failure"
    start = None
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        master = spawn(["roscore", "-p", str(port)], "master")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if (
                    xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"]).getPid(
                        "/dst_runner"
                    )[0]
                    == 1
                ):
                    break
            except OSError:
                pass
            if master.poll() is not None:
                raise RuntimeError("Owned ROS master exited")
            time.sleep(0.2)
        else:
            raise RuntimeError("ROS master startup timeout")
        import rospy
        from nav_msgs.msg import Odometry, Path as RosPath
        from geometry_msgs.msg import PoseStamped
        from visualization_msgs.msg import Marker
        from traj_utils.msg import PolyTraj
        from std_msgs.msg import String

        rospy.init_node("dst_episode_recorder", disable_signals=True)

        def odom(msg):
            pos = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
            if coverage:
                coverage.odometry(pos)
            if state["last_odom"] is not None:
                state["distance"] += math.sqrt(
                    sum((u - v) ** 2 for u, v in zip(pos, state["last_odom"]))
                )
            state["last_odom"] = pos
            state["odometry_count"] += 1
            emit(
                "odometry",
                {
                    "stamp": msg.header.stamp.to_sec(),
                    "position": pos,
                    "orientation": [
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                    ],
                    "velocity": [
                        msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z,
                    ],
                },
            )

        def fsm(msg):
            if msg.text != state["fsm"]:
                emit("fsm", {"state": msg.text, "stamp": msg.header.stamp.to_sec()})
            state["fsm"] = msg.text

        def trajectory(msg):
            state["trajectory_count"] += 1
            emit(
                "trajectory",
                {
                    "drone_id": msg.drone_id,
                    "traj_id": msg.traj_id,
                    "start_time": msg.start_time.to_sec(),
                    "order": msg.order,
                    "coef_x": list(msg.coef_x),
                    "coef_y": list(msg.coef_y),
                    "coef_z": list(msg.coef_z),
                    "duration": list(msg.duration),
                },
            )

        def execution(msg):
            try:
                emit("execution", json.loads(msg.data))
            except Exception as exc:
                failure.append("execution_parse: " + str(exc))
                stop.set()

        def error(msg):
            failure.append(msg.data)
            stop.set()

        subscriptions = [
            rospy.Subscriber(
                "/quad_0/lidar_slam/odom", Odometry, odom, queue_size=1000
            ),
            rospy.Subscriber("/planning/state", Marker, fsm, queue_size=100),
            rospy.Subscriber(
                "/planning/trajectory", PolyTraj, trajectory, queue_size=100
            ),
            rospy.Subscriber("/dst_collection/error", String, error, queue_size=10),
            rospy.Subscriber(
                "/dst_collection/execution", String, execution, queue_size=1000
            ),
        ]
        if coverage:
            from sensor_msgs.msg import PointCloud2

            def cloud(msg):
                try:
                    coverage.cloud(msg)
                except Exception as exc:
                    failure.append("coverage_audit: " + str(exc))
                    stop.set()

            subscriptions.append(
                rospy.Subscriber(
                    "/quad0_pcl_render_node/cloud",
                    PointCloud2,
                    cloud,
                    queue_size=1,
                    buff_size=8 * 1024 * 1024,
                )
            )
        trigger = rospy.Publisher(
            "/waypoint_generator/waypoints", RosPath, queue_size=1, latch=True
        )
        cmd = [
            "roslaunch",
            "epic_planner",
            "collect.launch",
            "map:=" + str(scene),
            "output:=" + str(out),
            "enabled:=" + str(not a.baseline).lower(),
            "config:=" + str(config),
            "learning_features:=" + str(a.learning_features).lower(),
            "policy_mode:=" + a.policy_mode,
            "policy_socket:=" + a.policy_socket,
            "use_gpu:=" + str(not a.cpu_render).lower(),
            *[f"init_{axis}:={value}" for axis, value in zip("xyz", a.init)],
            f"init_yaw:={a.init_yaw}",
        ]
        launch = spawn(cmd, "launch")
        last_inventory = 0.0
        deadline = time.monotonic() + a.startup_timeout
        while time.monotonic() < deadline and not stop.is_set():
            if time.monotonic() - last_inventory > 1:
                remember_children()
                last_inventory = time.monotonic()
            if launch.poll() is not None:
                raise RuntimeError("roslaunch exited during startup")
            if (
                state["fsm"] == "WAIT_TRIGGER"
                and state["odometry_count"] >= 20
                and trigger.get_num_connections()
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Planner readiness timeout: " + str(state["fsm"]))
        time.sleep(2)
        msg = RosPath()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.z = a.init[2]
        pose.pose.orientation.w = 1.0
        msg.poses = [pose]
        trigger.publish(msg)
        start = time.monotonic()
        emit("trigger", {"stamp": msg.header.stamp.to_sec()})
        reason = "time_limit"
        while time.monotonic() - start < a.duration:
            if time.monotonic() - last_inventory > 1:
                remember_children()
                last_inventory = time.monotonic()
            if stop.is_set():
                reason = "collection_error" if failure else "interrupted"
                break
            if launch.poll() is not None:
                reason = "node_failure"
                break
            if state["fsm"] == "FINISH":
                reason = "epic_finish"
                break
            time.sleep(0.1)
    except Exception as exc:
        failure.append(str(exc))
    finally:
        # Signal only process groups created by this invocation; no global pkill.
        remember_children()
        forced = []
        for child in reversed(children):
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=18)
            except subprocess.TimeoutExpired:
                forced.append(child.pid)
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)
        survivors = alive_owned()
        if survivors:
            for pid in survivors:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 3
            while alive_owned() and time.monotonic() < deadline:
                time.sleep(0.1)
            for pid in alive_owned():
                try:
                    os.kill(pid, signal.SIGKILL)
                    forced.append(pid)
                except ProcessLookupError:
                    pass
        if forced:
            failure.append("forced_shutdown: " + str(forced))
        if "rospy" in locals():
            rospy.signal_shutdown("Episode complete")
        launch_log = out / "launch.log"
        if launch_log.exists():
            log_text = launch_log.read_text(errors="replace")
            if any(
                t in log_text
                for t in [
                    "terminate called",
                    "Aborted (Signal",
                    "Segmentation fault",
                    "process has died",
                ]
            ):
                failure.append("node_crash: inspect launch.log")
        if not a.baseline:
            status_path = out / "decisions.jsonl.status.json"
            try:
                status = json.loads(status_path.read_text())
                if status["failed"] or status["written"] != status["enqueued"]:
                    failure.append("incomplete_decision_writer")
            except Exception as exc:
                failure.append("missing_or_invalid_decision_status: " + str(exc))
        emit(
            "episode_end",
            {
                "reason": reason,
                "terminated": reason == "epic_finish",
                "truncated": reason == "time_limit",
            },
        )
        if writer.is_alive():
            events.put(None, timeout=5)
            writer.join(timeout=10)
        if writer.is_alive():
            failure.append("telemetry_writer_not_stopped")
        for f in streams:
            f.close()
        summary = {
            "reason": reason,
            "terminated": reason == "epic_finish",
            "truncated": reason == "time_limit",
            "errors": failure,
            "distance_m": state["distance"],
            "odometry_count": state["odometry_count"],
            "trajectory_count": state["trajectory_count"],
            "last_state": state["fsm"],
            "elapsed_after_trigger": time.monotonic() - start if start else None,
            "owned_pids": [p.pid for p in children],
            "surviving_owned_pids": alive_owned(),
        }
        if coverage:
            try:
                summary["coverage"] = coverage.save(out)
            except Exception as exc:
                failure.append("coverage_save: " + str(exc))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    return (
        0
        if not failure
        and reason in ["epic_finish", "time_limit"]
        and state["trajectory_count"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
