#!/usr/bin/env python3
"""Compile against the existing catkin build and test the actual C++ recorder."""

import json
import os
import re
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xmlrpc.client

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
from validate_episode import rows, validate_decision

with tempfile.TemporaryDirectory(prefix="dst-recorder-probe-") as tmp:
    tmp = Path(tmp)
    binary = tmp / "probe"
    flags = (
        root / "build/epic_planner/CMakeFiles/epic_planner.dir/flags.make"
    ).read_text()
    includes = next(
        line.split("=", 1)[1]
        for line in flags.splitlines()
        if line.startswith("CXX_INCLUDES =")
    )
    link = shlex.split(
        (root / "build/epic_planner/CMakeFiles/epic_planner.dir/link.txt").read_text()
    )
    output_index = link.index("-o") + 1
    deps = [
        v
        for i, v in enumerate(link)
        if i != output_index
        and not v.startswith("-Wl,-soname,")
        and (
            v.startswith(("-l", "-L", "-Wl,")) or re.search(r"\.(so(\.[0-9.]+)?|a)$", v)
        )
    ]
    subprocess.run(
        [
            "g++",
            "-std=c++14",
            "-O1",
            *shlex.split(includes),
            str(root / "tests/recorder_probe.cpp"),
            "-L" + str(root / "devel/lib"),
            "-Wl,-rpath," + str(root / "devel/lib"),
            "-lepic_planner",
            *deps,
            "-lpcl_common",
            "-lpcl_octree",
            "-o",
            str(binary),
        ],
        check=True,
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ,
        ROS_MASTER_URI=f"http://127.0.0.1:{port}",
        ROS_IP="127.0.0.1",
        ROS_HOSTNAME="127.0.0.1",
        ROS_LOG_DIR=str(tmp / "roslog"),
    )
    with (tmp / "master.log").open("w") as log:
        master = subprocess.Popen(
            ["roscore", "-p", str(port)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    if (
                        xmlrpc.client.ServerProxy(env["ROS_MASTER_URI"]).getPid(
                            "/probe"
                        )[0]
                        == 1
                    ):
                        break
                except OSError:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("Probe master did not start")
            data = tmp / "probe.jsonl"
            subprocess.run([str(binary), str(data)], env=env, check=True, timeout=20)
            decisions = list(rows(data))
            assert len(decisions) == 4
            for i, d in enumerate(decisions):
                validate_decision(d, i)
            d = decisions[-1]
            assert (
                d["selected_candidate_id"] == 1
                and d["candidates"][0]["position"][0] == 2
            )
            assert len(d["nodes"]) == 3 and len(d["edges"]) == 2
            status = json.loads(Path(str(data) + ".status.json").read_text())
            assert status == {
                "schema_version": 1,
                "enqueued": 4,
                "written": 4,
                "failed": False,
            }
            print(
                "PASS: actual C++ empty/single/multiple branches, graph closure, input order, flush, and exclusive output"
            )
        finally:
            os.killpg(master.pid, signal.SIGINT)
            master.wait(timeout=20)
