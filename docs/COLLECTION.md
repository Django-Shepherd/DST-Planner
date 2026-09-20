# EPIC expert data collection

Expert collection mode uses EPIC's existing viewpoint generation, topological graph, LKH tour solver,
and local trajectory planner. No learned policy, checkpoint, or training dependency
is required. The recorder is disabled unless `collection/enabled` is true.

Upstream: https://github.com/Robotics-STAR-Lab/EPIC
Base commit: `d73c3150e57d669ac21bcad5c859f52c2827c9ba`.

## Official garage scene

Use the author's maps linked in the upstream README:
https://drive.google.com/drive/folders/1tuoVo8PL1m2cmmufkHpu4e7hK36WhJs3

`garage.pcd`: Google Drive file ID `1nvXdB-uCoQqOKGD0TCK1jATvwqg9Nckq`.
Place it at `src/MARSIM/map_generator/resource/garage.pcd`. Large maps and generated
runs are excluded from Git. The collection launch defaults to the unchanged
`garage.yaml`, initial position `(5, 0, 2)`, GPU rendering, 60 m sensor range,
40 degree sensor pitch and 0.2 degree angular resolution. Like `algorithm.xml`,
it overrides `bubble_topo/cube_discrete_size` to 0.3.

The collection launch omits RViz and the interactive waypoint tool. The Python
runner sends the start message automatically and gives every episode a separate LKH directory.
The sensor odometry argument has no leading slash to avoid the upstream include's
`/quad_0//lidar_slam/odom` double separator. No simulator algorithms are replaced.

## Build and collect

Ubuntu 20.04, ROS Noetic, catkin_tools, and the upstream build dependencies are
required. Python scripts use ROS's system Python 3; validation uses only the
standard library. GPU rendering requires a usable OpenGL display.

```bash
source /opt/ros/noetic/setup.bash
catkin build -j4 -p2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
/usr/bin/python3 scripts/download_garage.py
# Inherit a working NVIDIA/OpenGL DISPLAY from your local graphical session.
# For a headless host, configure a usable GPU display before running this command.
/usr/bin/python3 scripts/run_episode.py \
  --map src/MARSIM/map_generator/resource/garage.pcd \
  --output runs/garage-001 --startup-timeout 90 --duration 600
/usr/bin/python3 scripts/validate_episode.py runs/garage-001
```

`--baseline` disables internal decision recording. `--init X Y Z` changes the
initial position; `--config PATH` changes the planner configuration. New output
directories are mandatory: an existing episode is never overwritten. Explicit
`--cpu-render` is a fallback with different renderer characteristics; the official
GPU setup is the validated default. Sensor settings remain visible in metadata.

Use `collect.py --map` for sequential runs on one map:

```bash
/usr/bin/python3 scripts/collect.py \
  --map src/MARSIM/map_generator/resource/garage.pcd \
  --output runs/garage-batch --count 2 --duration 600
```

Every episode owns a separate loopback ROS master. Cleanup addresses only the
processes started for that episode; there is no global `pkill`. A failed episode
stops the batch and is retained with logs. `manifest.json` lists accepted/rejected
runs. Add `--require-complete` to the validator or batch command when truncated
runs must be rejected.

The same entry point accepts `--scenes scenes.json` for a list of maps and starts.
This mode defaults to three workers and a 1200-second episode budget, enables
learning features, and requires natural completion. The `--map` mode defaults
to two sequential episodes with a 180-second budget; use `--learning-features`
and `--require-complete` when needed. Both modes forward `--cpu-render` and
`--startup-timeout` to the episode runner. Map-specific configuration, coverage
grid, position and yaw come from each scene-list entry, or from `--config`,
`--coverage-grid`, `--init` and `--init-yaw` in single-map mode.

Creating an empty `STOP` file in the batch output directory prevents queued
episodes from starting and lets active episodes finish. Ctrl-C or SIGTERM asks
active runners to stop and clean up their processes.

## Files and semantics

- `metadata.json`: schema version, upstream commit, tracked diff, SHA-256 of
  collection sources, map SHA-256, full configuration, simulator settings and
  episode identity. A scene's bytes are reproducible; real-time ROS scheduling
  and EPIC's internal random yaw perturbations are not bitwise deterministic.
- `decisions.jsonl`: pointer-free snapshots copied before temporary viewpoint
  nodes are removed. Contains robot position/velocity/yaw, planning anchor,
  `uses_previous_goal`, graph nodes/directed edges, reachable candidates, full
  floating-point cost matrix, expert tour and selected candidate ID.
- `decisions.jsonl.status.json`: produced only when the recorder drains and closes;
  contains enqueued/written counts and the failure flag. Missing status invalidates
  the episode. Overflow/write errors are published and cause collection to stop.
- `events.jsonl`: timestamped odometry, published polynomial position trajectories,
  FSM state changes, trigger and episode end. This is executed-motion telemetry,
  not a ready-made RL transition dataset.
- `summary.json`: end reason, separate `terminated`/`truncated` flags, distance,
  message counts and errors. `epic_finish` means EPIC reported FINISH; it does not
  independently prove 100% map coverage. `time_limit` is truncation, not success.
- `launch.log`, `master.log`, `roslog/`, `lkh/`: diagnostic files and per-run solver
  scratch space. The last LKH scratch files are not a history of all decisions.

Graph node IDs and candidate IDs are local to a snapshot. Candidate order is the
original reachable-viewpoint order and is never sorted using the expert answer.
For LKH, route element 0 denotes the robot, and route element `k>0` maps to
candidate `k-1`; `selected_candidate_id = expert_route[1]-1`. The stored matrix
is before LKH's integer scaling (`int(cost*100)`). Its first row can be based on
the previous goal; use `planning_anchor` instead of assuming it always starts at
the measured position. Directed adjacency without a cached weight is exported
as `null` and counted by validation.

`single_candidate` has route `[0,1]` and no LKH matrix. `no_candidates` and
`no_reachable_candidates` have no action. No-candidate records are observations,
not automatically terminal transitions. EPIC plans during warm-up too: use the
trigger timestamp and episode-end boundary when selecting samples. Replanning
can change the selected target before arrival, so adjacent decisions must not
be interpreted as completed `(s,a,s')` transitions without a later definition.
With learning features enabled, snapshots also contain observed gain and raw viewpoint reachability. Rewards are assigned offline; see DST_LEARNING.md.

Serialization copies the current snapshot in the planning callback; disk writes
run on a bounded background queue (64 pending records by default, each capped at
64 MiB). `capture_ms` measures snapshot/serialization time, excluding write time.
The collector adds timing overhead, even though it does not alter decisions.

## Small upstream correctness fixes

These are separate from collection logic and also apply with recording disabled:

1. `fast_exploration_manager.cpp`: convert LKH's candidate index using `-1` when
   reading `last_frame_value`, matching other candidate lookups and preventing an
   out-of-range read when the final candidate is selected.
2. `fast_exploration_fsm.h`: disconnect the cloud/odometry synchronizer before its
   input subscribers are destroyed.
3. `minco_planner/src/planner_manager.cpp`: when no prior yaw trajectory exists,
   initialize from the current yaw instead of indexing an empty trajectory.
   This was reproduced with recording disabled as well as enabled.
4. `path_searching/src/bubble_astar.cpp`: destroy owning node/bubble pools once,
   instead of deleting uninitialized/aliased `safe_bubble` pointers.
5. `MARSIM/local_sensing/src/opengl_render_node.cpp`: scope ROS timers to `main`
   so they are destroyed before ROS's static timer managers.

The complete integration modifies nine upstream files and adds separate helper
classes and collect.launch. See EPIC_CHANGES.md for the current file-by-file list.
Original launch/config files are preserved.

## Validation

The validator rejects incomplete writes, broken episode boundaries, invalid
node/candidate mappings, malformed matrices/routes, nonfinite state, crashes,
and runs with no actual flight. It does not certify obstacle clearance or
research dataset diversity. Run corruption tests against a valid captured run:

```bash
/usr/bin/python3 tests/run_recorder_probe.py
/usr/bin/python3 tests/test_collection_validation.py runs/garage-001
```

For learning data, follow the generated-scene collection and admission workflow
in the main guide. The basic garage example demonstrates recording but does not
include the coverage grid required by dataset preparation.
