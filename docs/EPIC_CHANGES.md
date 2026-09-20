# EPIC Integration Changes

This package adds map generation, DST learning and online policy inference to a fixed EPIC revision. Clone EPIC, check out the pinned commit, apply the adapter patch and copy the additional directories as described in the main guide.

## Base revision and patch application

- Repository: `https://github.com/Robotics-STAR-Lab/EPIC.git`
- Branch: `main`
- Pinned commit: `d73c3150e57d669ac21bcad5c859f52c2827c9ba`
- Patch: `patches/epic-adapter.patch`; modified and added files are listed below.

The commit, rather than the moving branch head, defines compatibility. The patch uses zero context. Use `git apply --unidiff-zero --check` before `git apply --unidiff-zero`. Apply it once to a clean checkout of the pinned revision.

The adapter modifies **nine upstream files** and adds **six source/launch files plus `.gitignore`**. Map generation and learning are separate source directories. PyTorch is not a dependency of the ROS C++ build.

## Modified upstream files

Paths are relative to the EPIC workspace. Patch hunks specify exact changes against the pinned revision.

| File | Change | Purpose and scope |
|---|---|---|
| `src/global_planner/exploration_manager/CMakeLists.txt` | Add `decision_recorder.cpp` and `policy_client.cpp` to `epic_planner` | Build recording and inference transport |
| `src/global_planner/exploration_manager/include/epic_planner/fast_exploration_manager.h` | Declare the recorder, decision/goal IDs, observed voxels, policy settings and execution-event interface | Own integration state; the default policy remains `epic` |
| `src/global_planner/exploration_manager/src/fast_exploration_manager.cpp` | Initialize parameters; copy graph snapshots before temporary candidates are removed; cover empty/single/multiple candidates; capture reachability before TSP heuristics; request Actor routes; associate applied goals with decisions | EPIC provides expert labels; strict Actor inference selects up to five viewpoints while EPIC connects and executes them |
| Same manager source: correctness fixes | Convert LKH indices with `-1`; treat every path-search result other than `REACH_END` as unreachable | Fix indexing and reachability; these changes also apply with collection disabled |
| `src/global_planner/exploration_manager/include/epic_planner/fast_exploration_fsm.h` | Disconnect the point-cloud/odometry synchronizer before subscriber destruction | Correct shutdown lifetime handling |
| `src/global_planner/exploration_manager/src/fast_exploration_fsm.cpp` | Record applied trajectories, planning failures and recovery; replan after completing a short Actor horizon; use cloud/odometry/synchronization queue capacities 5/500/500 | Align transitions with actual execution; avoid treating a short route as task completion; retain synchronized pose processing |
| `src/global_planner/exploration_manager/src/fsm_utils.cpp` | Update observed voxels in the synchronized cloud callback and record trajectory stops | Maintain common online/offline gain features and execution boundaries |
| `src/local_planner/minco_planner/src/planner_manager.cpp` | Initialize from current yaw when no previous yaw trajectory exists | Avoid empty-trajectory access during initialization/recovery |
| `src/local_planner/path_searching/src/bubble_astar.cpp` | Release only the owning node and bubble pools | Avoid freeing uninitialized or aliased pointers |
| `src/MARSIM/local_sensing/src/opengl_render_node.cpp` | Scope ROS timers to `main` | Destroy timers before ROS static managers |

The manager source occupies two table rows but counts as one file. Correctness fixes and queue changes also affect runs with recording disabled; the patched baseline is not claimed to be bitwise identical to the original commit. EPIC still provides frontier extraction, topology, local trajectories and its original launch/configuration files.

## Files added by the patch

The following six files share the prefix `src/global_planner/exploration_manager/`:

| Files | Role |
|---|---|
| `include/epic_planner/decision_recorder.h`, `src/decision_recorder.cpp` | Pointer-free graph snapshots, bounded background writing and completion/error status |
| `include/epic_planner/observed_volume.h` | Approximate candidate information gain from observed LiDAR rays, without using the full simulator map as model input |
| `include/epic_planner/policy_client.h`, `src/policy_client.cpp` | Unix-socket transport, default 1500 ms timeout, request/graph-version matching and route validity checks |
| `launch/collect.launch` | Parameterized map, configuration, start, yaw, recording and policy options for automated operation without RViz |

The root `.gitignore` excludes Catkin outputs, run records and large generated/downloaded data.

## Additional source directories

| Directory | Entry points and purpose |
|---|---|
| `maps/` | `generate.py test` creates forest, partition and dungeon test scenes; `generate.py train` creates training maps and starts; `generate.py custom` exposes custom map parameters; optional analysis tools live in `maps/tools/` |
| `scripts/` | Expert collection, episode validation, coverage auditing, closed-loop evaluation and plotting |
| `learning/dst_planner/` | One strict dataset preparation entry point (`python -m dst_planner.data`), shared features, execution-aligned data, graph encoder/Actor/twin-Q/value-diffusion modules, training, export and serving |
| `tests/` | Model masks, permutation behavior, gradients, export, resume, actual C++ transport/recording, map generation and data validation |
| `docs/` | Integration changes, data/model contracts, collection format and validation evidence |

## Runtime modes and method contract

| `policy_mode` | Route selection | Use |
|---|---|---|
| `epic` | EPIC/LKH | Expert collection without a model; default |
| `shadow` | EPIC/LKH while also querying the Actor | Input/transport diagnostics |
| `strict` | DST Actor, with explicit failure on errors | Formal method evaluation |
| `fallback_epic` | DST Actor, with explicitly recorded EPIC fallback on errors | Debugging that permits fallback |

A single candidate is selected deterministically without a network request. Empty candidate sets, short sequences and true completion are distinct events. Actions contain at most five **candidate exploration viewpoints**, not five adjacent skeleton nodes. EPIC connects viewpoints with feasible paths.

Rewards are `-actual flown metres + 100 * true_finish`: no distance scaling, exactly one true-finish bonus, and none on timeout. Transitions use actual trajectory activation boundaries. Recovery-control intervals are excluded from TD supervision.

Online inference loads only the shared encoder and Actor. Twin Q networks and conditional value diffusion are training components. See [DST_LEARNING.md](DST_LEARNING.md) for detailed contracts.

The runner defaults `__GL_SYNC_TO_VBLANK=0` for its simulation subprocesses unless explicitly inherited. This avoids swap-synchronization throttling on the tested host; it does not configure a display server. A working NVIDIA/OpenGL display is still required.

## Build and dependencies

After applying the patch and copying the directories, run the following commands
in Bash from `EPIC_WS`. This setup assumes Ubuntu 20.04 x86_64 with ROS Noetic,
Conda and an NVIDIA driver already installed. ROS uses `/usr/bin/python3`;
learning and map generation use a separate Conda environment.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git \
  python3-catkin-tools python3-rosdep python3-numpy mesa-utils \
  libeigen3-dev libopencv-dev libpcl-dev libboost-all-dev \
  libglew-dev libglfw3-dev libgl1-mesa-dev libarmadillo-dev qtbase5-dev

# Run sudo rosdep init first if rosdep has never been initialized.
rosdep update
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic

conda create -n dst-planner python=3.10.18 pip -y
LEARNING_PY="$(conda run -n dst-planner python -c 'import sys; print(sys.executable)')"
"$LEARNING_PY" -m pip install "numpy==1.26.4"
"$LEARNING_PY" -m pip install "torch==2.1.0" \
  --index-url https://download.pytorch.org/whl/cu121
"$LEARNING_PY" -m pip install -e learning
"$LEARNING_PY" -m pip install open3d scipy Pillow PyYAML

catkin build -j4 -p2 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
glxinfo -B
```

If rosdep reports only the unused legacy keys `svo_msgs` and `vikit_ros`,
rerun its install command with `--skip-keys "svo_msgs vikit_ros"`.
Build the complete Catkin workspace initially; `epic_planner --no-deps` is
suitable only for rebuilding an existing workspace whose dependencies have
already been built.

Upstream CMake files require Eigen, PCL, OpenCV, Boost, Armadillo, Qt, GLEW,
GLFW and OpenGL. This EPIC revision links GLFW through
`/usr/lib/x86_64-linux-gnu/libglfw.so`, so the documented setup targets x86_64
Linux. Rendering requires an NVIDIA driver and a working OpenGL display;
the learning environment installs the PyTorch CUDA runtime separately.

After reopening a terminal, restore the workspace and interpreter settings
before running collection, training or evaluation. Replace the path below with
your EPIC checkout:

```bash
cd /path/to/EPIC-dst
EPIC_WS="$(pwd -P)"
source /opt/ros/noetic/setup.bash
source "$EPIC_WS/devel/setup.bash"
LEARNING_PY="$(conda run -n dst-planner python -c 'import sys; print(sys.executable)')"
```

EPIC's original README, author credits and citations remain in the checkout. The DST guide is copied as `README_DST.md`. Example experiment results are available in [EXPERIMENTS.md](EXPERIMENTS.md).

## Troubleshooting

| Symptom | What to check |
|---|---|
| Patch application fails | Confirm the pinned EPIC commit, a clean checkout and `--unidiff-zero`. Apply the patch only once. |
| `rospy` or generated ROS messages cannot be imported | Use `/usr/bin/python3` and source ROS Noetic and the workspace's `devel/setup.bash`. |
| `dst_planner`, Torch or Open3D cannot be imported | Use `LEARNING_PY` and install the learning and map dependencies in that interpreter. |
| `no odom` or simulation startup timeout | Check `glxinfo -B`, `DISPLAY` and the episode's `launch.log`. |
| Output directory/file already exists | Choose a new run directory or export filename. Use `--resume` only for an existing training run. |
| Not enough admitted maps or episodes | Read `admission.json`, address the cause and [retry the affected scenes](COLLECTION.md#retry-failed-scenes). |
| Training runs out of GPU memory | Reduce `--micro-batch` while keeping global batch divisible by micro-batch times GPU count. |
| Actor socket is already in use | Stop its owning service or choose another socket path for both service and runner. |
| Strict evaluation fails or times out | Read the scene summary and policy/runner logs. Evaluate the selected checkpoint and retain its failure result. |
