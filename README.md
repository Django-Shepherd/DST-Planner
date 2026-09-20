# DST-Planner

**DST-Planner: Diffusion-Augmented Sequential Topological Planning for Robotic Exploration in Complex Environments**

This repository provides procedural map generators, the DST training and inference implementation, and an integration patch for EPIC. The workflow is: clone a fixed EPIC revision, apply the patch and copy the additional source directories, build the ROS workspace, generate maps, collect EPIC expert data, train DST, and evaluate the Actor in closed-loop exploration.

EPIC provides sensing, candidate viewpoints, topology, feasible paths, local trajectories and control. Expert collection uses EPIC's existing route solver. During learned inference, the DST Actor selects up to five **candidate exploration viewpoints** and EPIC connects them. Intermediate skeleton nodes do not count toward the five-viewpoint horizon.

See the [file-by-file integration changes](docs/EPIC_CHANGES.md), [learning and data contracts](docs/DST_LEARNING.md), [map generation tools](maps/README.md), and [example experiment](docs/EXPERIMENTS.md).

![DST-Planner pipeline](assets/pipeline.png)

## 1. Package layout and requirements

```text
patches/epic-adapter.patch       EPIC source changes and new C++/launch files
maps/
  generate.py                   Training, test and custom map generation
  *_map_generator.py            Shared geometry and individual map algorithms
  tools/                        Optional map analysis, sampling and viewing
scripts/
  collect.py                    Single-map and scene-list expert collection
  run_episode.py                One ROS episode and its process lifecycle
  evaluate_policy.py            Closed-loop Actor evaluation
learning/dst_planner/
  data.py                       Dataset admission, conversion and splitting
  features.py                   Shared online and offline graph features
  model.py                      Graph encoder, Actor, critics and value diffusion
  train.py                      Training and checkpoint resume
  predict.py                    Actor export and inference service
tests/                          Map, data, model, recorder and transport checks
docs/                           Integration, data and method documentation
```

The commands below use **Bash** on Ubuntu 20.04 x86_64 with ROS Noetic and the EPIC build dependencies. Simulation requires an NVIDIA GPU and a working OpenGL display. Training supports one or more NVIDIA GPUs; Actor inference also runs on CPU.

Keep the two Python runtimes separate:

- `/usr/bin/python3`: ROS collection and evaluation orchestration, with ROS Python modules and NumPy.
- `LEARNING_PY`: Python >=3.10 for map generation, dataset conversion, training and Actor inference, with PyTorch, NumPy, PyYAML, Open3D, SciPy and Pillow.

The examples use Python 3.10.18, PyTorch 2.1.0+cu121 and NumPy 1.26.4.

## 2. Clone EPIC and apply the integration

Clone this repository first. The EPIC adapter targets EPIC's `main` branch at commit `d73c3150e57d669ac21bcad5c859f52c2827c9ba`. The commands create a separate EPIC workspace next to DST-Planner:

```bash
git clone https://github.com/Django-Shepherd/DST-Planner.git
cd DST-Planner
DST_SOURCE="$(pwd -P)"

git clone --branch main --single-branch \
  https://github.com/Robotics-STAR-Lab/EPIC.git ../EPIC-dst
git -C ../EPIC-dst checkout --detach \
  d73c3150e57d669ac21bcad5c859f52c2827c9ba
EPIC_WS="$(cd ../EPIC-dst && pwd -P)"

git -C "$EPIC_WS" apply --unidiff-zero --check "$DST_SOURCE/patches/epic-adapter.patch"
git -C "$EPIC_WS" apply --unidiff-zero "$DST_SOURCE/patches/epic-adapter.patch"

cp -a "$DST_SOURCE/learning" "$DST_SOURCE/maps" "$DST_SOURCE/assets" \
  "$DST_SOURCE/scripts" \
  "$DST_SOURCE/tests" "$DST_SOURCE/docs" "$DST_SOURCE/patches" "$EPIC_WS/"
cp "$DST_SOURCE/README.md" "$EPIC_WS/README_DST.md"
cp "$DST_SOURCE/LICENSE" "$EPIC_WS/LICENSE_DST"

cd "$EPIC_WS"
git rev-parse HEAD
git status --short
```

`HEAD` must match the pinned commit. The branch may advance, so the commit is the actual compatibility target. The patch changes nine upstream files and adds six source/launch files plus `.gitignore`. Apply this zero-context patch with **`--unidiff-zero`**, as shown above, once to the specified revision.

EPIC's original `README.md` is preserved; this guide is copied as `README_DST.md`. If the patch check fails, inspect the base revision and checkout state before continuing. Run all remaining commands from `EPIC_WS`. The simulation-map generator reads EPIC's `garage.yaml` template, so it also needs the assembled workspace.

## 3. Install dependencies and build the complete workspace

Install ROS Noetic first. On a system with ROS already installed, install common tools, explicit native libraries and declared ROS dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git \
  python3-catkin-tools python3-rosdep python3-numpy mesa-utils \
  libeigen3-dev libopencv-dev libpcl-dev libboost-all-dev \
  libglew-dev libglfw3-dev libgl1-mesa-dev libarmadillo-dev qtbase5-dev

# If rosdep has never been initialized, first run: sudo rosdep init
rosdep update
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic
```

The native packages cover Eigen, PCL, OpenCV, Boost, Armadillo, Qt, GLEW, GLFW and OpenGL used by upstream CMake files. GPU rendering needs an NVIDIA driver and an OpenGL context; the learning environment installs the PyTorch CUDA runtime separately. This EPIC revision links GLFW through `/usr/lib/x86_64-linux-gnu/libglfw.so`, so this guide targets x86_64 Linux.

Upstream `local_sensing/package.xml` still declares legacy `svo_msgs` and `vikit_ros` dependencies that are not used by its current renderer CMake. If these are the only unresolved rosdep keys, append `--skip-keys "svo_msgs vikit_ros"` to the rosdep command. Resolve other reported dependencies rather than ignoring them.

If Conda is installed, create a separate learning environment as follows. Alternatively, set `LEARNING_PY` to the absolute interpreter path of an existing compatible environment:

```bash
conda create -n dst-planner python=3.10.18 pip -y
LEARNING_PY="$(conda run -n dst-planner python -c 'import sys; print(sys.executable)')"

"$LEARNING_PY" -m pip install "numpy==1.26.4"
"$LEARNING_PY" -m pip install "torch==2.1.0" \
  --index-url https://download.pytorch.org/whl/cu121
"$LEARNING_PY" -m pip install -e learning
"$LEARNING_PY" -m pip install open3d scipy Pillow PyYAML
```

There is no need to activate the learning environment in the ROS shell. Calling its interpreter explicitly avoids switching ROS build tools to Conda Python. The CUDA 12.1 PyTorch wheel needs a compatible NVIDIA driver; other combinations require validation.

The first build must include the complete workspace, including messages and EPIC/MARSIM dependencies:

```bash
source /opt/ros/noetic/setup.bash
catkin build -j4 -p2 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash

/usr/bin/python3 -c 'import rospy, numpy; from traj_utils.msg import PolyTraj'
"$LEARNING_PY" -c 'import torch, open3d, scipy, yaml; import dst_planner; print(torch.__version__, torch.cuda.is_available())'
```

Reduce `-j4 -p2` if memory is limited. A first build must not use only `epic_planner --no-deps`.

When reopening a terminal, enter the EPIC workspace, source the two setup files and set the learning interpreter again:

```bash
cd /absolute/path/to/EPIC-dst
source /opt/ros/noetic/setup.bash
source devel/setup.bash
LEARNING_PY="$(conda run -n dst-planner python -c 'import sys; print(sys.executable)')"
```

Before collection, check the OpenGL display available to this shell:

```bash
printf 'DISPLAY=%s\n' "$DISPLAY"
glxinfo -B
```

`DISPLAY` must identify a working GPU display on this machine. Headless hosts also need a usable OpenGL display; no fixed display number is assumed. The episode runner defaults its child-process `__GL_SYNC_TO_VBLANK` to `0` to avoid sensor throttling by swap synchronization, while respecting an explicitly inherited value.

## 4. Generate one forest, partition and dungeon test map

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
"$LEARNING_PY" maps/generate.py test \
  --output data/generated-scenes --size 40 --seed 20260916
```

Each `data/generated-scenes/{forest,partition,dungeon}/` directory contains PCD, PLY and PNG representations, `scene.json` with the seed and start position, a matching `epic.yaml`, and `coverage_grid.npz` for coverage auditing. All three representations share the same scene geometry.

Keep these maps separate from the training maps. One test map per type is insufficient for a scene-stratified train/validation split.

## 5. Generate 20 training maps and collect EPIC expert episodes

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
"$LEARNING_PY" maps/generate.py train \
  --output data/learning-scenes --maps-per-type 10 --seed 917000

/usr/bin/python3 scripts/collect.py \
  --scenes data/learning-scenes/scenes.json \
  --output runs/learning-expert --workers 3 --duration 1200
```

This preset creates ten 40 m forest maps and ten 40 m partition maps, each with two starts: 40 collection attempts. Dungeon is an additional held-out scene type in this example. `--workers 3` runs three simulations concurrently; use `1` on a resource-constrained host. `--duration` is the per-episode time budget in seconds after triggering exploration.

Collection uses the EPIC expert and needs no checkpoint or Actor service. `collect.py --scenes` enables learning features, selects scene starts, attaches the coverage grid, requires natural completion and gives each episode a separate ROS master. It cleans up only processes that it started. For repeated runs on one map, use `collect.py --map` as shown in the [collection guide](docs/COLLECTION.md).

The batch command exits with a nonzero status if any episode fails. Inspect `manifest.json` and the per-scene log files, then retry the affected scenes; the results of other attempts are retained.

Each episode records decision graphs, applied trajectories, odometry, end reasons, coverage, configurations and hashes. Natural `FINISH` and time-limit truncation remain separate. FINISH alone is not proof of adequate coverage.

Use new output directories for generation, collection, conversion and evaluation. `scenes.json` contains absolute paths to local map/configuration files; update those paths or regenerate the manifest if the workspace moves.

## 6. Admit data, retry failed scenes and split by map

```bash
PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.data \
  --roots runs/learning-expert \
  --output data/dst-dataset --min-maps 20
```

Admission requires an EPIC expert episode with complete records, natural completion, learning features, a coverage audit, at least 95% observed reachable floor, no odometry samples with projected obstacle clearance below 0.5 m, and no out-of-map samples. Rejected episodes remain on disk; `admission.json` records rejection reasons.

Forty attempts do not guarantee forty accepted episodes or twenty accepted maps. The collection `completion.json` records that attempts ended; the dataset converter decides final admission. `--min-maps 20` checks accepted unique maps. Add `--min-episodes 40` if at least forty accepted episodes are required.

To retry specific failed scenes, select their names from the original manifest without changing their map or start parameters:

```bash
# Replace these names with the scenes that require another collection attempt.
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
wanted = {'forest_00_start0', 'partition_00_start1'}
scenes = json.loads(Path('data/learning-scenes/scenes.json').read_text())
chosen = [scene for scene in scenes if scene['name'] in wanted]
assert {scene['name'] for scene in chosen} == wanted, 'Scene name not found'
with Path('data/learning-scenes/retry-scenes.json').open('x') as output:
    json.dump(chosen, output, indent=2)
PY

/usr/bin/python3 scripts/collect.py \
  --scenes data/learning-scenes/retry-scenes.json \
  --output runs/learning-expert-retry --workers 1 --duration 1200

PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.data \
  --roots runs/learning-expert runs/learning-expert-retry \
  --output data/dst-dataset-v2 --min-maps 20
```

Repeated attempts may still fail. Inspect each episode's `launch.log`, `summary.json` and admission reason for planning, coverage, start-position or runtime problems. Preserve each attempt in a new directory. A failed conversion also retains its admission report. Set training `--dataset` to the final successfully produced dataset path.

The converter groups episodes by point-cloud SHA-256, keeping all starts of a map in one split. The approximately 80/20 split is stratified by scene type; each included type needs at least two accepted maps. Outputs are:

- `dataset.pt`: TD transitions and expert sequence supervision.
- `manifest.json`: split map hashes, sample counts and dataset hash.
- `admission.json`: accepted and rejected episode records.

Rewards are **`-actual flown metres + 100 * true_finish`**, without distance scaling. A true finish earns one bonus per episode; timeout earns none. Transitions are aligned with actually applied trajectories and decisions. Recovery-control intervals are excluded from TD supervision; see the [learning contracts](docs/DST_LEARNING.md).

## 7. Train DST

The examples assume `data/dst-dataset/dataset.pt` was created successfully. Choose either the single-GPU or two-GPU launch and retain its output path consistently.

Single GPU:

```bash
PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.train \
  --dataset data/dst-dataset/dataset.pt \
  --output runs/dst-training \
  --steps 10000 --batch-size 256 --micro-batch 32 --device cuda:0
```

Two GPUs:

```bash
PYTHONPATH=learning "$LEARNING_PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m dst_planner.train \
  --dataset data/dst-dataset/dataset.pt \
  --output runs/dst-training-2gpu \
  --steps 10000 --batch-size 256 --micro-batch 128 --seed 20260917
```

`--batch-size` is global and must be divisible by `micro-batch * GPU count`. Reduce the micro-batch if memory is insufficient while preserving divisibility. The single-GPU value of 32 is a starting point, not a guarantee for arbitrary graph sizes. Multi-GPU training uses NCCL; CPU inference is independent of the training GPU count.

Training includes the shared graph encoder, prefix sequence Actor, twin Q networks, conditional value diffusion, trust-weighted TD targets, and whole-sequence policy gradients plus BC.

Outputs include `config.json`, `metrics.jsonl`, `best.pt` selected by validation sequence cross-entropy, and `last.pt`. `completion.json` is written only when all requested steps finish.

Resume by repeating the original launch with `--resume PATH/last.pt`, retaining the output directory, dataset, total steps, architecture, batch settings and GPU count. Two-GPU resume uses the same distributed launcher. `--stop-after N` pauses at absolute step N while preserving the original learning-rate schedule.

For example, resume the single-GPU run with the same arguments:

```bash
PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.train \
  --dataset data/dst-dataset/dataset.pt \
  --output runs/dst-training \
  --steps 10000 --batch-size 256 --micro-batch 32 --device cuda:0 \
  --resume runs/dst-training/last.pt
```

The last saved checkpoint is updated at validation intervals, every 250 steps by default. To pause at a known step, add `--stop-after N` when starting the run. Resume without that pause flag. For distributed training, repeat the original two-GPU command and add its checkpoint path.

## 8. Export the Actor and run strict closed-loop inference

The remaining commands use `runs/dst-training`. If the two-GPU example was used, replace that directory consistently with `runs/dst-training-2gpu` in export, evaluation and service commands.

```bash
PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.predict export \
  --checkpoint runs/dst-training/best.pt \
  --output runs/dst-training/actor.pt
```

The export contains only the shared graph encoder and Actor, with no Q networks or diffusion value model. It refuses to overwrite an existing destination; use another filename for a different version.

With ROS and `devel/setup.bash` sourced and the GPU display available, evaluate all three independent test scenes:

```bash
/usr/bin/python3 scripts/evaluate_policy.py \
  --actor runs/dst-training/actor.pt \
  --scene-dir data/generated-scenes \
  --learning-python "$LEARNING_PY" \
  --output runs/dst-evaluation \
  --workers 3 --duration 300 --device cpu
```

The evaluator reads each map's configuration and start, launches an independent Actor service and ROS simulation per scene, and cleans them up afterward. No manually started service is needed. Use `--workers 1` for sequential evaluation. The explicit 300-second budget applies to each scene; the script's default is 1200 seconds, so set a consistent value when comparing runs.

Evaluation uses `strict` policy control. Model errors do not silently fall back to EPIC. A pass requires natural completion, actual model use, no inference errors or fallback, no LKH decisions, valid records, at least 95% observed reachable-floor coverage, and no near-obstacle or out-of-map samples.

Inspect `runs/dst-evaluation/evaluation.json` and per-scene summaries/logs. They report completion, distance, time, coverage, decision sources, model calls/errors and inference-computation latency quantiles. Preserve failures reported by a nonzero exit code or `passed: false`.

For custom integration, run a persistent service:

```bash
PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.predict serve \
  --actor runs/dst-training/actor.pt \
  --socket /tmp/dst-policy.sock --device cpu --threads 2
```

Leave the service running. In another terminal on the same host, run one forest episode. Replace the workspace path with your installation directory:

```bash
cd /absolute/path/to/EPIC-dst
source /opt/ros/noetic/setup.bash
source devel/setup.bash
SCENE=data/generated-scenes/forest
read -r INIT_X INIT_Y INIT_Z < <(
  /usr/bin/python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["initial_position"])' "$SCENE/scene.json"
)

/usr/bin/python3 scripts/run_episode.py \
  --map "$SCENE/forest.pcd" --config "$SCENE/epic.yaml" \
  --coverage-grid "$SCENE/coverage_grid.npz" \
  --init "$INIT_X" "$INIT_Y" "$INIT_Z" --init-yaw 0 \
  --learning-features --policy-mode strict --policy-socket /tmp/dst-policy.sock \
  --output runs/dst-forest --startup-timeout 90 --duration 300
```

The runner starts the ROS master and simulation, triggers exploration and records the episode. It does not stop the separately started Actor service; stop that service with Ctrl-C when finished. Both terminals need access to the appropriate display/socket. The one-scene example uses the generated forest preset's default yaw of zero; custom scene lists may specify a different `initial_yaw`.

For an EPIC expert run on the same scene, change `--policy-mode strict` to `--policy-mode epic`, omit `--policy-socket`, and choose a new `--output` directory. This provides a same-scene baseline with the same telemetry and coverage checks.

## 9. Tests

After installing dependencies and building the workspace, run the existing checks:

```bash
/usr/bin/python3 tests/test_collection_dispatch.py
/usr/bin/python3 tests/test_scene_generation_cli.py
"$LEARNING_PY" tests/test_dataset_preparation.py
"$LEARNING_PY" tests/test_learning.py
"$LEARNING_PY" tests/test_training_resume.py
/usr/bin/python3 tests/test_policy_transport.py
/usr/bin/python3 tests/run_recorder_probe.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$LEARNING_PY" tests/test_generated_maps.py
```

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for an example training run and its closed-loop results. Export `best.pt` selected on validation data; it can differ substantially from `last.pt`. The repository contains no pretrained weights, so generate data and train before running inference.

## 10. Outputs and troubleshooting

| Step | Main output |
|---|---|
| Test-map generation | `data/generated-scenes/{forest,partition,dungeon}/`: point cloud, image, scene metadata, EPIC configuration and coverage grid |
| Training-map generation | `data/learning-scenes/scenes.json`: map paths, configurations and starts |
| Expert collection | `runs/learning-expert/<scene>/`: decisions, events, summary and logs |
| Dataset preparation | `data/dst-dataset/dataset.pt`, `manifest.json`, `admission.json` |
| Training | `runs/dst-training/{best.pt,last.pt,config.json,metrics.jsonl,completion.json}` |
| Actor export | `runs/dst-training/actor.pt` |
| Closed-loop evaluation | `runs/dst-evaluation/evaluation.json` and per-scene summaries/logs |

The two-GPU example uses `runs/dst-training-2gpu`; substitute that directory when exporting or evaluating its model. If data preparation succeeded in a retry directory such as `data/dst-dataset-v2`, use that dataset path for training.

| Symptom | What to check |
|---|---|
| Patch application fails | Confirm the exact EPIC commit, a clean checkout and the `--unidiff-zero` flag. Do not apply the patch twice. |
| `rospy` or generated ROS messages cannot be imported | Use `/usr/bin/python3` and source both ROS Noetic and the workspace's `devel/setup.bash`. |
| `dst_planner`, Torch or Open3D cannot be imported | Use `LEARNING_PY` and install the learning package and map dependencies in that interpreter. |
| `no odom` or simulation startup timeout | Check `glxinfo -B`, the current `DISPLAY`, and the episode's `launch.log`. |
| Output directory/file already exists | Choose a new run directory or export filename. Use `--resume` only for an existing training run. |
| Not enough admitted maps or episodes | Read `admission.json`, address the reported cause and collect the missing scenes again. Combine attempt directories with `--roots`. |
| Training runs out of GPU memory | Reduce `--micro-batch` while keeping global batch divisible by micro-batch times GPU count. |
| Actor socket is already in use | Stop the service that owns it or choose another socket path for both service and runner. |
| Strict evaluation fails or times out | Read the scene's summary and policy/runner logs. Evaluate the selected checkpoint and retain its failure result. |

Maps, datasets and checkpoints are generated locally by these commands. The example covers 40 m procedural scenes; paper-scale maps and Hotel/Cave experiments are separate experiments. Keep EPIC's upstream credits, citations and license notices.

## License

The original DST-planner code, map generators and documentation are released
under the [MIT License](learning/LICENSE).

EPIC and the EPIC integration patch retain the upstream GPL-3.0 license;
see [patches/LICENSE](patches/LICENSE). The MIT license does not relicense
EPIC or other third-party components. In the assembled workspace, EPIC's
root `LICENSE` is preserved and the DST license is copied as `LICENSE_DST`.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{yuan2026dstplanner,
  title   = {DST-Planner: Diffusion-Augmented Sequential Topological Planning for Robotic Exploration in Complex Environments},
  author  = {Yuan, Yiqing and Li, Zhi and Liang, Jiahui and Duan, Peiming and Zhang, Xiaoxun and Guo, Jihe and Cheng, Hui},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```
