# DST-Planner

**DST-Planner: Diffusion-Augmented Sequential Topological Planning for Robotic Exploration in Complex Environments**

Map generation, data collection, training and inference for DST-Planner, built on [EPIC](https://github.com/Robotics-STAR-Lab/EPIC). EPIC provides the exploration environment and expert planner; DST learns to select viewpoint sequences for exploration.

![DST-Planner pipeline](assets/pipeline.png)

## Reproduction

The steps below cover the generated-map example: **clone and patch EPIC → build → collect data → train → test**. Datasets and pretrained weights are not included. See [example results and scope](docs/EXPERIMENTS.md).

**Requirements:** Ubuntu 20.04 x86_64, ROS Noetic, Conda, an NVIDIA GPU with a compatible driver, and a working OpenGL display. Run the commands in Bash, keeping ROS on `/usr/bin/python3` and learning in its separate Conda environment.

### 1. Prepare the EPIC workspace

```bash
git clone https://github.com/Django-Shepherd/DST-Planner.git
cd DST-Planner
DST_SOURCE="$(pwd -P)"

git clone --branch main --single-branch \
  https://github.com/Robotics-STAR-Lab/EPIC.git ../EPIC-dst
git -C ../EPIC-dst checkout --detach d73c3150e57d669ac21bcad5c859f52c2827c9ba
EPIC_WS="$(cd ../EPIC-dst && pwd -P)"

git -C "$EPIC_WS" apply --unidiff-zero --check "$DST_SOURCE/patches/epic-adapter.patch"
git -C "$EPIC_WS" apply --unidiff-zero "$DST_SOURCE/patches/epic-adapter.patch"
cp -a learning maps assets scripts tests docs patches "$EPIC_WS/"
cp README.md "$EPIC_WS/README_DST.md"
cp LICENSE "$EPIC_WS/LICENSE_DST"
cd "$EPIC_WS"
```

Use the pinned EPIC commit above. The patch applies the required source changes; the [integration guide](docs/EPIC_CHANGES.md) explains each change. Run all following commands from `EPIC_WS`.

### 2. Install dependencies and build

With ROS Noetic and Conda installed:

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

If rosdep reports only the unused legacy keys `svo_msgs` and `vikit_ros`, rerun it with `--skip-keys "svo_msgs vikit_ros"`. Simulation needs a working GPU display, including on headless hosts. After reopening a terminal, enter `EPIC_WS`, source both setup files and set `LEARNING_PY` again.

### 3. Generate maps and collect training data

Create 20 forest/partition maps with two starts each, collect EPIC expert episodes, then prepare the dataset:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
"$LEARNING_PY" maps/generate.py train \
  --output data/learning-scenes --maps-per-type 10 --seed 917000

/usr/bin/python3 scripts/collect.py \
  --scenes data/learning-scenes/scenes.json \
  --output runs/learning-expert --workers 3 --duration 1200

"$LEARNING_PY" -m dst_planner.data \
  --roots runs/learning-expert \
  --output data/dst-dataset --min-maps 20
```

Collection uses EPIC directly and requires no checkpoint. Only validated, naturally completed episodes enter the dataset; train/validation splits keep each map separate. If admission fails, inspect `admission.json` and [retry the affected scenes](docs/COLLECTION.md#retry-failed-scenes). Use fresh output directories; set `--workers 1` if resources are limited.

### 4. Train DST

```bash
"$LEARNING_PY" -m dst_planner.train \
  --dataset data/dst-dataset/dataset.pt \
  --output runs/dst-training \
  --steps 10000 --batch-size 256 --micro-batch 32 --device cuda:0
```

Training saves the validation-selected `best.pt` and the latest `last.pt`. For multi-GPU training, checkpoint resume and method details, see the [learning guide](docs/DST_LEARNING.md).

### 5. Run inference and test

Generate separate forest, partition and dungeon test maps, export the trained Actor, and evaluate it in EPIC:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
"$LEARNING_PY" maps/generate.py test \
  --output data/generated-scenes --size 40 --seed 20260916

"$LEARNING_PY" -m dst_planner.predict export \
  --checkpoint runs/dst-training/best.pt \
  --output runs/dst-training/actor.pt

/usr/bin/python3 scripts/evaluate_policy.py \
  --actor runs/dst-training/actor.pt \
  --scene-dir data/generated-scenes \
  --learning-python "$LEARNING_PY" \
  --output runs/dst-evaluation \
  --workers 3 --duration 300 --device cpu
```

The evaluator starts and stops the Actor service and ROS simulation automatically. Results are written to `runs/dst-evaluation/evaluation.json`; failures remain visible and do not silently fall back to EPIC.

For custom maps, see the [map guide](maps/README.md). For build or runtime issues, see [troubleshooting](docs/EPIC_CHANGES.md#troubleshooting).

## License

Original DST code, map generators and documentation use the [MIT License](LICENSE). EPIC and the integration patch retain [GPL-3.0](patches/LICENSE).

## Citation

If you find DST-Planner useful, please consider citing:

```bibtex
@article{yuan2026dstplanner,
  title   = {DST-Planner: Diffusion-Augmented Sequential Topological Planning for Robotic Exploration in Complex Environments},
  author  = {Yuan, Yiqing and Li, Zhi and Liang, Jiahui and Duan, Peiming and Zhang, Xiaoxun and Guo, Jihe and Cheng, Hui},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

## Acknowledgments

We thank the authors of [EPIC](https://github.com/Robotics-STAR-Lab/EPIC) for sharing their exploration framework. EPIC provides the foundation for this project's exploration environment, expert data collection and planning integration.

**We strongly encourage everyone using DST-Planner to also cite the [EPIC paper](https://doi.org/10.1109/LRA.2025.3555878):**

```bibtex
@article{10945408,
  author  = {Geng, Shuang and Ning, Zelin and Zhang, Fu and Zhou, Boyu},
  title   = {EPIC: A Lightweight LiDAR-Based AAV Exploration Framework for Large-Scale Scenarios},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025},
  volume  = {10},
  number  = {5},
  pages   = {5090-5097},
  doi     = {10.1109/LRA.2025.3555878}
}
```
