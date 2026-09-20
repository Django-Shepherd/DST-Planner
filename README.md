# DST-Planner

Map generation, expert data collection, training and inference for DST-Planner, built on [EPIC](https://github.com/Robotics-STAR-Lab/EPIC).

![DST-Planner pipeline](assets/pipeline.png)

## Reproduction

This example uses generated maps on Ubuntu 20.04 with ROS Noetic, Conda and an NVIDIA GPU with a working OpenGL display. Datasets and pretrained weights are not included.

### 1. Prepare EPIC

```bash
git clone https://github.com/Django-Shepherd/DST-Planner.git
cd DST-Planner
DST_SOURCE="$(pwd -P)"
git clone --branch main --single-branch https://github.com/Robotics-STAR-Lab/EPIC.git ../EPIC-dst
git -C ../EPIC-dst checkout --detach d73c3150e57d669ac21bcad5c859f52c2827c9ba
EPIC_WS="$(cd ../EPIC-dst && pwd -P)"
git -C "$EPIC_WS" apply --unidiff-zero --check "$DST_SOURCE/patches/epic-adapter.patch"
git -C "$EPIC_WS" apply --unidiff-zero "$DST_SOURCE/patches/epic-adapter.patch"
cp -a learning maps assets scripts tests docs patches "$EPIC_WS/"
cp README.md "$EPIC_WS/README_DST.md"
cp LICENSE "$EPIC_WS/LICENSE_DST"
cd "$EPIC_WS"
```

[Install dependencies and build](docs/EPIC_CHANGES.md#build-and-dependencies), then run the remaining commands in the same Bash shell from `EPIC_WS`, with `LEARNING_PY` set by that guide.

### 2. Collect data

Generate 20 forest/partition maps, collect EPIC expert episodes, and prepare the training dataset:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$LEARNING_PY" maps/generate.py train \
  --output data/learning-scenes --maps-per-type 10 --seed 917000
/usr/bin/python3 scripts/collect.py --scenes data/learning-scenes/scenes.json \
  --output runs/learning-expert --workers 3 --duration 1200
"$LEARNING_PY" -m dst_planner.data \
  --roots runs/learning-expert --output data/dst-dataset --min-maps 20
```

Use new output directories. If data admission fails, inspect `admission.json` and [retry failed scenes](docs/COLLECTION.md#retry-failed-scenes).

### 3. Train

```bash
"$LEARNING_PY" -m dst_planner.train \
  --dataset data/dst-dataset/dataset.pt --output runs/dst-training \
  --steps 10000 --batch-size 256 --micro-batch 32 --device cuda:0
```

### 4. Evaluate

Generate separate forest, partition and dungeon test maps, export the selected checkpoint, and run closed-loop evaluation:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$LEARNING_PY" maps/generate.py test \
  --output data/generated-scenes --size 40 --seed 20260916
"$LEARNING_PY" -m dst_planner.predict export \
  --checkpoint runs/dst-training/best.pt --output runs/dst-training/actor.pt
/usr/bin/python3 scripts/evaluate_policy.py \
  --actor runs/dst-training/actor.pt --scene-dir data/generated-scenes \
  --learning-python "$LEARNING_PY" --output runs/dst-evaluation \
  --workers 3 --duration 300 --device cpu
```

The evaluator manages the Actor service and ROS simulation. Results are saved to `runs/dst-evaluation/evaluation.json`.

Further details: [maps](maps/README.md), [training and resume](docs/DST_LEARNING.md), [example results](docs/EXPERIMENTS.md), [troubleshooting](docs/EPIC_CHANGES.md#troubleshooting).

## License

Original DST code, map generators and documentation use [MIT](LICENSE). EPIC and the integration patch retain [GPL-3.0](patches/LICENSE).

## Citation

If you use DST-Planner, please cite our [paper](https://doi.org/10.1109/LRA.2026.3699251):

```bibtex
@article{yuan2026dstplanner,
  title   = {DST-Planner: Diffusion-Augmented Sequential Topological Planning for Robotic Exploration in Complex Environments},
  author  = {Yuan, Yiqing and Li, Zhi and Liang, Jiahui and Duan, Peiming and Zhang, Xiaoxun and Guo, Jihe and Cheng, Hui},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {11},
  number  = {7},
  pages   = {8801--8808},
  year    = {2026},
  doi     = {10.1109/LRA.2026.3699251}
}
```

## Acknowledgments

We thank the [EPIC authors](https://github.com/Robotics-STAR-Lab/EPIC) for their open-source exploration framework. We strongly recommend also citing [EPIC](https://doi.org/10.1109/LRA.2025.3555878) when using DST-Planner:

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
