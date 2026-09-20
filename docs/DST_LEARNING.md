# DST learning and online policy

For the ordered clone, patch, build and run workflow, start with `README_DST.md`
in the assembled EPIC workspace (or the root README of the DST source package).
This page describes the model/data contracts and additional commands.

The action contract is a sequence of up to five candidate exploration viewpoints. Intermediate skeleton nodes are handled by EPIC path search, not counted as actions. Rewards are `-actual_distance_m + 100 * true_finish`: no reward scaling, exactly one finish bonus, and no bonus on timeout.

## Source layout and dependencies

`learning/dst_planner` is an installable ROS-independent Python package (Python >=3.10, PyTorch >=2.1, NumPy <2, PyYAML). `pip install -e learning` installs it; alternatively use `PYTHONPATH=learning`. Existing ROS Noetic tools use the system Python. Map generation additionally uses Open3D/SciPy/Pillow.

- `features.py`: common offline/online graph tensor construction; no expert labels or TSP-adjusted costs enter the model.
- `data.py`: the dataset preparation entry point, strict episode admission, validated sequence labels and actual applied-plan transitions; map-level train/validation split stratified by scene type.
- `model.py`: shared linear global attention/local graph encoder, prefix Transformer actor, twin Q, conditional diffusion value model.
- `train.py`: full losses, synchronous distributed gradient averaging, one optimizer owning shared parameters, atomic checkpoints and per-rank RNG resume.
- `predict.py`: actor-only export and local Unix-socket inference. No ROS/PyTorch environment coupling.

The diffusion target is the bootstrapped TD target, with a cosine 10-step schedule. The critic target combines deterministic twin-Q values with diffusion-derived trust. Actor BC uses expert prefixes; its trusted-Q term uses sampled actor sequences and a whole-sequence score-function gradient with an independent action-independent rollout baseline.

## Observation and reward contracts

The encoder sees the full graph. The actor sees a separate viewpoint path-reachability matrix captured **before** EPIC's TSP matrix heuristic adjustments. It masks unavailable/repeated viewpoints. Sequences may be shorter than five; this does not mean exploration has finished.

Information gain is `observed_voxel_raycast_v1`: a coarse **estimate**, from observed LiDAR points only. Measured endpoints mark occupied 1 m voxels; up to approximately 1024 sampled rays per cloud mark observed voxels (0.5 m steps, capped at 18 m). Candidate rays cover 36 azimuths and seven elevations from -60 to +60 degrees, capped at 12 m and the exploration box, and stop at known occupied voxels. Unique unseen sampled cells give an estimated volume in cubic metres. This is not simulator ground truth or an exact sensor/frustum volume integration. Offline and online use the same estimate, recorded in every snapshot.

`trajectory_applied` events link a decision ID to the actual target, trajectory ID, publication time and trajectory start time. The dataset verifies the target and trajectory exist. Rewards integrate recorded 3D odometry between activations of successive actually applied plans; activation is the later of publication and trajectory start. Multiple trajectories for one decision remain one plan interval. The source graph observation precedes activation; its age is recorded as `apply_latency_s`. No target arrival is inferred from a new global planning callback. Recovery intervals are excluded from TD supervision. The raw data remains available for auditing.

Dataset rewards can be negative over a whole completed run because distances are unscaled metres. No additional reward normalization is applied. Both termination and time-limit truncation end bootstrapping in the paper-mode objective, while the raw flags remain separate. The dataset converter accepts only naturally completed, validated EPIC expert episodes. A coverage grid and a completed coverage audit are required, with at least 95% observed reachable floor, no samples below 0.5 m projected obstacle clearance, and no out-of-map samples; rejected episodes remain on disk. These are dataset admission criteria, not altered rewards.

Use `python -m dst_planner.data --roots RUN_DIRECTORY --output DATASET_DIRECTORY`
with the learning interpreter. Multiple run directories can follow `--roots`.
The converter keeps all starts of one map in the same split, requires at least
two accepted maps per scene type, and retains rejection reasons in
`admission.json` even when the requested minimum dataset size is not met.

The runner sets `__GL_SYNC_TO_VBLANK=0` for its simulation children unless an explicit value is already present. This avoids swap-synchronization throttling in headless OpenGL sessions. EPIC uses synchronized cloud and odometry poses.

## Training parameters

The main README provides the single-GPU workflow. Run the additional commands below from the assembled EPIC workspace with its configured `LEARNING_PY` interpreter; use the system Python for ROS orchestration.

| Parameter | Default or example value | Where to set it |
|---|---|---|
| Training steps | 10,000 | `--steps` |
| Global batch | 256 | `--batch-size` |
| Micro-batch | CLI default 16; examples use 32 on one GPU or 128 on two GPUs | `--micro-batch` |
| Hidden width | 128 | `--hidden` |
| Seed | 20260917 | `--seed` |
| Validation interval | 250 steps | `--eval-every` |
| CPU threads per rank | 4 | `--threads` |
| Graph/history layers | 3 / 2 | `ModelConfig` in `model.py` |
| Attention heads / action horizon | 4 / 5 | `ModelConfig` in `model.py` |
| Diffusion steps / samples | 10 / 5 | `ModelConfig` in `model.py` |
| Learning rate | Adam, 1e-4 cosine-decayed to 1e-5 | `train.py` |
| Discount / target update rate | 0.99 / 0.005 | `train.py` |
| BC weight / gradient-norm limit | 3 / 1 | `train.py` |

Architecture or loss changes must be made in the corresponding Python configuration or training code. Preserve configuration, data and GPU count when resuming an existing run.

### Two-GPU training

```bash
PYTHONPATH=learning "$LEARNING_PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m dst_planner.train \
  --dataset data/dst-dataset/dataset.pt \
  --output runs/dst-training-2gpu \
  --steps 10000 --batch-size 256 --micro-batch 128 --seed 20260917
```

`--batch-size` is global and must be divisible by `micro-batch * GPU count`.
Reduce the micro-batch if memory is insufficient while preserving divisibility.
Multi-GPU training uses NCCL; CPU inference is independent of the training GPU
count. Use `runs/dst-training-2gpu` consistently when exporting or evaluating
this run.

### Resume training

Repeat the original launch with `--resume PATH/last.pt`, retaining the output
directory, dataset, total steps, architecture, batch settings and GPU count.
For the single-GPU example in the main guide:

```bash
PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.train \
  --dataset data/dst-dataset/dataset.pt \
  --output runs/dst-training \
  --steps 10000 --batch-size 256 --micro-batch 32 --device cuda:0 \
  --resume runs/dst-training/last.pt
```

The checkpoint is saved at validation intervals, every 250 steps by default.
To pause at an absolute step while preserving the original learning-rate
schedule, start the run with `--stop-after N`; omit that flag when resuming.
For distributed training, repeat the original distributed command and add its
checkpoint path.

## Checkpoints and deployment

`best.pt` is selected by validation sequence cross-entropy. `last.pt` contains the latest saved training state. Both include optimizer, scheduler, target-network and random-state data for resume. `completion.json` marks the end of the requested training schedule.

`predict export` extracts the encoder and Actor from a checkpoint. `predict serve` loads this export and responds over a local Unix socket. The C++ client checks request identity, graph version, candidate indices, duplicate selections, horizon length and reachability. Online and offline paths share `features.py`.

The policy modes are described in [EPIC_CHANGES.md](EPIC_CHANGES.md). The main README covers automated multi-scene evaluation. `evaluate_policy.py` launches one Actor service per scene and shuts down the services and simulations afterward.

Report natural completion, distance, time, coverage and inference latency separately. Coverage is observed reachable floor area; information gain is an approximation from observed LiDAR. The example results and checkpoint-selection behavior are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

### Persistent service and a single-scene run

For custom integration, start a persistent service using the exported Actor:

```bash
PYTHONPATH=learning "$LEARNING_PY" -m dst_planner.predict serve \
  --actor runs/dst-training/actor.pt \
  --socket /tmp/dst-policy.sock --device cpu --threads 2
```

Leave the service running. In another terminal on the same host, run the
generated forest scene. Replace the workspace path with your installation
directory and inherit a working GPU display:

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

The runner starts the ROS master and simulation, triggers exploration and
records the episode. Stop the separately started Actor service with Ctrl-C
when finished. Both terminals must be able to access the socket. This example
uses the generated forest preset's default yaw of zero; custom scene lists may
specify a different `initial_yaw`.

For an EPIC expert baseline on the same scene, change `--policy-mode strict`
to `--policy-mode epic`, omit `--policy-socket`, and choose a new output directory.
