# Procedural Map Generators

These tools generate forest, partition, dungeon and hybrid terrain maps for DST-planner and EPIC simulation. End-to-end engineering validation covers the first three types.

Dependencies: Python 3, NumPy, Open3D, Pillow, SciPy and PyYAML. The optional complexity analyzer also needs `opencv-python`; the area-visualization utility and `scripts/plot_generated_runs.py` need Matplotlib. Install these optional dependencies with `"$LEARNING_PY" -m pip install "numpy<2" opencv-python matplotlib`. Use the learning/map interpreter separately from ROS Python, as described by `LEARNING_PY` in the main guide.

## Generate three simulation-ready test maps

Run from the assembled EPIC workspace root, using a new output directory:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
"$LEARNING_PY" maps/generate.py test \
  --output data/generated-example --size 40 --seed 20260916
```

Each type produces a PCD, PLY, PNG, `scene.json` with seed/geometry/start/hash metadata, a boundary-matched `epic.yaml`, and `coverage_grid.npz`. All formats share scene elements. Point clouds are downsampled to 0.1 m voxels; the audit grid uses 0.25 m cells. Starts are selected using obstacle clearance and checked against the 3D point cloud.

The 40 m preset uses 60 forest pillars, 25 partition panels and up to 12 dungeon rooms; the dungeon algorithm determines the actual room count. Walls are 3 m high and the exploration bound is 2.8 m. Seeds control Python and NumPy randomness, without guaranteeing identical file bytes across dependency versions.

## Generate training scenes

The same entry point generates the forest and partition training preset:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
"$LEARNING_PY" maps/generate.py train \
  --output data/learning-scenes --maps-per-type 10 --seed 917000
```

This creates ten 40 m maps per type and two starts per map. The output
`scenes.json` can be passed directly to `scripts/collect.py --scenes`.
The second start lies near the map interior and faces its center. Keep test-map
seeds and outputs separate from training data.

## Custom map parameters

```bash
"$LEARNING_PY" maps/generate.py custom --map-type forest \
  --output-dir data/custom --room-size 40 --num-columns 60 \
  --output-types pcd,png --batch-size 1 --seed 42
```

Use `python maps/generate.py custom --help` for per-type parameters. Each batch creates geometry once and writes the requested representations. Existing output files are rejected. The batch interface supports small batch sizes and forwards generator arguments. The individual generator modules define algorithms and have no standalone demo entry points.

Optional utilities live in `maps/tools/`. Run `maps/tools/sample_free_points.py --input DIR --map-size 40 --clearance 1 --count 5` with the learning interpreter to sample from PNG images. `maps/tools/calculate_map_area.py --help` and `maps/tools/calculate_map_complexity.py --help` describe optional analysis utilities; `maps/tools/pcd_watch.py --help` describes the interactive point-cloud viewer. These do not influence EPIC decisions or runtime coverage auditing; simulation starts use the separate 3D point-cloud check.

## Checks

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
"$LEARNING_PY" tests/test_generated_maps.py
```

Checks cover batch count/prefix handling, seeded and cross-format consistency, overwrite rejection, floor filtering and the reachable-area denominator. See the [example experiment](../docs/EXPERIMENTS.md).
