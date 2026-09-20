#!/usr/bin/env python3
"""Generate training scenes, held-out EPIC scenes, or custom maps."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import importlib
import time


_GENERATORS = {
    "forest": ("forest_map_generator", "ForestMapGenerator"),
    "partition": ("partition_map_generator", "PartitionMapGenerator"),
    "dungeon": ("dungeon_map_generator", "DungeonMapGenerator"),
    "hybrid": ("hybrid_terrain_generator", "HybridTerrainGenerator"),
}


def create_generator(kind, **parameters):
    """Load geometry dependencies only when generation is requested."""
    module, class_name = _GENERATORS[kind]
    return getattr(importlib.import_module(module), class_name)(**parameters)


def generate(root, kind, seed, size):
    """Write one map, its EPIC configuration, and a safe starting position."""
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import cKDTree
    import yaml

    out = root / kind
    out.mkdir(parents=True, exist_ok=False)
    common = dict(
        room_size=size, wall_height=3.0, floor_thickness=0.1, output_dir=str(out)
    )
    if kind == "forest":
        params = dict(num_columns=60, min_radius=0.4, max_radius=1.0)
    elif kind == "partition":
        params = dict(
            num_columns=0, num_cubes=25, min_length=2.0, max_length=5.0, half_width=0.3
        )
    else:
        params = dict(
            max_rooms=12,
            min_room_size=5,
            max_room_size=10,
            min_distance=2,
            corridor_width=3,
        )

    generator = create_generator(kind, **common, **params)
    result = generator.export_map(
        {"mesh": kind + ".ply", "pcd": kind + ".pcd", "png": kind + ".png"},
        seed=seed,
        voxel_size=0.1,
        verify_geometry=True,
    )
    pcd = result["contents"]["pcd"]
    original_count = result["original_points"]
    elements = json.dumps(generator.map_elements, sort_keys=True)
    map_path = out / (kind + ".pcd")

    xyz = np.asarray(pcd.points)
    resolution = 0.25
    n = int(size / resolution)
    obstacles = xyz[(xyz[:, 2] > 0.5) & (xyz[:, 2] < 2.8)]
    ij = np.floor(obstacles[:, :2] / resolution).astype(int)
    ij = ij[(ij >= 0).all(axis=1) & (ij < n).all(axis=1)]
    occupied = np.zeros((n, n), dtype=bool)
    occupied[ij[:, 1], ij[:, 0]] = True
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    clearance = ndimage.distance_transform_edt(~occupied) * resolution
    safe = clearance >= 0.8
    components, count = ndimage.label(safe)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    reachable = components == sizes.argmax()
    candidates = np.argwhere(reachable & (clearance >= 1.2))
    if not len(candidates):
        raise ValueError("No safe starting point")
    xy = (candidates[:, ::-1] + 0.5) * resolution
    start_xy = xy[np.argmin(np.linalg.norm(xy - [3, 3], axis=1))]
    start = [float(start_xy[0]), float(start_xy[1]), 1.5]
    nearest = float(cKDTree(xyz).query(start)[0])
    assert nearest >= 0.8, "Unsafe 3D starting point"
    np.savez_compressed(
        out / "coverage_grid.npz",
        occupied=occupied,
        reachable=reachable,
        resolution=resolution,
        clearance=clearance,
    )

    workspace = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            workspace / "src/global_planner/exploration_manager/config/garage.yaml"
        ).read_text()
    )
    config.update(
        {
            "box_0/down": [-0.1, -0.1, -0.2],
            "box_0/up": [size + 0.1, size + 0.1, 2.8],
            "ViewpointManager/sample_pillar_max_height": 2.5,
            "MaxVelMag": 3.0,
        }
    )
    (out / "epic.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    meta = {
        "kind": kind,
        "seed": seed,
        "size_m": size,
        "wall_height_m": 3.0,
        "parameters": params,
        "initial_position": start,
        "initial_clearance_3d_m": nearest,
        "original_points": original_count,
        "points": len(xyz),
        "voxel_size_m": 0.1,
        "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        "safe_free_components": count,
        "reachable_free_area_m2": int(reachable.sum()) * resolution**2,
        "total_safe_free_area_m2": int(safe.sum()) * resolution**2,
        "elements": json.loads(elements),
        "generator_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    (out / "scene.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                k: meta[k]
                for k in [
                    "kind",
                    "initial_position",
                    "points",
                    "reachable_free_area_m2",
                    "safe_free_components",
                ]
            }
        ),
        flush=True,
    )


def _training_starts(scene, initial_position):
    """Keep the original start and add a safe start near the map interior."""
    import numpy as np

    grid = np.load(scene / "coverage_grid.npz")
    ij = np.argwhere(grid["reachable"] & (grid["clearance"] >= 1.5))
    xy = (ij[:, ::-1] + 0.5) * float(grid["resolution"])
    initial = np.array(initial_position)
    # Keep the second start away from outer corners where EPIC can finish prematurely.
    interior = xy[np.linalg.norm(xy - [30.0, 30.0], axis=1).argmin()]
    return [initial.tolist(), [float(interior[0]), float(interior[1]), 1.5]]


def generate_training_scenes(root, maps_per_type, seed):
    """Write forest/partition maps and a two-start-per-map collection manifest."""
    scenes = []
    for i in range(maps_per_type):
        for offset, kind in enumerate(["forest", "partition"]):
            parent = root / f"map_{i:02d}"
            parent.mkdir(exist_ok=True)
            generate(parent, kind, seed + 2 * i + offset, 40)
            scene = parent / kind
            meta = json.loads((scene / "scene.json").read_text())
            for j, point in enumerate(
                _training_starts(scene, meta["initial_position"])
            ):
                scenes.append(
                    dict(
                        name=f"{kind}_{i:02d}_start{j}",
                        map=str(scene / (kind + ".pcd")),
                        config=str(scene / "epic.yaml"),
                        coverage_grid=str(scene / "coverage_grid.npz"),
                        initial_position=point,
                        initial_yaw=0.0
                        if j == 0
                        else math.atan2(20 - point[1], 20 - point[0]),
                    )
                )
            (root / "scenes.json").write_text(json.dumps(scenes, indent=2))
            print("generated", kind, i, flush=True)
    (root / "generation_complete.json").write_text(
        json.dumps(
            {"complete": True, "maps": 2 * maps_per_type, "episodes": len(scenes)}
        )
    )


def add_custom_arguments(parser):
    """Register geometry and export options for standalone maps."""
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for repeatable scene geometry",
    )
    # Basic options.
    parser.add_argument(
        "--map-type",
        type=str,
        choices=["forest", "partition", "dungeon", "hybrid"],
        default="forest",
        help="Map type: forest, partition, dungeon or hybrid",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "output"),
        help="Output root (default: output directory next to this script)",
    )

    # Map dimensions and settings.
    parser.add_argument(
        "--room-size",
        type=int,
        default=80,
        help="Room dimensions (default: 80; hybrid default: 320)",
    )
    parser.add_argument(
        "--floor-thickness",
        type=float,
        default=0.1,
        help="Floor thickness (default: 0.1)",
    )

    # Forest map options.
    parser.add_argument(
        "--wall-height", type=float, default=3, help="Wall/pillar height"
    )
    parser.add_argument(
        "--num-columns", type=int, default=240, help="Pillar/tree count (default: 240)"
    )
    parser.add_argument(
        "--min-radius", type=float, default=0.5, help="Minimum pillar radius"
    )
    parser.add_argument(
        "--max-radius", type=float, default=2, help="Maximum pillar radius"
    )

    # Partition-specific options.
    parser.add_argument(
        "--num-cubes",
        type=int,
        default=81,
        help="Partition count (default: 81; used only for partition maps)",
    )
    parser.add_argument(
        "--min-length", type=float, default=2, help="Minimum partition length"
    )
    parser.add_argument(
        "--max-length", type=float, default=7, help="Maximum partition length"
    )
    parser.add_argument(
        "--half-width", type=float, default=0.5, help="Partition half-width"
    )

    # Dungeon-specific options.
    dungeon_group = parser.add_argument_group("Dungeon options")
    dungeon_group.add_argument(
        "--max-rooms", type=int, default=30, help="[Dungeon] Maximum room count"
    )
    dungeon_group.add_argument(
        "--min-room-size", type=int, default=4, help="[Dungeon] Minimum room dimensions"
    )
    dungeon_group.add_argument(
        "--max-room-size",
        type=int,
        default=12,
        help="[Dungeon] Maximum room dimensions",
    )
    dungeon_group.add_argument(
        "--min-distance", type=int, default=5, help="[Dungeon] Minimum room spacing"
    )
    dungeon_group.add_argument(
        "--corridor-width",
        type=int,
        default=1,
        help="[Dungeon] Corridor width in grid cells",
    )

    # Hybrid-specific options.
    hybrid_group = parser.add_argument_group("Hybrid terrain options")
    hybrid_group.add_argument(
        "--num-regions",
        type=int,
        default=16,
        help="[Hybrid] Region count (default: 16; must be a perfect square)",
    )
    hybrid_group.add_argument(
        "--num-dense-regions",
        type=int,
        default=8,
        help="[Hybrid] Dense-obstacle region count",
    )
    hybrid_group.add_argument(
        "--num-pillars-dense",
        type=int,
        default=120,
        help="[Hybrid] Pillars per dense region",
    )
    hybrid_group.add_argument(
        "--num-pillars-sparse",
        type=int,
        default=20,
        help="[Hybrid] Pillars per sparse region",
    )
    hybrid_group.add_argument(
        "--num-partitions-dense",
        type=int,
        default=60,
        help="[Hybrid] Partitions per dense region",
    )
    hybrid_group.add_argument(
        "--num-partitions-sparse",
        type=int,
        default=10,
        help="[Hybrid] Partitions per sparse region",
    )

    # Output options.
    parser.add_argument(
        "--output-types",
        type=str,
        default="all",
        help="Output formats: 'mesh', 'pcd', 'png', combinations such as 'mesh,pcd', or 'all' (default: all)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: map type and batch index)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of maps to generate (default: 1)",
    )

    # Visualization options.
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize after generation (single models only)",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    test = commands.add_parser("test", help="Generate one held-out map per scene type.")
    test.add_argument("--output", required=True, help="New output directory.")
    test.add_argument(
        "--size", type=int, default=40, help="Map side length in metres (default: 40)."
    )
    test.add_argument(
        "--seed", type=int, default=20260916, help="First map seed (default: 20260916)."
    )

    train = commands.add_parser(
        "train", help="Generate 40-m forest/partition training maps."
    )
    train.add_argument("--output", required=True, help="New output directory.")
    train.add_argument(
        "--maps-per-type",
        type=int,
        default=10,
        help="Maps per scene type (default: 10).",
    )
    train.add_argument(
        "--seed", type=int, default=917000, help="First map seed (default: 917000)."
    )

    custom = commands.add_parser(
        "custom", help="Generate custom forest, partition, dungeon, or hybrid maps."
    )
    add_custom_arguments(custom)
    args = parser.parse_args(argv)
    if args.command == "test" and args.size < 1:
        parser.error("--size must be positive")
    if args.command == "train" and args.maps_per_type < 1:
        parser.error("--maps-per-type must be positive")
    if args.command == "custom":
        if args.map_type == "hybrid" and args.room_size == 80:
            args.room_size = 320
        args.formats = (
            ["mesh", "pcd", "png"]
            if args.output_types == "all"
            else [fmt.strip() for fmt in args.output_types.split(",")]
        )
        if not args.formats or any(
            fmt not in {"mesh", "pcd", "png"} for fmt in args.formats
        ):
            parser.error("--output-types must be mesh, pcd, png, or all")
        if len(set(args.formats)) != len(args.formats):
            parser.error("--output-types must not repeat a format")
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        if args.prefix and (
            Path(args.prefix).name != args.prefix or args.prefix in {".", ".."}
        ):
            parser.error("--prefix must be a basename")
    return args


def generate_custom(args):
    """Use the shared batch exporter for user-selected geometry and formats."""
    parameters = dict(
        room_size=args.room_size,
        wall_height=args.wall_height,
        floor_thickness=args.floor_thickness,
        output_dir=str(Path(args.output_dir) / args.map_type),
    )
    if args.map_type == "forest":
        parameters.update(
            num_columns=args.num_columns,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
        )
    elif args.map_type == "partition":
        parameters.update(
            num_columns=args.num_columns,
            num_cubes=args.num_cubes,
            min_length=args.min_length,
            max_length=args.max_length,
            half_width=args.half_width,
        )
    elif args.map_type == "dungeon":
        parameters.update(
            max_rooms=args.max_rooms,
            min_room_size=args.min_room_size,
            max_room_size=args.max_room_size,
            min_distance=args.min_distance,
            corridor_width=args.corridor_width,
        )
    else:
        parameters.update(
            num_regions=args.num_regions,
            num_dense_regions=args.num_dense_regions,
            num_pillars_dense=args.num_pillars_dense,
            num_pillars_sparse=args.num_pillars_sparse,
            num_partitions_dense=args.num_partitions_dense,
            num_partitions_sparse=args.num_partitions_sparse,
            min_pillar_radius=args.min_radius,
            max_pillar_radius=args.max_radius,
            min_partition_length=args.min_length,
            max_partition_length=args.max_length,
        )
    generator = create_generator(args.map_type, **parameters)
    started = time.monotonic()
    generator.generate_batch(
        args.formats,
        prefix=args.prefix,
        batch_size=args.batch_size,
        seed=args.seed,
        map_name=args.map_type,
        visualize=args.visualize,
    )
    print(f"Completed in {time.monotonic() - started:.2f}s")


def main(argv=None):
    args = parse_args(argv)
    if args.command == "custom":
        generate_custom(args)
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.command == "test":
        for index, kind in enumerate(["forest", "partition", "dungeon"]):
            generate(root, kind, args.seed + index, args.size)
    else:
        generate_training_scenes(root, args.maps_per_type, args.seed)


if __name__ == "__main__":
    main()
