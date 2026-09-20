import json
import os
from pathlib import Path
import random
import numpy as np
import open3d as o3d
import uuid
from datetime import datetime


class BaseMapGenerator:
    def __init__(
        self, room_size=80, wall_height=3, floor_thickness=0.1, output_dir="output"
    ):
        """
        Base class for map generators with shared utilities.

        Parameters:
        - room_size: Map dimensions.
        - wall_height: Wall height.
        - floor_thickness: Floor thickness.
        - output_dir: Output directory.
        """
        self.room_size = room_size
        self.wall_height = wall_height
        self.floor_thickness = floor_thickness
        self.output_dir = output_dir

        # Ensure the output directory exists.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save random state to keep geometry consistent across output formats.
        self.random_state = None

        # Store map elements for reuse across output formats.
        self.map_elements = {}

    def generate_mesh(self):
        """Generate a 3D mesh; implemented by subclasses."""
        # Save the current random state for the other output formats.
        self.random_state = np.random.get_state()
        # Clear cached map elements.
        self._clear_map_elements()

        raise NotImplementedError("Subclass must implement abstract method")

    def _clear_map_elements(self):
        """Clear cached elements; subclasses add their own element types."""
        self.random_state = None
        self.map_elements = {}

    def generate_pcd(self, mesh=None):
        """Generate points directly rather than sampling a mesh; implemented by subclasses."""
        # Restore random state to match mesh generation.
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            # Generate map elements first if no mesh has been generated.
            self._generate_map_elements()

        raise NotImplementedError("Subclass must implement abstract method")

    def _generate_map_elements(self):
        """Generate map elements; implemented by subclasses."""
        # Prepare elements when generate_pcd or generate_png is called directly.
        raise NotImplementedError("Subclass must implement abstract method")

    def generate_png(self, size=(800, 800)):
        """Generate a 2D top-down image; implemented by subclasses."""
        # Restore random state to match mesh generation.
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            # Generate map elements first if no mesh has been generated.
            self._generate_map_elements()

        raise NotImplementedError("Subclass must implement abstract method")

    def save_mesh(self, mesh, filename=None):
        """Save the mesh to a file."""
        if filename is None:
            filename = self._get_filename("mesh", "ply")

        full_path = os.path.join(self.output_dir, filename)
        if not o3d.io.write_triangle_mesh(full_path, mesh):
            raise OSError(f"Could not write mesh: {full_path}")
        return full_path

    def save_pcd(self, pcd, filename=None):
        """Save the point cloud to a file."""
        if filename is None:
            filename = self._get_filename("pcd", "pcd")

        full_path = os.path.join(self.output_dir, filename)
        if not o3d.io.write_point_cloud(full_path, pcd):
            raise OSError(f"Could not write point cloud: {full_path}")
        return full_path

    def save_png(self, img, filename=None):
        """Save the PNG image to a file."""
        if filename is None:
            filename = self._get_filename("png", "png")

        full_path = os.path.join(self.output_dir, filename)
        img.save(full_path)
        return full_path

    def _get_filename(self, type_str, extension):
        """Generate a unique filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{self.get_map_type()}_{type_str}_{timestamp}_{unique_id}.{extension}"

    def get_map_type(self):
        """Return the map type name; implemented by subclasses."""
        raise NotImplementedError("Subclass must implement abstract method")

    def export_map(
        self,
        filenames,
        *,
        seed=None,
        initialize_mesh=True,
        voxel_size=None,
        verify_geometry=False,
    ):
        """Generate and write one scene, returning paths and generated content.

        ``filenames`` maps output formats to basenames in generation order. A mesh
        normally initializes shared geometry even when only PCD or PNG is saved.
        ``original_points`` records the point count before optional downsampling.
        """
        if not filenames or any(fmt not in {"mesh", "pcd", "png"} for fmt in filenames):
            raise ValueError("Output formats must be mesh, pcd, or png")
        for name in filenames.values():
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Output filenames must be basenames")
        targets = {fmt: Path(self.output_dir) / name for fmt, name in filenames.items()}
        if any(path.exists() for path in targets.values()):
            raise FileExistsError("Output exists; choose a new directory or prefix")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        contents = {}
        if initialize_mesh:
            contents["mesh"] = self.generate_mesh()
        else:
            self._clear_map_elements()
            if "mesh" not in filenames:
                self._generate_map_elements()
        elements = (
            json.dumps(self.map_elements, sort_keys=True) if verify_geometry else None
        )
        for fmt in filenames:
            if fmt not in contents:
                contents[fmt] = getattr(self, "generate_" + fmt)()
        if (
            verify_geometry
            and json.dumps(self.map_elements, sort_keys=True) != elements
        ):
            raise ValueError("Formats changed scene geometry")

        original_points = None
        if "pcd" in contents:
            original_points = len(contents["pcd"].points)
            if voxel_size is not None:
                contents["pcd"] = contents["pcd"].voxel_down_sample(voxel_size)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        paths = {}
        for fmt, name in filenames.items():
            paths[fmt] = getattr(self, "save_" + fmt)(contents[fmt], name)
        return {
            "paths": paths,
            "contents": contents,
            "original_points": original_points,
        }

    def generate_batch(
        self,
        formats=("mesh", "pcd", "png"),
        prefix=None,
        batch_size=1,
        seed=None,
        *,
        filename_factory=None,
        subdirectories=True,
        initialize_mesh=True,
        reseed_unseeded=False,
        visualize=False,
        map_name=None,
    ):
        """Export batches through one seed-aware path and restore the output root.

        A supplied seed advances by one per map. Optional filename and directory
        policies preserve the established Python batch helpers below.
        """
        formats = tuple(formats)
        if not formats or len(set(formats)) != len(formats):
            raise ValueError("Output formats must be nonempty and unique")
        suffixes = {"mesh": "_mesh.ply", "pcd": "_pcd.pcd", "png": "_map.png"}
        if any(fmt not in suffixes for fmt in formats):
            raise ValueError("Output formats must be mesh, pcd, or png")
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        original_output_dir = self.output_dir
        paths = {fmt: [] for fmt in formats}
        try:
            for index in range(batch_size):
                self.output_dir = (
                    str(Path(original_output_dir) / f"batch_{index + 1}")
                    if (subdirectories and batch_size > 1)
                    else original_output_dir
                )
                if filename_factory is None:
                    name = prefix or f"{map_name or self.get_map_type()}_{index}"
                    filenames = {fmt: name + suffixes[fmt] for fmt in formats}
                else:
                    filenames = filename_factory(index)
                # Unseeded legacy helpers retain their original per-map reseeding.
                if seed is None and reseed_unseeded and batch_size > 1:
                    np.random.seed(int(datetime.now().timestamp()) + index)
                result = self.export_map(
                    filenames,
                    seed=None if seed is None else seed + index,
                    initialize_mesh=initialize_mesh,
                )
                for fmt, path in result["paths"].items():
                    paths[fmt].append(path)
                    print(path, flush=True)
                if visualize and batch_size == 1:
                    o3d.visualization.draw_geometries([result["contents"]["mesh"]])
        finally:
            self.output_dir = original_output_dir
        return paths

    def _single_format_batch(self, fmt, prefix, batch_size, seed):
        extension = {"mesh": "ply", "pcd": "pcd", "png": "png"}[fmt]

        def filenames(index):
            name = f"{prefix}_{index}" if prefix else str(index)
            return {fmt: f"{name}.{extension}"}

        return self.generate_batch(
            (fmt,),
            batch_size=batch_size,
            seed=seed,
            filename_factory=filenames,
            subdirectories=False,
            initialize_mesh=False,
            reseed_unseeded=True,
        )[fmt]

    def generate_mesh_batch(self, prefix=None, batch_size=1, seed=None):
        """Generate mesh files, preserving the prefix/index filename convention."""
        return self._single_format_batch("mesh", prefix, batch_size, seed)

    def generate_pcd_batch(self, prefix=None, batch_size=1, seed=None):
        """Generate point clouds, preserving the prefix/index filename convention."""
        return self._single_format_batch("pcd", prefix, batch_size, seed)

    def generate_png_batch(self, prefix=None, batch_size=1, seed=None):
        """Generate top-down images, preserving the prefix/index filename convention."""
        return self._single_format_batch("png", prefix, batch_size, seed)

    def generate_all_batch(self, batch_size=1, seed=None):
        """Generate all formats using the established index-based filenames."""
        return self.generate_batch(
            batch_size=batch_size,
            seed=seed,
            reseed_unseeded=True,
            filename_factory=lambda index: {
                "mesh": f"{index}_mesh.ply",
                "pcd": f"{index}_pcd.pcd",
                "png": f"{index}_map.png",
            },
        )

    def generate_all(self, prefix=None, batch_size=1, seed=None):
        """Generate all formats using a shared prefix or timestamp-based name."""

        def filenames(index):
            name = prefix or f"{self.get_map_type()}_{datetime.now():%Y%m%d_%H%M%S}"
            return {
                "mesh": name + "_mesh.ply",
                "png": name + "_map.png",
                "pcd": name + "_pcd.pcd",
            }

        return self.generate_batch(
            ("mesh", "png", "pcd"),
            batch_size=batch_size,
            seed=seed,
            filename_factory=filenames,
            reseed_unseeded=True,
        )
