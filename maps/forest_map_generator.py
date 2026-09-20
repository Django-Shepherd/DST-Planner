import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from base_map_generator import BaseMapGenerator


class ForestMapGenerator(BaseMapGenerator):
    def __init__(
        self,
        room_size=80,
        wall_height=3,
        num_columns=240,
        min_radius=0.5,
        max_radius=2.0,
        floor_thickness=0.1,
        distribution="uniform",
        output_dir="output/forest",
    ):
        """
        Forest map generator with randomly placed trees.

        Parameters:
        - room_size: Map dimensions.
        - wall_height: Wall height.
        - num_columns: Number of trees.
        - min_radius: Minimum tree radius.
        - max_radius: Maximum tree radius.
        - floor_thickness: Floor thickness.
        - distribution: Distribution mode ('random' or 'uniform').
        - output_dir: Output directory.
        """
        super().__init__(room_size, wall_height, floor_thickness, output_dir)
        self.num_columns = num_columns
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.distribution = distribution

    def _clear_map_elements(self):
        """Initialize map elements."""
        self.map_elements = {"columns": []}

    def get_map_type(self):
        """Return the map type identifier."""
        return "forest"

    def generate_mesh(self):
        """Generate a forest mesh."""
        self.random_state = np.random.get_state()
        self._clear_map_elements()

        combined_mesh = o3d.geometry.TriangleMesh()

        # 1. Floor.
        floor_box = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=self.room_size, depth=self.floor_thickness
        )
        floor_box.translate((0, 0, 0))
        floor_color = [0.824, 0.706, 0.549]
        floor_box.paint_uniform_color(floor_color)
        combined_mesh += floor_box

        # 2. Four perimeter walls.
        wall_thickness = 0.1
        wall_color = [0.878, 0.949, 0.976]

        left_wall = o3d.geometry.TriangleMesh.create_box(
            width=wall_thickness, height=self.room_size, depth=self.wall_height
        )
        left_wall.translate((0, 0, 0))
        left_wall.paint_uniform_color(wall_color)
        combined_mesh += left_wall

        right_wall = o3d.geometry.TriangleMesh.create_box(
            width=wall_thickness, height=self.room_size, depth=self.wall_height
        )
        right_wall.translate((self.room_size - wall_thickness, 0, 0))
        right_wall.paint_uniform_color(wall_color)
        combined_mesh += right_wall

        front_wall = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=wall_thickness, depth=self.wall_height
        )
        front_wall.translate((0, 0, 0))
        front_wall.paint_uniform_color(wall_color)
        combined_mesh += front_wall

        back_wall = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=wall_thickness, depth=self.wall_height
        )
        back_wall.translate((0, self.room_size - wall_thickness, 0))
        back_wall.paint_uniform_color(wall_color)
        combined_mesh += back_wall

        # 3. Pillars representing trees.
        column_color_options = [
            [0.545, 0.271, 0.075],
        ]

        if self.distribution == "uniform":
            # Use an adaptive rectangular grid for uniform placement.
            cols = int(np.sqrt(self.num_columns * 1.2))
            rows = int(np.ceil(self.num_columns / cols))

            while cols * rows < self.num_columns:
                cols += 1
                rows = int(np.ceil(self.num_columns / cols))

            margin = self.max_radius + 0.2
            available_width = self.room_size - 2 * margin
            available_height = self.room_size - 2 * margin

            cell_width = available_width / cols
            cell_height = available_height / rows

            # Print grid information only once.
            if not hasattr(self, "_grid_info_printed"):
                print(f"\n{'=' * 60}")
                print(f"[Forest Generator] Map configuration:")
                print(f"  Room dimensions: {self.room_size}m x {self.room_size}m")
                print(f"  Pillar count: {self.num_columns}")
                print(
                    f"  Pillar diameter: {self.min_radius * 2:.1f}m - {self.max_radius * 2:.1f}m"
                )
                print(f"  Boundary margin: {margin:.1f}m")
                print(
                    f"  Available space: {available_width:.1f}m x {available_height:.1f}m"
                )
                print(f"\n[Forest Generator] Grid layout:")
                print(
                    f"  Grid dimensions: {cols} columns x {rows} rows = {cols * rows} cells"
                )
                print(f"  Used cells: {self.num_columns} / {cols * rows}")
                print(f"  Empty cells: {cols * rows - self.num_columns}")
                print(f"  Cell dimensions: {cell_width:.2f}m x {cell_height:.2f}m")
                print(
                    f"  Position jitter: +/-{cell_width * 0.4:.2f}m (x), +/-{cell_height * 0.4:.2f}m (y)"
                )
                print(f"{'=' * 60}\n")
                self._grid_info_printed = True

            # Generate all grid positions, including cells that may remain empty.
            all_positions = []
            for i in range(cols * rows):
                row = i // cols
                col = i % cols

                base_x = margin + col * cell_width + cell_width / 2
                base_y = margin + row * cell_height + cell_height / 2

                all_positions.append((row, col, base_x, base_y))

            # Shuffle positions so unused cells are distributed uniformly.
            np.random.shuffle(all_positions)

            # Use only the first num_columns positions.
            for i in range(self.num_columns):
                row, col, base_x, base_y = all_positions[i]

                jitter_x = cell_width * 0.4
                jitter_y = cell_height * 0.4
                center_x = base_x + np.random.uniform(-jitter_x, jitter_x)
                center_y = base_y + np.random.uniform(-jitter_y, jitter_y)

                radius = np.random.uniform(self.min_radius, self.max_radius)

                min_coord = radius + 0.15
                max_coord = self.room_size - radius - 0.15
                center_x = np.clip(center_x, min_coord, max_coord)
                center_y = np.clip(center_y, min_coord, max_coord)
                height_col = self.wall_height

                self.map_elements["columns"].append(
                    {
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius": radius,
                        "height": height_col,
                    }
                )

                cyl = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=radius, height=height_col, resolution=30, split=1
                )
                cyl.translate((center_x, center_y, height_col / 2))

                color_idx = np.random.randint(0, len(column_color_options))
                cyl.paint_uniform_color(column_color_options[color_idx])

                combined_mesh += cyl

        else:
            # Fully random placement.
            for i in range(self.num_columns):
                radius = np.random.uniform(self.min_radius, self.max_radius)
                center_x = np.random.uniform(radius, self.room_size - radius)
                center_y = np.random.uniform(radius, self.room_size - radius)
                height_col = self.wall_height

                self.map_elements["columns"].append(
                    {
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius": radius,
                        "height": height_col,
                    }
                )

                cyl = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=radius, height=height_col, resolution=30, split=1
                )
                cyl.translate((center_x, center_y, height_col / 2))

                color_idx = np.random.randint(0, len(column_color_options))
                cyl.paint_uniform_color(column_color_options[color_idx])

                combined_mesh += cyl

        return combined_mesh

    def _generate_map_elements(self):
        """Generate forest elements when no mesh has been generated yet."""
        self.map_elements = {"columns": []}

        self.random_state = np.random.get_state()

        if self.distribution == "uniform":
            # Use the same uniform grid as generate_mesh.
            cols = int(np.sqrt(self.num_columns * 1.2))
            rows = int(np.ceil(self.num_columns / cols))

            while cols * rows < self.num_columns:
                cols += 1
                rows = int(np.ceil(self.num_columns / cols))

            margin = self.max_radius + 0.2
            available_width = self.room_size - 2 * margin
            available_height = self.room_size - 2 * margin

            cell_width = available_width / cols
            cell_height = available_height / rows

            # Generate all grid positions and shuffle them.
            all_positions = []
            for i in range(cols * rows):
                row = i // cols
                col = i % cols

                base_x = margin + col * cell_width + cell_width / 2
                base_y = margin + row * cell_height + cell_height / 2

                all_positions.append((row, col, base_x, base_y))

            np.random.shuffle(all_positions)

            for i in range(self.num_columns):
                row, col, base_x, base_y = all_positions[i]

                jitter_x = cell_width * 0.4
                jitter_y = cell_height * 0.4
                center_x = base_x + np.random.uniform(-jitter_x, jitter_x)
                center_y = base_y + np.random.uniform(-jitter_y, jitter_y)

                radius = np.random.uniform(self.min_radius, self.max_radius)
                min_coord = radius + 0.15
                max_coord = self.room_size - radius - 0.15
                center_x = np.clip(center_x, min_coord, max_coord)
                center_y = np.clip(center_y, min_coord, max_coord)
                height_col = self.wall_height

                self.map_elements["columns"].append(
                    {
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius": radius,
                        "height": height_col,
                    }
                )
        else:
            # Fully random placement.
            for i in range(self.num_columns):
                radius = np.random.uniform(self.min_radius, self.max_radius)
                center_x = np.random.uniform(radius, self.room_size - radius)
                center_y = np.random.uniform(radius, self.room_size - radius)
                height_col = self.wall_height

                self.map_elements["columns"].append(
                    {
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius": radius,
                        "height": height_col,
                    }
                )

    def generate_pcd(self, mesh=None):
        """Generate points directly without sampling a mesh."""
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            self._generate_map_elements()

        pcd = o3d.geometry.PointCloud()
        points = []

        # 1. Floor points.
        floor_density = 200
        num_points_x = int(self.room_size * np.sqrt(floor_density))
        num_points_y = int(self.room_size * np.sqrt(floor_density))

        x_coords = np.linspace(0, self.room_size, num_points_x)
        y_coords = np.linspace(0, self.room_size, num_points_y)

        for x in x_coords:
            for y in y_coords:
                points.append([x, y, 0])

        # 2. Wall points.
        wall_thickness = 0.1
        wall_density = 200
        num_points_long = int(self.room_size * np.sqrt(wall_density))
        num_points_thick = max(2, int(wall_thickness * np.sqrt(wall_density)))
        num_points_height = int(self.wall_height * np.sqrt(wall_density))

        # Left wall.
        for y in np.linspace(0, self.room_size, num_points_long):
            for x in np.linspace(0, wall_thickness, num_points_thick):
                for z in np.linspace(0, self.wall_height, num_points_height):
                    points.append([x, y, z])

        # Right wall.
        for y in np.linspace(0, self.room_size, num_points_long):
            for x in np.linspace(
                self.room_size - wall_thickness, self.room_size, num_points_thick
            ):
                for z in np.linspace(0, self.wall_height, num_points_height):
                    points.append([x, y, z])

        # Front wall.
        for x in np.linspace(0, self.room_size, num_points_long):
            for y in np.linspace(0, wall_thickness, num_points_thick):
                for z in np.linspace(0, self.wall_height, num_points_height):
                    points.append([x, y, z])

        # Back wall.
        for x in np.linspace(0, self.room_size, num_points_long):
            for y in np.linspace(
                self.room_size - wall_thickness, self.room_size, num_points_thick
            ):
                for z in np.linspace(0, self.wall_height, num_points_height):
                    points.append([x, y, z])

        # 3. Pillar/tree points.
        surface_density = 40000

        for column in self.map_elements["columns"]:
            center_x = column["center_x"]
            center_y = column["center_y"]
            radius = column["radius"]
            height = column["height"]

            surface_area = 2 * np.pi * radius * radius + 2 * np.pi * radius * height
            num_points = int(surface_area * surface_density / 15)

            side_ratio = (2 * np.pi * radius * height) / surface_area
            side_points = int(num_points * side_ratio)

            for _ in range(side_points):
                theta = np.random.uniform(0, 2 * np.pi)
                h = np.random.uniform(0, height)

                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)
                z = h

                points.append([x, y, z])

            cap_points = num_points - side_points
            for _ in range(cap_points):
                r = np.sqrt(np.random.uniform(0, 1)) * radius
                theta = np.random.uniform(0, 2 * np.pi)

                x = center_x + r * np.cos(theta)
                y = center_y + r * np.sin(theta)

                z = height if np.random.random() > 0.5 else 0

                points.append([x, y, z])

        if points:
            points_array = np.vstack(points)
            pcd.points = o3d.utility.Vector3dVector(points_array)

            colors = np.zeros((points_array.shape[0], 3))

            floor_mask = points_array[:, 2] < self.floor_thickness
            colors[floor_mask] = [0.824, 0.706, 0.549]

            wall_mask = (points_array[:, 2] >= self.floor_thickness) & (
                (points_array[:, 0] < wall_thickness)
                | (points_array[:, 0] > self.room_size - wall_thickness)
                | (points_array[:, 1] < wall_thickness)
                | (points_array[:, 1] > self.room_size - wall_thickness)
            )
            colors[wall_mask] = [0.878, 0.949, 0.976]

            column_mask = ~(floor_mask | wall_mask)
            colors[column_mask] = [0.545, 0.271, 0.075]

            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def generate_png(self, size=(800, 800), flip_vertical=True):
        """Generate a 2D top-down image."""
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            self._generate_map_elements()

        img = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(img)

        scale_x = size[0] / self.room_size
        scale_y = size[1] / self.room_size

        border_width = 3
        draw.rectangle(
            [(0, 0), (size[0] - 1, size[1] - 1)], outline="black", width=border_width
        )

        wall_thickness = 0.1
        wall_width = max(1, int(wall_thickness * scale_x))

        draw.rectangle([(0, 0), (size[0] - 1, wall_width)], fill="lightblue")
        draw.rectangle(
            [(0, size[1] - wall_width), (size[0] - 1, size[1] - 1)], fill="lightblue"
        )
        draw.rectangle([(0, 0), (wall_width, size[1] - 1)], fill="lightblue")
        draw.rectangle(
            [(size[0] - wall_width, 0), (size[0] - 1, size[1] - 1)], fill="lightblue"
        )

        for column in self.map_elements["columns"]:
            x = column["center_x"]
            y = column["center_y"]
            radius = column["radius"]

            ix = int(x * scale_x)
            iy = int(y * scale_y)
            ir = max(1, int(radius * scale_x))

            draw.ellipse(
                [(ix - ir, iy - ir), (ix + ir, iy + ir)], fill="brown", outline="black"
            )

        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img
