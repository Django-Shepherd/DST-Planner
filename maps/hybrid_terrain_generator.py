import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import random
import math
from base_map_generator import BaseMapGenerator


class HybridTerrainGenerator(BaseMapGenerator):
    def __init__(
        self,
        room_size=320,  # Large map dimensions.
        wall_height=3,
        floor_thickness=0.1,
        num_regions=16,  # Number of regions in a 4x4 grid.
        num_dense_regions=5,  # Number of dense-obstacle regions.
        num_pillars_dense=120,  # Pillars per dense region at reduced density.
        num_pillars_sparse=20,  # Pillars per sparse region at substantially reduced density.
        num_partitions_dense=60,  # Partitions per dense region at reduced density.
        num_partitions_sparse=10,  # Partitions per sparse region at substantially reduced density.
        min_pillar_radius=0.5,
        max_pillar_radius=2.0,
        min_partition_length=2,
        max_partition_length=7,
        output_dir="output/hybrid",
    ):
        """
        Generate large hybrid terrain maps with nonuniform obstacle density.

        Parameters:
        - room_size: Map dimensions.
        - wall_height: Wall and obstacle height.
        - floor_thickness: Floor thickness.
        - num_regions: Number of regions in an n-by-n grid.
        - num_dense_regions: Number of dense-obstacle regions.
        - num_pillars_dense: Pillars per dense region.
        - num_pillars_sparse: Pillars per sparse region.
        - num_partitions_dense: Partitions per dense region.
        - num_partitions_sparse: Partitions per sparse region.
        - min_pillar_radius: Minimum pillar radius.
        - max_pillar_radius: Maximum pillar radius.
        - min_partition_length: Minimum partition length.
        - max_partition_length: Maximum partition length.
        - output_dir: Output directory.
        """
        super().__init__(room_size, wall_height, floor_thickness, output_dir)
        self.num_regions = num_regions
        self.num_dense_regions = num_dense_regions
        self.num_pillars_dense = num_pillars_dense
        self.num_pillars_sparse = num_pillars_sparse
        self.num_partitions_dense = num_partitions_dense
        self.num_partitions_sparse = num_partitions_sparse
        self.min_pillar_radius = min_pillar_radius
        self.max_pillar_radius = max_pillar_radius
        self.min_partition_length = min_partition_length
        self.max_partition_length = max_partition_length

        # Compute region dimensions.
        self.region_size = room_size / np.sqrt(num_regions)

    def _clear_map_elements(self):
        """Initialize map elements."""
        self.map_elements = {
            "regions": [],  # Store region types and positions.
            "pillars": [],  # Store pillar positions and dimensions.
            "partitions": [],  # Store partition positions and dimensions.
        }

        # Initialize region assignment flags.
        self.region_types = np.zeros(
            (int(np.sqrt(self.num_regions)), int(np.sqrt(self.num_regions))), dtype=int
        )
        # 0: unassigned; 1: dense obstacles; 2: sparse obstacles.

    def get_map_type(self):
        """Return the map type identifier."""
        return "hybrid_terrain"

    def _generate_map_elements(self):
        """Generate map elements."""
        # Clear map elements.
        self._clear_map_elements()

        # Set random state.
        self.random_state = np.random.get_state()

        # 1. Assign region types.
        region_grid_size = int(np.sqrt(self.num_regions))

        # Create dense-obstacle regions.
        dense_regions = set()
        for _ in range(self.num_dense_regions):
            # Randomly select an unassigned region.
            while True:
                region_i = np.random.randint(0, region_grid_size)
                region_j = np.random.randint(0, region_grid_size)

                if (region_i, region_j) not in dense_regions:
                    break

            # Mark the region as dense.
            self.region_types[region_i, region_j] = 1
            dense_regions.add((region_i, region_j))

            # Store region information.
            self.map_elements["regions"].append(
                {"type": "dense", "region_i": region_i, "region_j": region_j}
            )

        # Mark remaining regions as sparse.
        for i in range(region_grid_size):
            for j in range(region_grid_size):
                if self.region_types[i, j] == 0:  # Unassigned.
                    self.region_types[i, j] = 2  # Sparse obstacles.

                    # Store region information.
                    self.map_elements["regions"].append(
                        {"type": "sparse", "region_i": i, "region_j": j}
                    )

        # 2. Generate obstacles.
        # Generate all pillars first.
        self._generate_pillars()

        # Generate partitions next.
        self._generate_partitions()

    def _generate_pillars(self):
        """Generate pillars in each region."""
        region_grid_size = int(np.sqrt(self.num_regions))

        # Visit each region.
        for i in range(region_grid_size):
            for j in range(region_grid_size):
                region_type = self.region_types[i, j]

                # Compute region bounds.
                x_min = i * self.region_size
                x_max = (i + 1) * self.region_size
                y_min = j * self.region_size
                y_max = (j + 1) * self.region_size

                # Choose the pillar count for this region type.
                if region_type == 1:  # Dense-obstacle region.
                    num_pillars = self.num_pillars_dense
                else:  # Sparse-obstacle region.
                    num_pillars = self.num_pillars_sparse

                # Generate pillars in this region.
                for _ in range(num_pillars):
                    # Sample pillar dimensions.
                    radius = np.random.uniform(
                        self.min_pillar_radius, self.max_pillar_radius
                    )

                    # Sample pillar positions inside the region.
                    center_x = np.random.uniform(x_min + radius, x_max - radius)
                    center_y = np.random.uniform(y_min + radius, y_max - radius)

                    # Randomize pillar height.
                    height_col = np.random.uniform(
                        self.wall_height, self.wall_height * 1.5
                    )

                    # Store pillar information.
                    self.map_elements["pillars"].append(
                        {
                            "center_x": center_x,
                            "center_y": center_y,
                            "radius": radius,
                            "height": height_col,
                            "region_i": i,
                            "region_j": j,
                        }
                    )

    def _generate_partitions(self):
        """Generate partitions in each region."""
        region_grid_size = int(np.sqrt(self.num_regions))

        # Visit each region.
        for i in range(region_grid_size):
            for j in range(region_grid_size):
                region_type = self.region_types[i, j]

                # Compute region bounds.
                x_min = i * self.region_size
                x_max = (i + 1) * self.region_size
                y_min = j * self.region_size
                y_max = (j + 1) * self.region_size

                # Choose the partition count for this region type.
                if region_type == 1:  # Dense-obstacle region.
                    num_partitions = self.num_partitions_dense
                else:  # Sparse-obstacle region.
                    num_partitions = self.num_partitions_sparse

                # Generate partitions in this region.
                for p in range(num_partitions):
                    # Sample partition dimensions.
                    half_length = np.random.uniform(
                        self.min_partition_length, self.max_partition_length
                    )
                    half_width = 0.5  # Fixed width.
                    height = self.wall_height

                    # Sample partition positions inside the region.
                    center_x = np.random.uniform(
                        x_min + half_length, x_max - half_length
                    )
                    center_y = np.random.uniform(
                        y_min + half_length, y_max - half_length
                    )

                    # Randomly choose the partition orientation.
                    is_horizontal = p % 2 == 0

                    # Store partition information.
                    self.map_elements["partitions"].append(
                        {
                            "center_x": center_x,
                            "center_y": center_y,
                            "half_length": half_length,
                            "half_width": half_width,
                            "height": height,
                            "is_horizontal": is_horizontal,
                            "region_i": i,
                            "region_j": j,
                        }
                    )

    def generate_mesh(self):
        """Generate a hybrid terrain mesh."""
        # Save the random state.
        self.random_state = np.random.get_state()

        # Clear existing elements and generate new ones.
        self._generate_map_elements()

        # Create the mesh that will contain all components.
        combined_mesh = o3d.geometry.TriangleMesh()

        # 1. Floor.
        floor_box = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=self.room_size, depth=self.floor_thickness
        )
        floor_box.translate((0, 0, 0))

        # Assign floor colours.
        floor_color = [0.824, 0.706, 0.549]  # Light wood colour.
        floor_box.paint_uniform_color(floor_color)
        combined_mesh += floor_box

        # 2. Four perimeter walls.
        wall_thickness = 1.0
        wall_color = [0.8, 0.8, 0.8]  # Light grey walls.

        # a) Left wall.
        left_wall = o3d.geometry.TriangleMesh.create_box(
            width=wall_thickness, height=self.room_size, depth=self.wall_height
        )
        left_wall.translate((0, 0, 0))
        left_wall.paint_uniform_color(wall_color)
        combined_mesh += left_wall

        # b) Right wall.
        right_wall = o3d.geometry.TriangleMesh.create_box(
            width=wall_thickness, height=self.room_size, depth=self.wall_height
        )
        right_wall.translate((self.room_size - wall_thickness, 0, 0))
        right_wall.paint_uniform_color(wall_color)
        combined_mesh += right_wall

        # c) Front wall.
        front_wall = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=wall_thickness, depth=self.wall_height
        )
        front_wall.translate((0, 0, 0))
        front_wall.paint_uniform_color(wall_color)
        combined_mesh += front_wall

        # d) Back wall.
        back_wall = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=wall_thickness, depth=self.wall_height
        )
        back_wall.translate((0, self.room_size - wall_thickness, 0))
        back_wall.paint_uniform_color(wall_color)
        combined_mesh += back_wall

        # 3. Generate pillars.
        column_color = [0.545, 0.271, 0.075]  # Reddish brown.

        for pillar in self.map_elements["pillars"]:
            center_x = pillar["center_x"]
            center_y = pillar["center_y"]
            radius = pillar["radius"]
            height = pillar["height"]

            cyl = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, height=height, resolution=30, split=1
            )
            # Place the bottom of each pillar flush with the floor.
            cyl.translate((center_x, center_y, height / 2))
            cyl.paint_uniform_color(column_color)
            combined_mesh += cyl

        # 4. Generate partitions.
        partition_color = [0.545, 0.271, 0.075]  # Reddish brown.

        for partition in self.map_elements["partitions"]:
            center_x = partition["center_x"]
            center_y = partition["center_y"]
            half_length = partition["half_length"]
            half_width = partition["half_width"]
            height = partition["height"]
            is_horizontal = partition["is_horizontal"]

            if is_horizontal:  # Horizontal partition.
                width = half_length * 2
                depth = half_width * 2
                partition_box = o3d.geometry.TriangleMesh.create_box(
                    width=width, height=depth, depth=height
                )
                partition_box.translate(
                    (center_x - half_length, center_y - half_width, 0)
                )
            else:  # Vertical partition.
                width = half_width * 2
                depth = half_length * 2
                partition_box = o3d.geometry.TriangleMesh.create_box(
                    width=width, height=depth, depth=height
                )
                partition_box.translate(
                    (center_x - half_width, center_y - half_length, 0)
                )

            partition_box.paint_uniform_color(partition_color)
            combined_mesh += partition_box

        # Remove duplicate vertices and triangles.
        combined_mesh.remove_duplicated_vertices()
        combined_mesh.remove_duplicated_triangles()
        combined_mesh.compute_vertex_normals()

        return combined_mesh

    def generate_pcd(self, mesh=None):
        """Generate point-cloud data directly."""
        # Reuse random state to match the mesh geometry.
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            # Generate map elements if random state has not been initialized.
            self._generate_map_elements()

        # Create a point cloud.
        pcd = o3d.geometry.PointCloud()
        points = []

        # 1. Floor points.
        floor_density = 100  # Points per unit area; reduce density for larger maps.
        num_points_x = int(
            self.room_size * np.sqrt(floor_density) / 10
        )  # Reduce the sampling rate.
        num_points_y = int(self.room_size * np.sqrt(floor_density) / 10)

        # Uniformly sample floor points.
        x_coords = np.linspace(0, self.room_size, num_points_x)
        y_coords = np.linspace(0, self.room_size, num_points_y)

        for x in x_coords:
            for y in y_coords:
                # Add a point if it is in open space.
                points.append([x, y, 0])  # z=0 is the floor plane.

        # 2. Wall points.
        wall_thickness = 1.0
        wall_density = 100  # Points per unit area.

        # Sample wall points.
        # Left wall.
        for y in np.linspace(
            0, self.room_size, int(self.room_size * np.sqrt(wall_density) / 10)
        ):
            for z in np.linspace(
                0, self.wall_height, int(self.wall_height * np.sqrt(wall_density))
            ):
                points.append([0, y, z])

        # Right wall.
        for y in np.linspace(
            0, self.room_size, int(self.room_size * np.sqrt(wall_density) / 10)
        ):
            for z in np.linspace(
                0, self.wall_height, int(self.wall_height * np.sqrt(wall_density))
            ):
                points.append([self.room_size, y, z])

        # Front wall.
        for x in np.linspace(
            0, self.room_size, int(self.room_size * np.sqrt(wall_density) / 10)
        ):
            for z in np.linspace(
                0, self.wall_height, int(self.wall_height * np.sqrt(wall_density))
            ):
                points.append([x, 0, z])

        # Back wall.
        for x in np.linspace(
            0, self.room_size, int(self.room_size * np.sqrt(wall_density) / 10)
        ):
            for z in np.linspace(
                0, self.wall_height, int(self.wall_height * np.sqrt(wall_density))
            ):
                points.append([x, self.room_size, z])

        # 3. Pillar points.
        surface_density = 20000  # Points per unit surface area.

        for pillar in self.map_elements["pillars"]:
            center_x = pillar["center_x"]
            center_y = pillar["center_y"]
            radius = pillar["radius"]
            height = pillar["height"]

            # Compute surface area.
            surface_area = 2 * np.pi * radius * radius + 2 * np.pi * radius * height
            num_points = int(
                surface_area * surface_density / 20
            )  # Adjust sampling density.

            # Points on the cylindrical side surface.
            side_ratio = (2 * np.pi * radius * height) / surface_area
            side_points = int(num_points * side_ratio)

            for _ in range(side_points):
                theta = np.random.uniform(0, 2 * np.pi)
                h = np.random.uniform(0, height)

                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)
                z = h

                points.append([x, y, z])

            # Points on the top and bottom circular faces.
            cap_points = num_points - side_points
            for _ in range(cap_points):
                r = np.sqrt(np.random.uniform(0, 1)) * radius
                theta = np.random.uniform(0, 2 * np.pi)

                x = center_x + r * np.cos(theta)
                y = center_y + r * np.sin(theta)
                z = height if np.random.random() > 0.5 else 0

                points.append([x, y, z])

        # 4. Partition points.
        surface_density = 20000  # Points per unit surface area.

        for partition in self.map_elements["partitions"]:
            center_x = partition["center_x"]
            center_y = partition["center_y"]
            half_length = partition["half_length"]
            half_width = partition["half_width"]
            height = partition["height"]
            is_horizontal = partition["is_horizontal"]

            if is_horizontal:
                length = 2 * half_length
                width = 2 * half_width
                x_min = center_x - half_length
                x_max = center_x + half_length
                y_min = center_y - half_width
                y_max = center_y + half_width
            else:
                length = 2 * half_width
                width = 2 * half_length
                x_min = center_x - half_width
                x_max = center_x + half_width
                y_min = center_y - half_length
                y_max = center_y + half_length

            # Compute surface area.
            surface_area = 2 * (length * width + length * height + width * height)
            num_points = int(
                surface_area * surface_density / 20
            )  # Adjust sampling density.

            # 1. Sample the top and bottom faces.
            top_bottom_area = 2 * length * width
            top_bottom_ratio = top_bottom_area / surface_area
            top_bottom_points = int(num_points * top_bottom_ratio)

            for _ in range(top_bottom_points):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                z = height if np.random.random() > 0.5 else 0

                points.append([x, y, z])

            # 2. Sample the four side faces.
            side_points = num_points - top_bottom_points
            for _ in range(side_points):
                # Select a random side face.
                side = np.random.randint(0, 4)

                if is_horizontal:
                    if side == 0:  # Front side.
                        x = np.random.uniform(x_min, x_max)
                        y = y_min
                        z = np.random.uniform(0, height)
                    elif side == 1:  # Back side.
                        x = np.random.uniform(x_min, x_max)
                        y = y_max
                        z = np.random.uniform(0, height)
                    elif side == 2:  # Left side.
                        x = x_min
                        y = np.random.uniform(y_min, y_max)
                        z = np.random.uniform(0, height)
                    else:  # Right side.
                        x = x_max
                        y = np.random.uniform(y_min, y_max)
                        z = np.random.uniform(0, height)
                else:
                    if side == 0:  # Front side.
                        x = np.random.uniform(x_min, x_max)
                        y = y_min
                        z = np.random.uniform(0, height)
                    elif side == 1:  # Back side.
                        x = np.random.uniform(x_min, x_max)
                        y = y_max
                        z = np.random.uniform(0, height)
                    elif side == 2:  # Left side.
                        x = x_min
                        y = np.random.uniform(y_min, y_max)
                        z = np.random.uniform(0, height)
                    else:  # Right side.
                        x = x_max
                        y = np.random.uniform(y_min, y_max)
                        z = np.random.uniform(0, height)

                points.append([x, y, z])

        # Convert to an Open3D point cloud.
        if points:
            points_array = np.vstack(points)
            pcd.points = o3d.utility.Vector3dVector(points_array)

            # Assign point colours.
            colors = np.zeros((points_array.shape[0], 3))

            # Floor points: light wood colour.
            floor_mask = points_array[:, 2] < self.floor_thickness
            colors[floor_mask] = [0.824, 0.706, 0.549]

            # Wall points: grey.
            wall_mask = (points_array[:, 2] >= self.floor_thickness) & (
                (points_array[:, 0] < wall_thickness)
                | (points_array[:, 0] > self.room_size - wall_thickness)
                | (points_array[:, 1] < wall_thickness)
                | (points_array[:, 1] > self.room_size - wall_thickness)
            )
            colors[wall_mask] = [0.8, 0.8, 0.8]

            # Obstacle points: reddish brown.
            obstacle_mask = ~(floor_mask | wall_mask)
            colors[obstacle_mask] = [0.545, 0.271, 0.075]

            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def generate_png(self, size=(800, 800), flip_vertical=True):
        """Generate a 2D top-down image."""
        # Reuse random state to match the mesh geometry.
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            # Generate map elements if random state has not been initialized.
            self._generate_map_elements()

        # Create an image with a white background.
        img = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(img)

        # Scale factor.
        scale_x = size[0] / self.room_size
        scale_y = size[1] / self.room_size

        # # Draw region boundaries for debugging.
        # region_grid_size = int(np.sqrt(self.num_regions))
        # region_width = size[0] / region_grid_size
        # region_height = size[1] / region_grid_size

        # for i in range(region_grid_size):
        #     for j in range(region_grid_size):
        #         region_type = self.region_types[i, j]

        #         # Add a subtle background tint to each region.
        #         if region_type == 1:  # Dense-obstacle region.
        #             color = (255, 240, 240, 50)  # Very pale pink.
        #             x1 = int(i * region_width)
        #             y1 = int(j * region_height)
        #             x2 = int((i + 1) * region_width)
        #             y2 = int((j + 1) * region_height)

        #             draw.rectangle([(x1, y1), (x2, y2)], fill=color, outline=(200, 200, 200))

        # # Draw perimeter walls.
        # border_width = 3
        # draw.rectangle([(0, 0), (size[0]-1, size[1]-1)],
        #               outline='black', width=border_width)

        # Draw pillars.
        for pillar in self.map_elements["pillars"]:
            x = pillar["center_x"]
            y = pillar["center_y"]
            radius = pillar["radius"]

            # Convert to image coordinates.
            ix = int(x * scale_x)
            iy = int(y * scale_y)
            ir = max(1, int(radius * scale_x))

            # Draw circles representing pillars.
            draw.ellipse(
                [(ix - ir, iy - ir), (ix + ir, iy + ir)], fill="brown", outline="black"
            )

        # Draw partitions.
        for partition in self.map_elements["partitions"]:
            center_x = partition["center_x"]
            center_y = partition["center_y"]
            half_length = partition["half_length"]
            half_width = partition["half_width"]
            is_horizontal = partition["is_horizontal"]

            # Convert to image coordinates.
            ix = int(center_x * scale_x)
            iy = int(center_y * scale_y)
            il = int(half_length * scale_x)
            iw = max(1, int(half_width * scale_y))

            if is_horizontal:
                # Draw horizontal partitions.
                draw.rectangle(
                    [(ix - il, iy - iw), (ix + il, iy + iw)],
                    fill="brown",
                    outline="black",
                )
            else:
                # Draw vertical partitions.
                draw.rectangle(
                    [(ix - iw, iy - il), (ix + iw, iy + il)],
                    fill="brown",
                    outline="black",
                )

        # Flip the image vertically if requested.
        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img
