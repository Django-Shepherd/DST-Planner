import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from base_map_generator import BaseMapGenerator


class PartitionMapGenerator(BaseMapGenerator):
    def __init__(
        self,
        room_size=80,
        wall_height=3,
        num_columns=120,
        min_length=2,
        max_length=7,
        num_cubes=81,
        half_width=0.5,
        floor_thickness=0.1,
        output_dir="output/partition",
    ):
        """
        Partition map generator with dividing panels.

        Parameters:
        - room_size: Map dimensions.
        - wall_height: Wall height.
        - num_columns: Number of additional pillars.
        - num_cubes: Number of partitions; a perfect square is expected.
        - floor_thickness: Floor thickness.
        - output_dir: Output directory.
        """
        super().__init__(room_size, wall_height, floor_thickness, output_dir)
        self.num_columns = num_columns
        self.num_cubes = num_cubes
        self.min_length = min_length
        self.max_length = max_length
        self.half_width = half_width

    def _clear_map_elements(self):
        """Initialize map elements."""
        self.map_elements = {
            "columns": [],  # Store pillar positions and dimensions.
            "partitions": [],  # Store partition positions and dimensions.
        }

    def get_map_type(self):
        """Return the map type identifier."""
        return "partition"

    def generate_mesh(self):
        """Generate a partition map mesh."""
        # Save the random state.
        self.random_state = np.random.get_state()

        # Clear map elements.
        self._clear_map_elements()

        # Create the mesh that will contain all components.
        combined_mesh = o3d.geometry.TriangleMesh()

        # 1. Floor.
        floor_box = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=self.room_size, depth=self.floor_thickness
        )
        floor_box.translate((0, 0, 0))

        # Colour the floor using a light wood tone.
        floor_color = [0.824, 0.706, 0.549]  # Warm wood floor colour.
        floor_box.paint_uniform_color(floor_color)
        combined_mesh += floor_box

        # 2. Four perimeter walls.
        wall_thickness = 2  # Wall thickness.
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

        # 3. Grid-arranged partitions.
        # Partition colour: reddish brown.
        partition_color_options = [
            [0.545, 0.271, 0.075],  # Reddish brown.
        ]

        # Compute the grid subdivision.
        divide_number = int(np.sqrt(self.num_cubes))
        divide_length = int(self.room_size / divide_number)

        # Generate partitions.
        for i in range(divide_number):
            for j in range(divide_number):
                # Randomly choose partition positions and dimensions.
                center_x = np.random.uniform(
                    i * divide_length + 1, (i + 1) * divide_length - 1
                )
                center_y = np.random.uniform(
                    j * divide_length + 1, (j + 1) * divide_length - 1
                )

                half_length = np.random.uniform(self.min_length, self.max_length)
                half_width = self.half_width  # Fixed width.
                height_cube = self.wall_height

                # Store partition information for point-cloud and PNG generation.
                self.map_elements["partitions"].append(
                    {
                        "i": i,
                        "j": j,
                        "center_x": center_x,
                        "center_y": center_y,
                        "half_length": half_length,
                        "half_width": half_width,
                        "height": height_cube,
                    }
                )

                # Alternate horizontal and vertical orientations by position.
                if (i + j) % 2 == 0:  # Horizontal partition.
                    partition = o3d.geometry.TriangleMesh.create_box(
                        width=half_length * 2, height=half_width * 2, depth=height_cube
                    )
                    partition.translate(
                        (center_x - half_length + 0, center_y - half_width + 0, 0)
                    )
                else:  # Vertical partition.
                    partition = o3d.geometry.TriangleMesh.create_box(
                        width=half_width * 2, height=half_length * 2, depth=height_cube
                    )
                    partition.translate(
                        (center_x - half_width + 0, center_y - half_length + 0, 0)
                    )

                # Assign partition colours.
                color_idx = np.random.randint(0, len(partition_color_options))
                partition.paint_uniform_color(partition_color_options[color_idx])

                combined_mesh += partition

        # 4. Random pillars.
        if self.num_columns > 0:
            # Pillar colour.
            column_color_options = [
                [0.545, 0.271, 0.075],  # Reddish brown.
            ]

            # Generate random pillars.
            for _ in range(self.num_columns):
                center_x = np.random.uniform(
                    4, self.room_size - 4
                )  # Keep pillars away from walls.
                center_y = np.random.uniform(
                    4, self.room_size - 4
                )  # Keep pillars away from walls.
                radius = np.random.uniform(0.5, 2.0)
                height_col = np.random.uniform(
                    self.wall_height, self.wall_height * 1.5
                )  # Pillars may extend above the walls.

                # Store pillar information for point-cloud and PNG generation.
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
                # Place the bottom of each pillar flush with the floor.
                cyl.translate((center_x, center_y, height_col / 2))

                # Randomly choose pillar colours.
                color_idx = np.random.randint(0, len(column_color_options))
                cyl.paint_uniform_color(column_color_options[color_idx])

                combined_mesh += cyl

        # Remove duplicate vertices and triangles.
        combined_mesh.remove_duplicated_vertices()
        combined_mesh.remove_duplicated_triangles()
        combined_mesh.compute_vertex_normals()

        return combined_mesh

    def _generate_map_elements(self):
        """Generate partition map elements when no mesh has been generated yet."""
        # Initialize elements directly instead of calling _clear_map_elements recursively.
        self.map_elements = {
            "columns": [],  # Store pillar positions and dimensions.
            "partitions": [],  # Store partition positions and dimensions.
        }

        # Set random state.
        self.random_state = np.random.get_state()

        # Generate partition information.
        divide_number = int(np.sqrt(self.num_cubes))
        divide_length = int(self.room_size / divide_number)

        # Generate partitions.
        for i in range(divide_number):
            for j in range(divide_number):
                # Randomly choose partition positions and dimensions.
                center_x = np.random.uniform(
                    i * divide_length + 1, (i + 1) * divide_length - 1
                )
                center_y = np.random.uniform(
                    j * divide_length + 1, (j + 1) * divide_length - 1
                )

                half_length = np.random.uniform(
                    self.min_length, self.max_length
                )  # Match generate_mesh.
                half_width = self.half_width  # Fixed width.
                height_cube = self.wall_height

                # Store partition information for point-cloud and PNG generation.
                self.map_elements["partitions"].append(
                    {
                        "i": i,
                        "j": j,
                        "center_x": center_x,
                        "center_y": center_y,
                        "half_length": half_length,
                        "half_width": half_width,
                        "height": height_cube,
                    }
                )

        # Generate pillar information.
        if self.num_columns > 0:
            for _ in range(self.num_columns):
                center_x = np.random.uniform(
                    4, self.room_size - 4
                )  # Keep pillars away from walls.
                center_y = np.random.uniform(
                    4, self.room_size - 4
                )  # Keep pillars away from walls.
                radius = np.random.uniform(0.5, 2.0)
                height_col = np.random.uniform(
                    self.wall_height, self.wall_height * 1.5
                )  # Pillars may extend above the walls.

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
        # Reuse random state to match the mesh geometry.
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            # Generate map elements if random state has not been initialized.
            self._generate_map_elements()

        # Create a point cloud.
        pcd = o3d.geometry.PointCloud()
        points = []

        # 1. Uniformly sampled floor points.
        floor_density = 200  # Points per unit area.
        num_points_x = int(self.room_size * np.sqrt(floor_density))
        num_points_y = int(self.room_size * np.sqrt(floor_density))

        x_coords = np.linspace(0, self.room_size, num_points_x)
        y_coords = np.linspace(0, self.room_size, num_points_y)

        for x in x_coords:
            for y in y_coords:
                points.append([x, y, 0])  # z=0 is the floor plane.

        # 2. Wall points.
        wall_thickness = 2  # Partition maps use thicker walls.
        wall_density = 200  # Points per unit area.
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

        # 3. Surface samples on partitions.
        surface_density = (
            40000  # Points per unit surface area, consistent with pillars.
        )
        for partition in self.map_elements["partitions"]:
            center_x = partition["center_x"]
            center_y = partition["center_y"]
            half_length = partition["half_length"]
            half_width = partition["half_width"]
            height = partition["height"]
            i = partition["i"]
            j = partition["j"]

            # Determine dimensions and position from partition orientation.
            if (i + j) % 2 == 0:  # Horizontal partition.
                length = 2 * half_length
                width = 2 * half_width
                x_min = center_x - half_length + 8
                x_max = x_min + length
                y_min = center_y - half_width + 8
                y_max = y_min + width
            else:  # Vertical partition.
                length = 2 * half_width
                width = 2 * half_length
                x_min = center_x - half_width + 8
                x_max = x_min + length
                y_min = center_y - half_length + 8
                y_max = y_min + width

            # Surface area: 2 * (length*width + length*height + width*height).
            surface_area = 2 * (length * width + length * height + width * height)
            num_points = int(
                surface_area * surface_density / 15
            )  # Adjust sampling density consistently with pillars.

            # Compute relative face areas.
            top_bottom_area = 2 * length * width  # Top and bottom faces.
            front_back_area = 2 * length * height  # Front and back faces.
            left_right_area = 2 * width * height  # Left and right faces.

            # Allocate sample counts.
            top_bottom_points = int(num_points * top_bottom_area / surface_area)
            front_back_points = int(num_points * front_back_area / surface_area)
            left_right_points = num_points - top_bottom_points - front_back_points

            # 1. Sample the top and bottom faces.
            for _ in range(top_bottom_points):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                z = (
                    height if np.random.random() > 0.5 else 0
                )  # Put half the samples on top and half on the bottom.
                points.append([x, y, z])

            # 2. Sample front and back faces.
            for _ in range(front_back_points):
                x = np.random.uniform(x_min, x_max)
                z = np.random.uniform(0, height)
                y = (
                    y_max if np.random.random() > 0.5 else y_min
                )  # Put half the samples on the front and half on the back.
                points.append([x, y, z])

            # 3. Sample left and right faces.
            for _ in range(left_right_points):
                y = np.random.uniform(y_min, y_max)
                z = np.random.uniform(0, height)
                x = (
                    x_max if np.random.random() > 0.5 else x_min
                )  # Put half the samples on the left and half on the right.
                points.append([x, y, z])

        # 4. Surface samples on pillars.
        surface_density = 40000  # Points per unit surface area.
        for column in self.map_elements["columns"]:
            center_x = column["center_x"]
            center_y = column["center_y"]
            radius = column["radius"]
            height = column["height"]

            # Surface area: 2*pi*r^2 for the ends plus 8*pi*r*h for the side.
            surface_area = 2 * np.pi * radius * radius + 8 * np.pi * radius * height
            num_points = int(
                surface_area * surface_density / 15
            )  # Adjust sampling density.

            # Points on the cylindrical side surface.
            side_ratio = (2 * np.pi * radius * height) / surface_area
            side_points = int(num_points * side_ratio)

            for _ in range(side_points):
                # Uniformly sample the side surface.
                theta = np.random.uniform(0, 2 * np.pi)
                h = np.random.uniform(0, height)

                # Place side samples at the cylinder radius.
                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)
                z = h

                points.append([x, y, z])

            # Points on the top and bottom circular faces.
            cap_points = num_points - side_points
            for _ in range(cap_points):
                # Uniformly sample circular faces.
                r = (
                    np.sqrt(np.random.uniform(0, 1)) * radius
                )  # Use sqrt to obtain uniform area sampling.
                theta = np.random.uniform(0, 2 * np.pi)

                x = center_x + r * np.cos(theta)
                y = center_y + r * np.sin(theta)

                # Put half the samples on the bottom and half on top.
                z = height if np.random.random() > 0.5 else 0

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

            # Other points (partitions and pillars): reddish brown.
            other_mask = ~(floor_mask | wall_mask)
            colors[other_mask] = [0.545, 0.271, 0.075]

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

        # Draw the boundary.
        border_width = 3
        draw.rectangle(
            [(0, 0), (size[0] - 1, size[1] - 1)], outline="black", width=border_width
        )

        # Draw perimeter walls.
        wall_thickness = 2
        wall_width = int(wall_thickness * scale_x)
        draw.rectangle([(0, 0), (size[0] - 1, wall_width)], fill="gray")  # Top wall.
        draw.rectangle(
            [(0, size[1] - wall_width), (size[0] - 1, size[1] - 1)], fill="gray"
        )  # Bottom wall.
        draw.rectangle([(0, 0), (wall_width, size[1] - 1)], fill="gray")  # Left wall.
        draw.rectangle(
            [(size[0] - wall_width, 0), (size[0] - 1, size[1] - 1)], fill="gray"
        )  # Right wall.

        # Draw partitions using stored geometry.
        for partition in self.map_elements["partitions"]:
            center_x = partition["center_x"]
            center_y = partition["center_y"]
            half_length = partition["half_length"]
            half_width = partition["half_width"]
            i = partition["i"]
            j = partition["j"]

            # Convert to image coordinates.
            ix = int(center_x * scale_x)
            iy = int(center_y * scale_y)

            if (i + j) % 2 == 0:  # Horizontal partition.
                length = int(half_length * 2 * scale_x)
                width = int(half_width * 2 * scale_y)
                draw.rectangle(
                    [
                        (ix - length // 2 + 4 * scale_x, iy - width // 2 + 4 * scale_y),
                        (ix + length // 2 + 4 * scale_x, iy + width // 2 + 4 * scale_y),
                    ],
                    fill="brown",
                )
            else:  # Vertical partition.
                length = int(half_length * 2 * scale_y)
                width = int(half_width * 2 * scale_x)
                draw.rectangle(
                    [
                        (ix - width // 2 + 4 * scale_x, iy - length // 2 + 4 * scale_y),
                        (ix + width // 2 + 4 * scale_x, iy + length // 2 + 4 * scale_y),
                    ],
                    fill="brown",
                )

        # Draw pillars using stored geometry.
        for column in self.map_elements["columns"]:
            x = column["center_x"]
            y = column["center_y"]
            radius = column["radius"]

            # Convert to image coordinates.
            ix = int(x * scale_x)
            iy = int(y * scale_y)
            ir = max(1, int(radius * scale_x))

            # Draw circles representing pillars.
            draw.ellipse(
                [(ix - ir, iy - ir), (ix + ir, iy + ir)], fill="brown", outline="black"
            )
        # Flip the image vertically if requested.
        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img
