import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import random
import math
from base_map_generator import BaseMapGenerator


class DungeonMapGenerator(BaseMapGenerator):
    def __init__(
        self,
        room_size=80,
        wall_height=3,
        floor_thickness=0.1,
        max_rooms=30,
        min_room_size=4,
        max_room_size=12,
        min_distance=5,
        corridor_width=2,  # Corridor width.
        output_dir="output/dungeon",
    ):
        """
        Dungeon generator with randomly placed rooms and connecting corridors.

        Parameters:
        - room_size: Overall map dimensions.
        - wall_height: Wall height.
        - floor_thickness: Floor thickness.
        - max_rooms: Maximum number of rooms.
        - min_room_size: Minimum room dimensions.
        - max_room_size: Maximum room dimensions.
        - min_distance: Minimum spacing between rooms.
        - corridor_width: Corridor width in grid cells.
        - output_dir: Output directory.
        """
        super().__init__(room_size, wall_height, floor_thickness, output_dir)
        self.max_rooms = max_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.min_distance = min_distance
        self.corridor_width = max(
            2, corridor_width
        )  # Ensure corridors are at least one cell wide.
        self.grid_size = room_size  # Map grid dimensions.

    def _clear_map_elements(self):
        """Initialize map elements."""
        self.map_elements = {
            "rooms": [],  # Store room information.
            "tunnels": [],  # Store connecting corridors.
            "player_start": None,  # Player start position.
            "player_end": None,  # Player goal position.
        }
        # Initialize the grid as walls (True indicates a wall).
        self.tile_map = [
            [True for _ in range(self.grid_size)] for _ in range(self.grid_size)
        ]

    def get_map_type(self):
        """Return the map type identifier."""
        return "dungeon"

    def _create_room(self, x, y, w, h):
        """Create a rectangular room."""
        # Mark the rectangle as floor (False indicates floor).
        for i in range(x, min(x + w, self.grid_size)):
            for j in range(y, min(y + h, self.grid_size)):
                self.tile_map[i][j] = False

        # Save room information.
        room_center = (x + w // 2, y + h // 2)
        self.map_elements["rooms"].append(
            {"x": x, "y": y, "width": w, "height": h, "center": room_center}
        )
        return room_center

    def _is_point_in_room(self, x, y, room_index=None):
        """
        Check whether a point lies inside any room.

        Parameters:
        - x, y: Pixel coordinates to inspect.
        - room_index: Room to exclude; None checks all rooms.

        Returns:
        - A boolean indicating whether the point lies inside a room.
        """
        for i, room in enumerate(self.map_elements["rooms"]):
            if room_index is not None and i == room_index:
                continue  # Skip the excluded room.

            room_x = room["x"]
            room_y = room["y"]
            room_w = room["width"]
            room_h = room["height"]

            if (room_x <= x < room_x + room_w) and (room_y <= y < room_y + room_h):
                return True

        return False

    def _is_path_clear(self, x1, y1, x2, y2, room1_index, room2_index):
        """
        Check whether a straight path between two points crosses another room.

        Parameters:
        - x1, y1: Start coordinates.
        - x2, y2: End coordinates.
        - room1_index, room2_index: Rooms to exclude from the check.

        Returns:
        - A boolean indicating whether the path avoids other rooms.
        """
        # Determine how many points to sample along the path.
        dist = max(abs(x2 - x1), abs(y2 - y1))
        if dist == 0:
            return True

        # Sample points along the path.
        num_samples = dist * 2  # Use enough samples for the crossing check.

        for i in range(num_samples + 1):
            t = i / num_samples
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)

            # Check whether the point lies inside another room.
            if self._is_point_in_room(x, y, room1_index) or self._is_point_in_room(
                x, y, room2_index
            ):
                excluded_rooms = [room1_index, room2_index]
                in_another_room = False

                for i, room in enumerate(self.map_elements["rooms"]):
                    if i in excluded_rooms:
                        continue

                    room_x = room["x"]
                    room_y = room["y"]
                    room_w = room["width"]
                    room_h = room["height"]

                    # Check whether the point is in this room.
                    if (room_x <= x < room_x + room_w) and (
                        room_y <= y < room_y + room_h
                    ):
                        in_another_room = True
                        break

                if in_another_room:
                    return False

        return True

    def _create_htunnel(self, x1, x2, y):
        """
        Create a horizontal corridor.

        Parameters:
        - x1, x2: Start and end x coordinates.
        - y: Centre y coordinate.
        """
        # Clamp coordinates to grid bounds.
        y = max(0, min(y, self.grid_size - 1))
        x1 = max(0, min(x1, self.grid_size - 1))
        x2 = max(0, min(x2, self.grid_size - 1))

        # Compute the lower and upper corridor bounds.
        half_width = self.corridor_width // 2
        y_start = max(0, y - half_width)
        y_end = min(self.grid_size - 1, y + half_width + (self.corridor_width % 2) - 1)

        # Create a horizontal corridor.
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y_pos in range(y_start, y_end + 1):
                self.tile_map[x][y_pos] = False

        # Save corridor information.
        self.map_elements["tunnels"].append(
            {
                "type": "horizontal",
                "x1": min(x1, x2),
                "x2": max(x1, x2),
                "y": y,
                "width": self.corridor_width,
            }
        )

    def _create_vtunnel(self, y1, y2, x):
        """
        Create a vertical corridor.

        Parameters:
        - y1, y2: Start and end y coordinates.
        - x: Centre x coordinate.
        """
        # Clamp coordinates to grid bounds.
        x = max(0, min(x, self.grid_size - 1))
        y1 = max(0, min(y1, self.grid_size - 1))
        y2 = max(0, min(y2, self.grid_size - 1))

        # Compute the left and right corridor bounds.
        half_width = self.corridor_width // 2
        x_start = max(0, x - half_width)
        x_end = min(self.grid_size - 1, x + half_width + (self.corridor_width % 2) - 1)

        # Create a vertical corridor.
        for y in range(min(y1, y2), max(y1, y2) + 1):
            for x_pos in range(x_start, x_end + 1):
                self.tile_map[x_pos][y] = False

        # Save corridor information.
        self.map_elements["tunnels"].append(
            {
                "type": "vertical",
                "y1": min(y1, y2),
                "y2": max(y1, y2),
                "x": x,
                "width": self.corridor_width,
            }
        )

    def _generate_dungeon_layout(self):
        """Generate the dungeon layout, including rooms and corridors."""
        # Preserve exterior walls.
        for x in range(self.grid_size):
            self.tile_map[x][0] = True
            self.tile_map[x][self.grid_size - 1] = True
        for y in range(self.grid_size):
            self.tile_map[0][y] = True
            self.tile_map[self.grid_size - 1][y] = True

        # Create rooms.
        room_centers = []
        for _ in range(self.max_rooms):
            room_added = False
            attempts = 0

            while (
                not room_added and attempts < 100
            ):  # Try up to ten times to find a suitable position.
                # Choose random room dimensions.
                w = random.randint(self.min_room_size, self.max_room_size)
                h = random.randint(self.min_room_size, self.max_room_size)
                x = random.randint(1, self.grid_size - w - 1)
                y = random.randint(1, self.grid_size - h - 1)

                new_center = (x + w // 2, y + h // 2)

                # Check distances from existing rooms.
                too_close = any(
                    math.sqrt(
                        (center[0] - new_center[0]) ** 2
                        + (center[1] - new_center[1]) ** 2
                    )
                    < self.min_distance
                    for center in room_centers
                )

                if (
                    not too_close or random.random() < 0.0
                ):  # Enforce spacing with 70% probability and skip the check otherwise.
                    room_center = self._create_room(x, y, w, h)
                    room_centers.append(room_center)
                    room_added = True

                attempts += 1

        # Connect rooms using a minimum spanning tree to avoid initial cycles.
        if len(room_centers) > 1:
            # Compute distances between all rooms.
            edges = []
            for i in range(len(room_centers)):
                for j in range(i + 1, len(room_centers)):
                    c1 = room_centers[i]
                    c2 = room_centers[j]
                    distance = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
                    edges.append((distance, i, j))

            # Sort by distance.
            edges.sort()

            # Use disjoint sets to construct the minimum spanning tree.
            parent = list(range(len(room_centers)))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                parent[find(x)] = find(y)

            # Create corridor connections.
            connected_pairs = set()  # Track connected room pairs.

            for dist, u, v in edges:
                if find(u) != find(v):
                    union(u, v)
                    connected_pairs.add((u, v))

            # Add approximately 30% extra connections to form cycles.
            extra_connections = int(0.3 * len(room_centers))
            added_extras = 0

            for dist, u, v in edges:
                if (
                    (u, v) not in connected_pairs
                    and (v, u) not in connected_pairs
                    and added_extras < extra_connections
                ):
                    # Check that the path does not cross another room.
                    if self._is_path_clear(
                        room_centers[u][0],
                        room_centers[u][1],
                        room_centers[v][0],
                        room_centers[v][1],
                        u,
                        v,
                    ):
                        connected_pairs.add((u, v))
                        added_extras += 1

            # Connect each selected pair using a suitable corridor route.
            for u, v in connected_pairs:
                c1 = room_centers[u]
                c2 = room_centers[v]

                # Choose a corridor route type.
                # 1. Try a direct connection.
                direct_path_clear = self._is_path_clear(
                    c1[0], c1[1], c2[0], c2[1], u, v
                )

                if (
                    direct_path_clear or random.random() < 0.2
                ):  # Force a straight connection with 20% probability.
                    # Choose horizontal-then-vertical or vertical-then-horizontal routing.
                    if random.random() < 0.5:
                        self._create_htunnel(c1[0], c2[0], c1[1])
                        self._create_vtunnel(c1[1], c2[1], c2[0])
                    else:
                        self._create_vtunnel(c1[1], c2[1], c1[0])
                        self._create_htunnel(c1[0], c2[0], c2[1])
                else:
                    # Try an intermediate point that avoids rooms.
                    # Choose an initial intermediate point with a clear route.
                    mid_x = (c1[0] + c2[0]) // 2
                    mid_y = (c1[1] + c2[1]) // 2

                    # Adjust the intermediate point based on the larger displacement.
                    dx = abs(c2[0] - c1[0])
                    dy = abs(c2[1] - c1[1])

                    if dx > dy:  # The horizontal displacement is larger.
                        mid_x = random.randint(
                            min(c1[0], c2[0]) + dx // 4, max(c1[0], c2[0]) - dx // 4
                        )
                    else:  # The vertical displacement is larger.
                        mid_y = random.randint(
                            min(c1[1], c2[1]) + dy // 4, max(c1[1], c2[1]) - dy // 4
                        )

                    # Keep the intermediate point within bounds.
                    mid_x = max(1, min(mid_x, self.grid_size - 2))
                    mid_y = max(1, min(mid_y, self.grid_size - 2))

                    # Check whether the intermediate point lies in a room.
                    for attempts in range(
                        10
                    ):  # Try up to ten times to find a suitable intermediate point.
                        if self._is_point_in_room(mid_x, mid_y):
                            # Randomly adjust the intermediate point.
                            offset = random.randint(3, 10)
                            direction = random.choice(
                                [(0, offset), (0, -offset), (offset, 0), (-offset, 0)]
                            )
                            mid_x += direction[0]
                            mid_y += direction[1]

                            # Clamp it to the grid bounds again.
                            mid_x = max(1, min(mid_x, self.grid_size - 2))
                            mid_y = max(1, min(mid_y, self.grid_size - 2))
                        else:
                            break

                    # Use a simple connection if the intermediate point is unsuitable.
                    if not (
                        0 <= mid_x < self.grid_size and 0 <= mid_y < self.grid_size
                    ):
                        # Fall back to a simple connection.
                        if random.random() < 0.5:
                            self._create_htunnel(c1[0], c2[0], c1[1])
                            self._create_vtunnel(c1[1], c2[1], c2[0])
                        else:
                            self._create_vtunnel(c1[1], c2[1], c1[0])
                            self._create_htunnel(c1[0], c2[0], c2[1])
                    else:
                        # Create the two corridor segments.
                        try:
                            self._create_htunnel(c1[0], mid_x, c1[1])
                            self._create_vtunnel(c1[1], mid_y, mid_x)
                            self._create_htunnel(mid_x, c2[0], mid_y)
                            self._create_vtunnel(mid_y, c2[1], c2[0])
                        except IndexError:
                            # Fall back to a simple connection on an indexing error.
                            print(
                                f"Warning: corridor generation failed; using a simple connection ({c1} -> {c2})"
                            )
                            if random.random() < 0.5:
                                self._create_htunnel(c1[0], c2[0], c1[1])
                                self._create_vtunnel(c1[1], c2[1], c2[0])
                            else:
                                self._create_vtunnel(c1[1], c2[1], c1[0])
                                self._create_htunnel(c1[0], c2[0], c2[1])

    def _generate_map_elements(self):
        """Generate dungeon map elements."""
        # Clear and directly initialize map elements and the grid.
        self._clear_map_elements()

        # Set random state.
        self.random_state = np.random.get_state()

        # Generate the dungeon layout.
        self._generate_dungeon_layout()

    def generate_mesh(self):
        """Generate a dungeon mesh."""
        # Save the random state.
        self.random_state = np.random.get_state()

        # Clear map elements.
        self._clear_map_elements()

        # Generate map elements such as rooms and corridors.
        self._generate_dungeon_layout()

        # Create the mesh that will contain all components.
        combined_mesh = o3d.geometry.TriangleMesh()

        # 1. Floor.
        floor_box = o3d.geometry.TriangleMesh.create_box(
            width=self.room_size, height=self.room_size, depth=self.floor_thickness
        )
        floor_box.translate((0, 0, 0))

        # Colour the floor grey to resemble stone.
        floor_color = [0.6, 0.6, 0.6]  # Grey.
        floor_box.paint_uniform_color(floor_color)
        combined_mesh += floor_box

        # 2. Walls.
        wall_thickness = 1.0
        wall_color = [0.4, 0.4, 0.4]  # Dark grey walls.

        # Create walls by traversing the grid.
        scale_factor = self.room_size / self.grid_size  # Scale factor.

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.tile_map[x][y]:  # This cell is a wall.
                    wall_box = o3d.geometry.TriangleMesh.create_box(
                        width=scale_factor, height=scale_factor, depth=self.wall_height
                    )
                    wall_box.translate((x * scale_factor, y * scale_factor, 0))
                    wall_box.paint_uniform_color(wall_color)
                    combined_mesh += wall_box

        return combined_mesh

    def generate_pcd(self, mesh=None):
        """Generate points directly without sampling a mesh."""
        # Reuse random state to match the mesh geometry.
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        else:
            # Generate map elements if random state has not been initialized.
            self._generate_map_elements()

        points = []

        # 1. Floor points.
        floor_density = 200  # Points per unit area.
        num_points_x = int(self.room_size * np.sqrt(floor_density))
        num_points_y = int(self.room_size * np.sqrt(floor_density))

        # Uniformly sample floor points.
        x_coords = np.linspace(0, self.room_size, num_points_x)
        y_coords = np.linspace(0, self.room_size, num_points_y)

        scale_factor = self.room_size / self.grid_size

        for x in x_coords:
            for y in y_coords:
                # Convert coordinates to grid indices.
                grid_x = int(x / scale_factor)
                grid_y = int(y / scale_factor)

                # This cell is floor rather than a wall.
                if (
                    0 <= grid_x < self.grid_size
                    and 0 <= grid_y < self.grid_size
                    and not self.tile_map[grid_x][grid_y]
                ):
                    points.append([x, y, 0])  # z=0 is the floor plane.

        # 2. Wall points.
        wall_density = 5000  # Points per unit area.

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.tile_map[x][y]:  # This cell is a wall.
                    # Wall coordinates in the world frame.
                    wx = x * scale_factor
                    wy = y * scale_factor

                    # Sample each wall face.
                    num_points_side = max(5, int(scale_factor * np.sqrt(wall_density)))

                    for _ in range(num_points_side * num_points_side):
                        # Select a random face.
                        face = np.random.randint(
                            5
                        )  # Five faces; the bottom is not visible.

                        if face == 0:  # Top face.
                            px = wx + np.random.uniform(0, scale_factor)
                            py = wy + np.random.uniform(0, scale_factor)
                            pz = self.wall_height
                        elif face == 1:  # Front face.
                            px = wx + np.random.uniform(0, scale_factor)
                            py = wy
                            pz = np.random.uniform(0, self.wall_height)
                        elif face == 2:  # Back face.
                            px = wx + np.random.uniform(0, scale_factor)
                            py = wy + scale_factor
                            pz = np.random.uniform(0, self.wall_height)
                        elif face == 3:  # Left face.
                            px = wx
                            py = wy + np.random.uniform(0, scale_factor)
                            pz = np.random.uniform(0, self.wall_height)
                        else:  # Right face.
                            px = wx + scale_factor
                            py = wy + np.random.uniform(0, scale_factor)
                            pz = np.random.uniform(0, self.wall_height)

                        points.append([px, py, pz])

        # Convert to an Open3D point cloud.
        if points:
            points_array = np.vstack(points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_array)

            # Assign point colours.
            colors = np.zeros((points_array.shape[0], 3))

            # Floor points: grey.
            floor_mask = points_array[:, 2] < self.floor_thickness
            colors[floor_mask] = [0.6, 0.6, 0.6]

            # Wall points: dark grey.
            wall_mask = ~floor_mask
            colors[wall_mask] = [0.4, 0.4, 0.4]

            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def generate_png(self, size=(800, 800), flip_vertical=True):
        """Generate a 2D top-down image.

        Parameters:
        - size: Output image dimensions.
        - flip_vertical: Whether to flip the image vertically.
        """
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
        scale_x = size[0] / self.grid_size
        scale_y = size[1] / self.grid_size

        # Draw the map.
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                ix = int(x * scale_x)
                iy = int(y * scale_y)
                iw = max(1, int(scale_x))
                ih = max(1, int(scale_y))

                if self.tile_map[x][y]:  # This cell is a wall.
                    draw.rectangle(
                        [(ix, iy), (ix + iw - 1, iy + ih - 1)], fill="darkgray"
                    )

        # Draw the player start position.
        if self.map_elements["player_start"]:
            px, py = self.map_elements["player_start"]
            ix = int(px * scale_x)
            iy = int(py * scale_y)
            r = max(3, int(min(scale_x, scale_y) / 2))
            draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], fill="yellow")

        # Draw the player goal position.
        if self.map_elements["player_end"]:
            px, py = self.map_elements["player_end"]
            ix = int(px * scale_x)
            iy = int(py * scale_y)
            r = max(3, int(min(scale_x, scale_y) / 2))
            draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], fill="green")

        # Flip the image vertically if requested.
        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img
