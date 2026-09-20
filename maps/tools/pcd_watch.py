"""
PCD point-cloud visualization tool.
Features:
1. Load and visualize PCD files.
2. Interactively rotate, zoom and pan.
3. Display point-cloud statistics.
4. View multiple files.
"""

import open3d as o3d
import numpy as np
import argparse
import os
import sys


class PCDVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.point_cloud = None

    def load_pcd(self, pcd_file):
        """Load a PCD file."""
        if not os.path.exists(pcd_file):
            print(f"Error: file not found - {pcd_file}")
            return False

        try:
            self.point_cloud = o3d.io.read_point_cloud(pcd_file)

            # Display point-cloud information.
            print(f"\n=== PCD file information ===")
            print(f"File path: {pcd_file}")
            print(f"Point count: {len(self.point_cloud.points)}")

            if self.point_cloud.has_colors():
                print(f"Contains colour information")

            if self.point_cloud.has_normals():
                print(f"Contains normal vectors")

            # Compute the bounding box.
            points = np.asarray(self.point_cloud.points)
            if len(points) > 0:
                min_bound = points.min(axis=0)
                max_bound = points.max(axis=0)
                print(f"\nBounding box:")
                print(f"  X: [{min_bound[0]:.2f}, {max_bound[0]:.2f}]")
                print(f"  Y: [{min_bound[1]:.2f}, {max_bound[1]:.2f}]")
                print(f"  Z: [{min_bound[2]:.2f}, {max_bound[2]:.2f}]")
                print(f"Dimensions: {max_bound - min_bound}")

            return True

        except Exception as e:
            print(f"Load failed: {e}")
            return False

    def visualize(self, show_coordinate=True, point_size=1.0):
        """Visualize the point cloud."""
        if self.point_cloud is None:
            print("Error: no point cloud has been loaded")
            return

        # Create a visualization window.
        self.vis.create_window(window_name="PCD Viewer", width=1280, height=720)

        # Add the point cloud.
        self.vis.add_geometry(self.point_cloud)

        # Add a coordinate frame.
        if show_coordinate:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            self.vis.add_geometry(coordinate_frame)

        # Configure rendering options.
        render_option = self.vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array(
            [0.1, 0.1, 0.1]
        )  # Dark grey background.

        # Set the initial viewpoint.
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)

        print("\n=== Controls ===")
        print("Left mouse button: rotate")
        print("Right mouse button: pan")
        print("Mouse wheel: zoom")
        print("H: show help")
        print("Q/ESC: quit")
        print("\nDisplaying point cloud; press Q or ESC to quit...")

        # Run the visualizer.
        self.vis.run()
        self.vis.destroy_window()


def visualize_simple(pcd_file, **kwargs):
    """Provide a simple visualization interface."""
    pcd = o3d.io.read_point_cloud(pcd_file)

    geometries = [pcd]

    # Add a coordinate frame.
    if kwargs.get("show_coordinate", True):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        geometries.append(coordinate_frame)

    # Visualize the cloud.
    o3d.visualization.draw_geometries(
        geometries,
        window_name="PCD Viewer",
        width=1280,
        height=720,
        point_show_normal=kwargs.get("show_normals", False),
    )


def batch_view(pcd_files):
    """View multiple PCD files in sequence."""
    print(f"\nFound {len(pcd_files)} PCD files")

    for i, pcd_file in enumerate(pcd_files):
        print(f"\n[{i + 1}/{len(pcd_files)}] {os.path.basename(pcd_file)}")
        visualizer = PCDVisualizer()

        if visualizer.load_pcd(pcd_file):
            visualizer.visualize()

        if i < len(pcd_files) - 1:
            response = input("\nView the next file? (y/n): ")
            if response.lower() != "y":
                break


def main():
    parser = argparse.ArgumentParser(description="PCD point-cloud viewer")
    parser.add_argument(
        "pcd_path", type=str, help="PCD file or directory containing PCD files"
    )
    parser.add_argument(
        "--no-coordinate", action="store_true", help="Hide the coordinate frame"
    )
    parser.add_argument(
        "--point-size", type=float, default=1.0, help="Point size (default: 1.0)"
    )
    parser.add_argument(
        "--show-normals", action="store_true", help="Show normal vectors"
    )
    parser.add_argument(
        "--simple", action="store_true", help="Use the simple one-shot viewer"
    )

    args = parser.parse_args()

    # Check the input path.
    if os.path.isfile(args.pcd_path):
        # Single file.
        pcd_files = [args.pcd_path]
    elif os.path.isdir(args.pcd_path):
        # Directory: find all PCD files.
        pcd_files = [
            os.path.join(args.pcd_path, f)
            for f in os.listdir(args.pcd_path)
            if f.endswith(".pcd")
        ]
        pcd_files.sort()
    else:
        print(f"Error: path not found - {args.pcd_path}")
        sys.exit(1)

    if not pcd_files:
        print(f"Error: no PCD files found")
        sys.exit(1)

    # Visualize the cloud.
    if args.simple:
        for pcd_file in pcd_files:
            print(f"\nDisplaying: {os.path.basename(pcd_file)}")
            visualize_simple(
                pcd_file,
                show_coordinate=not args.no_coordinate,
                show_normals=args.show_normals,
            )
    else:
        if len(pcd_files) == 1:
            visualizer = PCDVisualizer()
            if visualizer.load_pcd(pcd_files[0]):
                visualizer.visualize(
                    show_coordinate=not args.no_coordinate, point_size=args.point_size
                )
        else:
            batch_view(pcd_files)


if __name__ == "__main__":
    main()
