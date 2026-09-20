"""
Sample free-space coordinates from PNG maps.

This tool can:
1. Find all PNG maps under the specified directory.
2. Sample free-space points while enforcing the requested obstacle clearance.
3. Save the sampled coordinates to text files.
"""

import os
import glob
import numpy as np
from PIL import Image
import random
import math

# Base directory.
BASE_DIR = "output"

# Obstacle clearance threshold in metres.
SAFETY_DISTANCE = 4.0
# Physical map dimensions in metres.
MAP_SIZE = 100.0  # Set this value to match the dimensions of the generated map.
# Number of points sampled per map.
NUM_POINTS = 5  # Sample five points by default.


def is_obstacle(img_np, x, y, map_type):
    """
    Determine whether a pixel is an obstacle.

    Parameters:
    - img_np: NumPy image array.
    - x, y: Pixel coordinates to inspect.
    - map_type: Map type.

    Returns:
    - A boolean indicating whether the pixel is an obstacle.
    """
    # Ensure coordinates are within image bounds.
    if x < 0 or y < 0 or x >= img_np.shape[1] or y >= img_np.shape[0]:
        return True  # Treat it as an obstacle.

    pixel_color = img_np[y, x]

    # Only white pixels are traversable; all other pixels are obstacles.
    return not np.all(pixel_color[:3] == [255, 255, 255])


def is_safe_distance_from_obstacles(img_np, x, y, map_type, safety_pixels):
    """
    Check whether a point has sufficient clearance from every obstacle.

    Parameters:
    - img_np: NumPy image array.
    - x, y: Pixel coordinates to inspect.
    - map_type: Map type.
    - safety_pixels: Required clearance in pixels.

    Returns:
    - A boolean indicating whether the point is safe.
    """
    # Check that the point itself is in free space.
    if is_obstacle(img_np, x, y, map_type):
        return False

    # Bound the search window by the image dimensions.
    search_range = int(math.ceil(safety_pixels))
    height, width = img_np.shape[:2]

    # Search for obstacles in the square window around the point.
    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            # Compute the pixel coordinates.
            nx, ny = x + dx, y + dy

            # Check image bounds.
            if 0 <= nx < width and 0 <= ny < height:
                # Compute the distance to any obstacle found.
                if is_obstacle(img_np, nx, ny, map_type):
                    # Use Euclidean distance.
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance <= safety_pixels:
                        return False  # The point is too close to an obstacle.

    return True  # All obstacles are sufficiently far away.


def sample_free_points(png_file, map_type, num_points=5):
    """
    Sample free-space points from a PNG map with sufficient obstacle clearance.

    Parameters:
    - png_file: Path to the PNG file.
    - map_type: Map type.
    - num_points: Number of points to sample.

    Returns:
    - A list of sampled points [(x1, y1), (x2, y2), ...].
    """
    try:
        # Read the image.
        img = Image.open(png_file)
        img_np = np.array(img)
        height, width = img_np.shape[:2]

        # Convert the required clearance to pixels.
        # For example, an 80 m map with 800 pixels gives 5 pixels per 0.5 m.
        pixels_per_meter = min(width, height) / MAP_SIZE
        safety_pixels = SAFETY_DISTANCE * pixels_per_meter
        print(f"Clearance: {SAFETY_DISTANCE} m = {safety_pixels:.2f} pixels")

        # Find safe free-space points.
        free_points = []
        safety_counter = 0
        max_attempts = 5000  # Allow more attempts because clearance constraints reject some samples.

        while len(free_points) < num_points and safety_counter < max_attempts:
            # Select a random point.
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            # Check free space and obstacle clearance.
            if is_safe_distance_from_obstacles(img_np, x, y, map_type, safety_pixels):
                # # Convert to normalized map coordinates in [0, 1].
                # norm_x = x / width
                # norm_y = y / height
                # free_points.append((norm_x, norm_y))
                free_points.append(
                    (x / pixels_per_meter, (height - y) / pixels_per_meter)
                )

            safety_counter += 1

        if len(free_points) < num_points:
            print(
                f"Warning: found only {len(free_points)}/{num_points} points with sufficient clearance"
            )
        else:
            print(
                f"Sampled {num_points} points with at least {SAFETY_DISTANCE} m clearance"
            )

        return free_points

    except Exception as e:
        print(f"Error processing {png_file}: {e}")
        import traceback

        traceback.print_exc()
        return []


def find_all_png_files(base_dir):
    """
    Recursively find PNG files under the specified directory.

    Parameters:
    - base_dir: Root directory to search.

    Returns:
    - A list of PNG file paths.
    """
    png_files = []

    # Recursively walk all subdirectories.
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".png"):
                png_files.append(os.path.join(root, file))

    return png_files


def save_points_to_txt(points, output_file):
    """
    Save sampled points to a text file.

    Parameters:
    - points: List of point coordinates.
    - output_file: Output file path.
    """
    try:
        with open(output_file, "w") as f:
            f.write(
                f"# Free-space samples (x, y): {len(points)} points, at least {SAFETY_DISTANCE} m from obstacles\n"
            )
            for i, (x, y) in enumerate(points):
                f.write(f"{x:.6f} {y:.6f}\n")
        print(f"Saved {len(points)} points to {output_file}")
    except Exception as e:
        print(f"Error saving points to {output_file}: {e}")


def main():
    """Run the command-line entry point."""
    global BASE_DIR, MAP_SIZE, SAFETY_DISTANCE, NUM_POINTS
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing map PNGs")
    parser.add_argument(
        "--map-size", required=True, type=float, help="Square map width in metres"
    )
    parser.add_argument("--clearance", type=float, default=4.0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.map_size <= 0 or args.clearance < 0 or args.count < 1:
        parser.error("map-size/count must be positive; clearance must be nonnegative")
    BASE_DIR, MAP_SIZE, SAFETY_DISTANCE, NUM_POINTS = (
        args.input,
        args.map_size,
        args.clearance,
        args.count,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Sampling free space with {SAFETY_DISTANCE} m clearance...")
    print(f"Search directory: {BASE_DIR}")

    # Find all PNG files.
    png_files = find_all_png_files(BASE_DIR)

    if not png_files:
        print(f"No PNG files found in {BASE_DIR}")
        return

    print(f"Found {len(png_files)} PNG files:")
    for png_file in png_files:
        print(f"  - {png_file}")

    print(f"\nProcessing PNG files...")

    for i, png_file in enumerate(png_files, 1):
        print(f"\n[{i}/{len(png_files)}] Processing: {os.path.basename(png_file)}")
        print(f"Full path: {png_file}")

        # Infer the map type from the filename when possible.
        basename = os.path.basename(png_file)
        if "_map.png" in basename:
            map_type = basename.replace("_map.png", "")
        elif ".png" in basename:
            map_type = basename.replace(".png", "")
        else:
            map_type = "unknown"

        # Sample free-space points.
        free_points = sample_free_points(png_file, map_type, num_points=NUM_POINTS)

        if free_points:
            # Build the output filename.
            output_dir = os.path.dirname(png_file)
            base_name = os.path.splitext(os.path.basename(png_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_free_points.txt")

            # Save points to a text file.
            save_points_to_txt(free_points, output_file)
        else:
            print(f"Skipping {png_file}: not enough free-space points found")

    print("\nSampling complete!")


if __name__ == "__main__":
    main()
