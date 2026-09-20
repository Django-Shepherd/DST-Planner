"""
Map complexity analysis tool.

This tool:
1. Visits directories for the three map types.
2. Finds a PNG example for each type.
3. Computes map complexity.
4. Saves the results to a text file in the corresponding directory.

Complexity definition:
Complexity = σ²/μ = Variance / Mean (of path length ratios)
ratio_i = shortest free-path length / Euclidean distance.
"""

import os
import glob
import numpy as np
from PIL import Image
import heapq
import random
import math
from collections import deque
import cv2


# Map types and their directories.
MAP_TYPES = ["partition", "dungeon"]
BASE_DIR = "output"

# Complexity analysis parameters.
NUM_SAMPLE_PAIRS = 10000  # Number of point pairs to sample.
MAX_PATH_LENGTH = 1000000  # Maximum path-search expansion count to bound runtime.


def is_obstacle(img_np, x, y, map_type):
    """
    Determine whether a pixel is an obstacle.

    Parameters:
    - img_np: NumPy image array.
    - x, y: Pixel coordinates to inspect.
    - map_type: Map type.

    Returns:
    - A boolean indicating whether the point is an obstacle.
    """
    # Ensure coordinates are within image bounds.
    if x < 0 or y < 0 or x >= img_np.shape[1] or y >= img_np.shape[0]:
        return True  # Treat out-of-bounds pixels as obstacles.

    pixel_color = img_np[y, x]

    # Only white pixels are traversable; all other pixels are obstacles.
    return not np.all(pixel_color[:3] == [255, 255, 255])


def euclidean_distance(p1, p2):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    - p1, p2: Point coordinates (x, y).

    Returns:
    - Euclidean distance.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def find_shortest_path(img_np, start, end, map_type):
    """
    Find the shortest path between two points using A*.

    Parameters:
    - img_np: NumPy image array.
    - start, end: Start and end coordinates (x, y).
    - map_type: Map type.

    Returns:
    - Path length, or infinity when the target cannot be reached.
    """
    # Return infinity if either endpoint is an obstacle.
    if is_obstacle(img_np, start[0], start[1], map_type) or is_obstacle(
        img_np, end[0], end[1], map_type
    ):
        return float("inf")

    height, width = img_np.shape[:2]

    # Allow axial and diagonal moves.
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),  # Axial directions.
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),  # Diagonal directions.
    ]

    # Cost function.
    def heuristic(a, b):
        return euclidean_distance(a, b)

    # A* search.
    visited = set()
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    open_set = [(f_score[start], start)]

    while open_set and len(visited) < MAX_PATH_LENGTH:
        _, current = heapq.heappop(open_set)

        if current == end:
            return g_score[current]  # Return the path length.

        if current in visited:
            continue

        visited.add(current)

        for dx, dy in directions:
            next_x, next_y = current[0] + dx, current[1] + dy
            neighbor = (next_x, next_y)

            # Check bounds and obstacles.
            if (
                0 <= next_x < width
                and 0 <= next_y < height
                and not is_obstacle(img_np, next_x, next_y, map_type)
            ):
                # A diagonal step has length sqrt(2).
                move_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # Return infinity if the target cannot be reached.
    return float("inf")


def sample_points_with_uniform_distances(img_np, map_type, num_samples):
    """
    Sample point pairs with approximately uniform Euclidean distances.

    Parameters:
    - img_np: NumPy image array.
    - map_type: Map type.
    - num_samples: Number of samples.

    Returns:
    - A list of point pairs [(start1, end1), (start2, end2), ...].
    """
    height, width = img_np.shape[:2]

    # Maximum sampling attempts.
    max_attempts = num_samples * 200000

    # Store sampled pairs and distances.
    point_pairs = []
    distances = []

    # Minimum and maximum distances.
    min_distance = width * 0.1
    max_distance = math.sqrt(width**2 + height**2) * 0.2

    attempts = 0
    while len(point_pairs) < num_samples and attempts < max_attempts:
        # Generate two random points.
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        end_x = random.randint(0, width - 1)
        end_y = random.randint(0, height - 1)

        start = (start_x, start_y)
        end = (end_x, end_y)

        # Check that both points are in free space.
        if is_obstacle(img_np, start_x, start_y, map_type) or is_obstacle(
            img_np, end_x, end_y, map_type
        ):
            attempts += 1
            continue

        # Compute Euclidean distance.
        dist = euclidean_distance(start, end)

        # Keep distances within the chosen range.
        if dist < min_distance or dist > max_distance:
            attempts += 1
            continue

        # Append the pair.
        point_pairs.append((start, end))
        distances.append(dist)

        attempts += 1

    # Check distance uniformity.
    if (
        len(point_pairs) > num_samples / 2
    ):  # Require at least half the requested samples.
        # Compute a distance-uniformity measure.
        hist, _ = np.histogram(distances, bins=10)
        uniformity = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else float("inf")

        if uniformity > 0.7:  # Report insufficient uniformity.
            print(
                f"Warning: point-pair distances are not sufficiently uniform (uniformity={uniformity:.2f})"
            )
    else:
        print(f"Warning: found only {len(point_pairs)}/{num_samples} valid point pairs")

    return point_pairs[:num_samples]  # Return at most num_samples pairs.


def calculate_complexity(img_np, map_type, num_samples=NUM_SAMPLE_PAIRS):
    """
    Compute map complexity.

    Parameters:
    - img_np: NumPy image array.
    - map_type: Map type.
    - num_samples: Number of point pairs to sample.

    Returns:
    - Complexity (variance divided by mean).
    - Number of valid sampled point pairs.
    - Mean path-length ratio.
    """
    # Sample point pairs.
    point_pairs = sample_points_with_uniform_distances(img_np, map_type, num_samples)

    # Compute the path-length ratio for each pair.
    ratios = []
    valid_pairs = 0

    for start, end in point_pairs:
        # Euclidean distance.
        euclidean_dist = euclidean_distance(start, end)

        # Shortest path length.
        path_length = find_shortest_path(img_np, start, end, map_type)

        # Compute the ratio only when a valid path exists.
        if path_length != float("inf") and path_length > 0 and euclidean_dist > 0:
            ratio = path_length / euclidean_dist
            ratios.append(ratio)
            valid_pairs += 1

    # Return zero if there are no valid point pairs.
    if len(ratios) == 0:
        return 0, 0, 0

    # Compute the mean.
    mean_ratio = np.mean(ratios)

    # Compute the variance.
    variance = np.var(ratios)

    # Complexity = variance / mean.
    complexity = variance / mean_ratio if mean_ratio > 0 else 0

    return complexity, valid_pairs, mean_ratio


def find_map_example(map_type):
    """
    Find a PNG example for the specified map type.

    Parameters:
    - map_type: Map type.

    Returns:
    - PNG path, or None if no example is found.
    """
    map_dir = os.path.join(BASE_DIR, map_type)
    if not os.path.exists(map_dir):
        print(f"Warning: map directory not found: {map_dir}")
        return None

    # Look for PNG filenames starting with map_type.
    png_files = glob.glob(os.path.join(map_dir, f"{map_type}*.png"))

    # Fall back to any PNG if no type-prefixed filename is found.
    if not png_files:
        png_files = glob.glob(os.path.join(map_dir, "*.png"))

    if not png_files:
        print(f"Warning: no PNG files found in {map_dir}")
        return None

    return png_files[0]  # Return the first PNG found.


def save_complexity_to_txt(map_type, complexity, valid_pairs, mean_ratio, output_file):
    """
    Save complexity results to a text file.

    Parameters:
    - map_type: Map type.
    - complexity: Complexity value.
    - valid_pairs: Number of valid point pairs.
    - mean_ratio: Mean path-length ratio.
    - output_file: Output file path.
    """
    try:
        with open(output_file, "w") as f:
            f.write(f"# Complexity analysis for {map_type} maps\n\n")
            f.write(f"Complexity (variance-to-mean ratio): {complexity:.6f}\n")
            f.write(f"Valid point pairs: {valid_pairs}/{NUM_SAMPLE_PAIRS}\n")
            f.write(f"Mean path-length ratio: {mean_ratio:.6f}\n\n")
            f.write(
                "Note: complexity = variance / mean, where ratio_i = shortest free-path length / Euclidean distance\n"
            )
        print(f"Saved complexity results to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")


def main():
    """Run the command-line entry point."""
    global BASE_DIR, NUM_SAMPLE_PAIRS
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Root containing partition/dungeon PNG directories",
    )
    parser.add_argument("--pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("pairs must be positive")
    BASE_DIR, NUM_SAMPLE_PAIRS = args.input, args.pairs
    random.seed(args.seed)
    np.random.seed(args.seed)
    print("Computing map complexity...")

    for map_type in MAP_TYPES:
        print(f"\nProcessing {map_type} maps...")

        # Find an example map.
        png_file = find_map_example(map_type)
        if not png_file:
            continue

        print(f"Found PNG: {png_file}")

        # Read the image.
        try:
            img = Image.open(png_file)
            img_np = np.array(img)
            print(f"Image dimensions: {img_np.shape[:2][::-1]}")

            # Use white (255,255,255) for free space and black (0,0,0) for obstacles.
            obstacle_mask = np.any(img_np[:, :, :3] != 255, axis=2)

            # Dilate obstacles.
            # Create a circular structuring element with diameter 9 and radius 4.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dilated_obstacles = cv2.dilate(
                obstacle_mask.astype(np.uint8), kernel, iterations=1
            )

            # Create a new image with white free space and black obstacles.
            new_img = np.ones_like(img_np) * 255
            new_img[dilated_obstacles == 1] = 0

            # Update the image array.
            img_np = new_img
            # Save dilated map for debugging
            debug_filename = os.path.join(
                os.path.dirname(png_file), f"{map_type}_dilated_map.png"
            )
            debug_img = Image.fromarray(img_np.astype(np.uint8))
            debug_img.save(debug_filename)
            print(f"Saved dilated map to: {debug_filename}")
            # Compute complexity.
            print("Computing complexity...")
            complexity, valid_pairs, mean_ratio = calculate_complexity(img_np, map_type)

            print(f"Complexity: {complexity:.6f}")
            print(f"Valid point pairs: {valid_pairs}/{NUM_SAMPLE_PAIRS}")
            print(f"Mean path-length ratio: {mean_ratio:.6f}")

            # Save results.
            output_dir = os.path.dirname(png_file)
            output_file = os.path.join(output_dir, f"{map_type}_complexity.txt")
            save_complexity_to_txt(
                map_type, complexity, valid_pairs, mean_ratio, output_file
            )

        except Exception as e:
            print(f"Error processing {png_file}: {e}")
            import traceback

            traceback.print_exc()

    print("\nComplexity analysis complete!")


if __name__ == "__main__":
    main()
