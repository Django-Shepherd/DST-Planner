"""
Map explorable-area analysis tool.

This tool can:
1. Load a PNG map.
2. Count the explorable white-pixel area.
3. Convert pixel area to physical area using the supplied map dimensions.
4. Save results to the corresponding directory.
"""

import os
import glob
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from datetime import datetime


def calculate_explorable_area(image_path, map_size_meters=80):
    """
    Compute the explorable area of a map.

    Parameters:
    - image_path: Path to the map PNG.
    - map_size_meters: Physical map dimensions in metres.

    Returns:
    - A dictionary of pixel area, physical area and other statistics.
    """
    try:
        # Load the image.
        img = Image.open(image_path)
        img_np = np.array(img)

        # Read image dimensions.
        height, width = img_np.shape[:2]
        total_pixels = width * height

        # Identify explorable pixels with all RGB channels near 255.
        # Use a threshold to allow slight variations in white pixels.
        white_threshold = 240
        if len(img_np.shape) == 3:  # Colour image.
            # Check that all RGB channels exceed the threshold.
            white_mask = np.all(img_np[:, :, :3] >= white_threshold, axis=2)
        else:  # Grayscale image.
            white_mask = img_np >= white_threshold

        # Count white pixels.
        white_pixels = np.sum(white_mask)

        # Compute the fraction of white pixels.
        white_percentage = (white_pixels / total_pixels) * 100

        # Compute the physical area represented by each pixel.
        pixel_size_meters = map_size_meters / max(width, height)

        # Compute the physical white-pixel area in square metres.
        physical_area = white_pixels * (pixel_size_meters**2)

        # Return results.
        result = {
            "image_path": image_path,
            "image_size": (width, height),
            "total_pixels": total_pixels,
            "explorable_pixels": white_pixels,
            "explorable_percentage": white_percentage,
            "map_size_meters": map_size_meters,
            "pixel_size_meters": pixel_size_meters,
            "explorable_area_sqm": physical_area,
        }

        return result

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def save_result(result, output_dir=None):
    """
    Save analysis results to a text file.

    Parameters:
    - result: Analysis result dictionary.
    - output_dir: Output directory; None uses the image directory.

    Returns:
    - Output file path.
    """
    if result is None:
        return None

    # Use the image directory if no output directory is specified.
    if output_dir is None:
        output_dir = os.path.dirname(result["image_path"])

    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Build the output filename.
    base_name = os.path.basename(result["image_path"])
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{name_without_ext}_area.txt"
    output_path = os.path.join(output_dir, output_filename)

    # Write results.
    with open(output_path, "w") as f:
        f.write(f"# Map explorable-area analysis\n")
        f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Image file: {os.path.basename(result['image_path'])}\n")
        f.write(
            f"Image dimensions: {result['image_size'][0]}x{result['image_size'][1]} pixels\n"
        )
        f.write(
            f"Physical map dimensions: {result['map_size_meters']}x{result['map_size_meters']} m\n\n"
        )

        f.write(f"Total pixels: {result['total_pixels']:,}\n")
        f.write(f"Explorable pixels: {result['explorable_pixels']:,}\n")
        f.write(f"Explorable fraction: {result['explorable_percentage']:.2f}%\n\n")

        f.write(f"Pixel size: {result['pixel_size_meters']:.6f} m/pixel\n")
        f.write(
            f"Explorable physical area: {result['explorable_area_sqm']:.2f} square metres\n"
        )

    return output_path


def generate_visualization(result, output_dir=None):
    """
    Generate a visualization of the results.

    Parameters:
    - result: Analysis result dictionary.
    - output_dir: Output directory; None uses the image directory.

    Returns:
    - Visualization file path.
    """
    if result is None:
        return None

    # Use the image directory if no output directory is specified.
    if output_dir is None:
        output_dir = os.path.dirname(result["image_path"])

    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Load the original image.
    img = Image.open(result["image_path"])
    img_np = np.array(img)

    # Create the visualization figure.
    plt.figure(figsize=(12, 8))

    # First subplot: original map.
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original map")
    plt.axis("off")

    # Second subplot: explorable-area mask.
    plt.subplot(1, 2, 2)
    white_threshold = 240
    if len(img_np.shape) == 3:  # Colour image.
        white_mask = np.all(img_np[:, :, :3] >= white_threshold, axis=2)
    else:  # Grayscale image.
        white_mask = img_np >= white_threshold

    plt.imshow(white_mask, cmap="binary")
    plt.title("Explorable area (white)")
    plt.axis("off")

    # Add summary information.
    plt.suptitle(f"Map: {os.path.basename(result['image_path'])}", fontsize=16)
    plt.figtext(
        0.5,
        0.01,
        f"Explorable area: {result['explorable_area_sqm']:.2f} square metres ({result['explorable_percentage']:.2f}%)",
        ha="center",
        fontsize=14,
    )

    # Save the visualization.
    base_name = os.path.basename(result["image_path"])
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{name_without_ext}_area_viz.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def process_map_directory(directory, map_size_meters=80, create_visualization=True):
    """
    Process all PNG maps in the specified directory.

    Parameters:
    - directory: Directory containing PNG maps.
    - map_size_meters: Physical map dimensions in metres.
    - create_visualization: Whether to generate visualizations.

    Returns:
    - A list of processing results.
    """
    # Find all PNG files in the directory.
    png_files = glob.glob(os.path.join(directory, "*.png"))

    if not png_files:
        print(f"Warning: no PNG files found in {directory}")
        return []

    results = []

    for png_file in png_files:
        # Skip previously generated visualization files.
        if "_area_viz.png" in png_file or "_dilated_map.png" in png_file:
            continue

        print(f"Processing map: {os.path.basename(png_file)}")

        # Compute area.
        result = calculate_explorable_area(png_file, map_size_meters)

        if result is not None:
            # Save results.
            output_path = save_result(result)
            print(f"  Results saved to: {output_path}")

            # Generate a visualization.
            if create_visualization:
                viz_path = generate_visualization(result)
                print(f"  Visualization saved to: {viz_path}")

            results.append(result)

    return results


def main():
    # Create the command-line argument parser.
    parser = argparse.ArgumentParser(description="Compute the explorable area of a map")

    # Add arguments.
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input PNG file or directory containing PNG files",
    )
    parser.add_argument(
        "--map-size",
        "-s",
        type=float,
        default=80.0,
        help="Physical map dimensions in metres (default: 80)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: directory containing the input file)",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization output"
    )
    parser.add_argument(
        "--map-types",
        "-t",
        type=str,
        default="forest,partition,dungeon",
        help="Comma-separated map types (default: forest,partition,dungeon)",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="Root directory containing map outputs",
    )

    # Parse arguments.
    args = parser.parse_args()

    # Process a specific input file or directory if provided.
    if args.input:
        if os.path.isfile(args.input):
            # Process a single file.
            result = calculate_explorable_area(args.input, args.map_size)
            if result:
                output_path = save_result(result, args.output)
                print(f"Results saved to: {output_path}")

                if not args.no_viz:
                    viz_path = generate_visualization(result, args.output)
                    print(f"Visualization saved to: {viz_path}")
        elif os.path.isdir(args.input):
            # Process a directory.
            results = process_map_directory(args.input, args.map_size, not args.no_viz)
            print(f"Processed {len(results)} maps in total")
        else:
            print(f"Error: input is not a valid file or directory: {args.input}")
    else:
        # Default mode: process directories for all selected map types.
        map_types = [t.strip() for t in args.map_types.split(",")]

        for map_type in map_types:
            map_dir = os.path.join(args.base_dir, map_type)

            if os.path.exists(map_dir):
                print(f"\nProcessing {map_type} maps...")
                results = process_map_directory(map_dir, args.map_size, not args.no_viz)
                print(f"Processed {len(results)} {map_type} maps in total")
            else:
                print(f"Warning: {map_type} map directory not found: {map_dir}")


if __name__ == "__main__":
    main()
