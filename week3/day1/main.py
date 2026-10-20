"""
Week 3 — Day 1: Point Cloud Basics with Open3D

This script:
1. Loads a sample point cloud.
2. Visualizes the original point cloud.
3. Applies voxel downsampling.
4. Removes statistical outliers.
5. Saves the processed point cloud and screenshots.
"""

from pathlib import Path
import open3d as o3d
import numpy as np


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

VOXEL_SIZE = 0.02
NB_NEIGHBORS = 20
STD_RATIO = 2.0


# --------------------------------------------------
# 2. Helper function: save a screenshot
# --------------------------------------------------

def save_screenshot(point_cloud, file_path, title):
    """
    Render a point cloud and save a screenshot.

    Parameters
    ----------
    point_cloud : open3d.geometry.PointCloud
        Point cloud to visualize.
    file_path : Path
        Destination image path.
    title : str
        Window title.
    """

    if len(point_cloud.points) == 0:
        print(f"Cannot save screenshot: {title} is empty.")
        return

    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        window_name=title,
        width=1280,
        height=720,
        visible=True
    )

    visualizer.add_geometry(point_cloud)

    options = visualizer.get_render_option()
    options.background_color = np.array([0.1, 0.1, 0.1])
    options.point_size = 3.0

    visualizer.poll_events()
    visualizer.update_renderer()

    visualizer.capture_screen_image(
        str(file_path),
        do_render=True
    )

    visualizer.destroy_window()

    print(f"Screenshot saved: {file_path}")

# --------------------------------------------------
# 3. Load a sample point cloud
# --------------------------------------------------

def load_point_cloud():

    print("\nLoading a sample point cloud...")

    sample_data = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud(sample_data.path)

    if point_cloud.is_empty():
        raise RuntimeError(
            "Failed to load the sample point cloud."
        )

    print("Point cloud loaded successfully.")
    print(f"Source file: {sample_data.path}")
    print(f"Number of points: {len(point_cloud.points)}")

    return point_cloud

# --------------------------------------------------
# 4. Display basic information
# --------------------------------------------------

def print_point_cloud_info(point_cloud, name):
    """
    Print basic point cloud information.
    """

    points = np.asarray(point_cloud.points)

    print(f"\n--- {name} ---")
    print(f"Number of points: {len(points)}")

    if len(points) > 0:
        print(f"Minimum XYZ: {points.min(axis=0)}")
        print(f"Maximum XYZ: {points.max(axis=0)}")

    print(f"Has colors: {point_cloud.has_colors()}")
    print(f"Has normals: {point_cloud.has_normals()}")


# --------------------------------------------------
# 5. Visualize the original point cloud
# --------------------------------------------------

def visualize_point_cloud(point_cloud, title):
    """
    Open an interactive Open3D visualization window.
    """

    if point_cloud.is_empty():
        print(f"Cannot visualize an empty point cloud: {title}")
        return

    o3d.visualization.draw_geometries(
        [point_cloud],
        window_name=title,
        width=1280,
        height=720
    )


# --------------------------------------------------
# 6. Voxel downsampling
# --------------------------------------------------

def downsample_point_cloud(point_cloud):
    """
    Reduce point density using voxel downsampling.
    """

    print("\nApplying voxel downsampling...")

    downsampled = point_cloud.voxel_down_sample(
        voxel_size=VOXEL_SIZE
    )

    print(f"Original points: {len(point_cloud.points)}")
    print(f"Downsampled points: {len(downsampled.points)}")

    if len(point_cloud.points) > 0:
        reduction = (
            1 - len(downsampled.points) / len(point_cloud.points)
        ) * 100

        print(f"Point reduction: {reduction:.2f}%")

    return downsampled


# --------------------------------------------------
# 7. Statistical outlier removal
# --------------------------------------------------

def remove_outliers(point_cloud):
    """
    Remove points that are unusually far from their neighbors.
    """

    print("\nRemoving statistical outliers...")

    if len(point_cloud.points) <= NB_NEIGHBORS:
        print(
            "Not enough points for the configured "
            "number of neighbors."
        )
        return point_cloud

    cleaned, indices = point_cloud.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO
    )

    print(f"Input points: {len(point_cloud.points)}")
    print(f"Cleaned points: {len(cleaned.points)}")
    print(f"Removed points: {len(point_cloud.points) - len(indices)}")

    return cleaned


# --------------------------------------------------
# 8. Save the processed point cloud
# --------------------------------------------------

def save_point_cloud(point_cloud, file_path):
    """
    Save a point cloud to a PLY file.
    """

    success = o3d.io.write_point_cloud(
        str(file_path),
        point_cloud
    )

    if not success:
        raise RuntimeError(
            f"Failed to save point cloud: {file_path}"
        )

    print(f"Point cloud saved: {file_path}")


# --------------------------------------------------
# 9. Main pipeline
# --------------------------------------------------

def main():
    print("=" * 60)
    print("WEEK 3 — DAY 1: POINT CLOUD BASICS")
    print("=" * 60)

    # Load the sample
    original = load_point_cloud()

    print_point_cloud_info(original, "Original Point Cloud")

    # Save and display the original point cloud
    save_screenshot(
        original,
        RESULTS_DIR / "before.png",
        "Original Point Cloud"
    )

    visualize_point_cloud(
        original,
        "Original Point Cloud"
    )

    # Apply voxel downsampling
    downsampled = downsample_point_cloud(original)

    print_point_cloud_info(
        downsampled,
        "Downsampled Point Cloud"
    )

    save_screenshot(
        downsampled,
        RESULTS_DIR / "downsampled.png",
        "Downsampled Point Cloud"
    )

    visualize_point_cloud(
        downsampled,
        "Downsampled Point Cloud"
    )

    # Remove statistical outliers
    cleaned = remove_outliers(downsampled)

    print_point_cloud_info(
        cleaned,
        "Cleaned Point Cloud"
    )

    save_screenshot(
        cleaned,
        RESULTS_DIR / "cleaned.png",
        "Cleaned Point Cloud"
    )

    visualize_point_cloud(
        cleaned,
        "Cleaned Point Cloud"
    )

    # Save the final point cloud
    save_point_cloud(
        cleaned,
        RESULTS_DIR / "cleaned.ply"
    )

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    print(f"Original points: {len(original.points)}")
    print(f"After downsampling: {len(downsampled.points)}")
    print(f"After outlier removal: {len(cleaned.points)}")

    print(f"\nResults directory: {RESULTS_DIR}")
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()