"""
WEEK 3 — DAY 3
Point Cloud Cleaning: Normals + ROI Cropping

Pipeline:
1. Load point cloud
2. Voxel downsampling
3. Statistical Outlier Removal
4. Radius Outlier Removal
5. Estimate normals
6. Crop Region of Interest (ROI)
7. Save point clouds and visualizations

Requirements:
    pip install open3d numpy matplotlib
"""

from pathlib import Path

import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

CLEANED_DIR = RESULTS_DIR / "cleaned"
NORMALS_DIR = RESULTS_DIR / "with_normals"
CROPPED_DIR = RESULTS_DIR / "cropped"

RANDOM_SEED = 42

# Voxel size relative to the cloud's bounding-box diagonal.
# Increase this value for stronger downsampling.
VOXEL_SIZE_RATIO = 0.005

# Statistical Outlier Removal
STATISTICAL_NB_NEIGHBORS = 20
STATISTICAL_STD_RATIO = 2.0

# Radius Outlier Removal
RADIUS_NB_POINTS = 8
RADIUS_RATIO = 0.015

# Normal estimation
NORMAL_RADIUS_RATIO = 0.03
NORMAL_MAX_NN = 30

# ROI boundaries as fractions of the cloud's bounding box.
# 0.10–0.90 keeps the central 80% along each axis.
# Change these values to select a different region.
ROI_MIN_RATIO = (0.10, 0.10, 0.10)
ROI_MAX_RATIO = (0.90, 0.90, 0.90)

# Visualization settings
MAX_DISPLAY_POINTS = 5000
NORMAL_DISPLAY_POINTS = 500
NORMAL_ARROW_LENGTH_RATIO = 0.025


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create all required output directories."""

    for directory in (
        DATA_DIR,
        CLEANED_DIR,
        NORMALS_DIR,
        CROPPED_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print("Output directories are ready.")


# ============================================================
# LOAD POINT CLOUD
# ============================================================

def load_point_cloud():
    """
    Load a point cloud from data/.

    Supported formats:
        .ply, .pcd, .xyz, .xyzn, .xyzrgb, .pts

    If no point cloud is found, download the Stanford Bunny
    mesh and sample points from its surface.
    """

    supported_extensions = {
        ".ply",
        ".pcd",
        ".xyz",
        ".xyzn",
        ".xyzrgb",
        ".pts",
    }

    files = sorted(
        file
        for file in DATA_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in supported_extensions
    )

    if files:
        file_path = files[0]

        print(f"\nLoading point cloud: {file_path.name}")

        pcd = o3d.io.read_point_cloud(str(file_path))

        if pcd.is_empty():
            raise ValueError(
                f"The point cloud is empty or could not be read: "
                f"{file_path}"
            )

        return pcd, file_path.stem

    print("\nNo point cloud found in data/.")
    print("Loading the Stanford Bunny mesh instead...")

    try:
        bunny_data = o3d.data.BunnyMesh()
        mesh_path = bunny_data.path

        mesh = o3d.io.read_triangle_mesh(mesh_path)

        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise RuntimeError("The Stanford Bunny mesh is empty.")

        # Do not pass seed= here:
        # some Open3D versions do not support this argument.
        pcd = mesh.sample_points_uniformly(
            number_of_points=30000
        )

        if pcd.is_empty():
            raise RuntimeError(
                "Failed to sample points from the Bunny mesh."
            )

        print("Stanford Bunny loaded successfully.")

        return pcd, "stanford_bunny"

    except Exception as error:
        raise RuntimeError(
            "Could not load a point cloud or download Stanford Bunny.\n"
            "Place a supported point-cloud file in week3_pointclouds/data/."
        ) from error


# ============================================================
# BASIC UTILITIES
# ============================================================

def get_diagonal(pcd):
    """Return the diagonal length of the cloud's bounding box."""

    bbox = pcd.get_axis_aligned_bounding_box()
    diagonal = float(np.linalg.norm(bbox.get_extent()))

    if diagonal <= 0:
        raise ValueError(
            "The point cloud has zero or invalid dimensions."
        )

    return diagonal


def save_point_cloud(pcd, output_path):
    """Save a point cloud and check the result."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_point_cloud(
        str(output_path),
        pcd,
    )

    if not success:
        raise RuntimeError(
            f"Could not save point cloud: {output_path}"
        )

    print(f"Saved: {output_path.relative_to(BASE_DIR)}")


def print_cloud_info(name, pcd):
    """Print basic information about a point cloud."""

    print(f"\n{name}")
    print("-" * 45)
    print(f"Points: {len(pcd.points):,}")

    if not pcd.is_empty():
        bbox = pcd.get_axis_aligned_bounding_box()

        print(
            "Bounding box dimensions: "
            f"{np.round(bbox.get_extent(), 5)}"
        )

        print(
            f"Has colors: {pcd.has_colors()}"
        )

        print(
            f"Has normals: {pcd.has_normals()}"
        )


# ============================================================
# VOXEL DOWNSAMPLING
# ============================================================

def voxel_downsample(pcd):
    """Reduce point density using voxel grid downsampling."""

    diagonal = get_diagonal(pcd)
    voxel_size = diagonal * VOXEL_SIZE_RATIO

    print("\n[1/5] Voxel Downsampling")
    print(f"Voxel size: {voxel_size:.6f}")

    downsampled = pcd.voxel_down_sample(
        voxel_size=voxel_size
    )

    if downsampled.is_empty():
        raise RuntimeError(
            "Voxel downsampling produced an empty cloud."
        )

    print(f"Before: {len(pcd.points):,} points")
    print(f"After:  {len(downsampled.points):,} points")

    return downsampled, voxel_size


# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

def statistical_filter(pcd):
    """
    Remove points whose average neighbor distance is unusually high.
    """

    print("\n[2/5] Statistical Outlier Removal")
    print(f"Neighbors: {STATISTICAL_NB_NEIGHBORS}")
    print(f"Standard deviation ratio: {STATISTICAL_STD_RATIO}")

    filtered, inlier_indices = (
        pcd.remove_statistical_outlier(
            nb_neighbors=STATISTICAL_NB_NEIGHBORS,
            std_ratio=STATISTICAL_STD_RATIO,
        )
    )

    print(f"Before: {len(pcd.points):,} points")
    print(f"After:  {len(filtered.points):,} points")
    print(f"Removed: {len(pcd.points) - len(filtered.points):,}")

    if filtered.is_empty():
        raise RuntimeError(
            "Statistical filtering removed all points. "
            "Try increasing STATISTICAL_STD_RATIO."
        )

    return filtered


# ============================================================
# RADIUS OUTLIER REMOVAL
# ============================================================

def radius_filter(pcd):
    """
    Remove points with too few neighbors within a given radius.
    """

    diagonal = get_diagonal(pcd)
    radius = diagonal * RADIUS_RATIO

    print("\n[3/5] Radius Outlier Removal")
    print(f"Radius: {radius:.6f}")
    print(f"Minimum neighbors: {RADIUS_NB_POINTS}")

    filtered, inlier_indices = pcd.remove_radius_outlier(
        nb_points=RADIUS_NB_POINTS,
        radius=radius,
    )

    print(f"Before: {len(pcd.points):,} points")
    print(f"After:  {len(filtered.points):,} points")
    print(f"Removed: {len(pcd.points) - len(filtered.points):,}")

    if filtered.is_empty():
        raise RuntimeError(
            "Radius filtering removed all points. "
            "Increase RADIUS_RATIO or decrease RADIUS_NB_POINTS."
        )

    return filtered, radius


# ============================================================
# NORMAL ESTIMATION
# ============================================================

def estimate_normals(pcd):
    """
    Estimate local surface normals using neighboring points.
    """

    diagonal = get_diagonal(pcd)
    normal_radius = diagonal * NORMAL_RADIUS_RATIO

    print("\n[4/5] Estimate Normals")
    print(f"Search radius: {normal_radius:.6f}")
    print(f"Maximum neighbors: {NORMAL_MAX_NN}")

    result = o3d.geometry.PointCloud(pcd)

    result.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=NORMAL_MAX_NN,
        )
    )

    # Orient normals consistently relative to the point cloud.
    result.orient_normals_consistent_tangent_plane(
        min(30, max(3, len(result.points) - 1))
    )

    print(f"Normals estimated: {result.has_normals()}")

    return result, normal_radius


# ============================================================
# ROI CROPPING
# ============================================================

def crop_roi(pcd):
    """
    Crop a region of interest using an axis-aligned bounding box.

    ROI limits are calculated as fractions of the current cloud's
    bounding-box dimensions.
    """

    print("\n[5/5] Crop Region of Interest (ROI)")

    original_bbox = pcd.get_axis_aligned_bounding_box()

    min_bound = np.asarray(original_bbox.min_bound)
    max_bound = np.asarray(original_bbox.max_bound)
    extent = max_bound - min_bound

    roi_min = min_bound + extent * np.asarray(ROI_MIN_RATIO)
    roi_max = min_bound + extent * np.asarray(ROI_MAX_RATIO)

    print(f"ROI minimum: {np.round(roi_min, 5)}")
    print(f"ROI maximum: {np.round(roi_max, 5)}")

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=roi_min,
        max_bound=roi_max,
    )

    cropped = pcd.crop(bbox)

    if cropped.is_empty():
        raise RuntimeError(
            "ROI crop produced an empty cloud. "
            "Adjust ROI_MIN_RATIO and ROI_MAX_RATIO."
        )

    print(f"Before crop: {len(pcd.points):,} points")
    print(f"After crop:  {len(cropped.points):,} points")

    return cropped, bbox


# ============================================================
# VISUALIZATION
# ============================================================

def get_display_indices(pcd, max_points=MAX_DISPLAY_POINTS):
    """Select points for plotting to keep figures manageable."""

    total = len(pcd.points)

    if total <= max_points:
        return np.arange(total)

    rng = np.random.default_rng(RANDOM_SEED)

    return rng.choice(
        total,
        size=max_points,
        replace=False,
    )


def plot_point_cloud(pcd, title, output_path):
    """Save a 3D scatter plot of a point cloud."""

    if pcd.is_empty():
        print(f"Skipping empty plot: {title}")
        return

    points = np.asarray(pcd.points)
    indices = get_display_indices(pcd)
    display_points = points[indices]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[indices]
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            c=colors,
            s=1,
        )
    else:
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            s=1,
        )

    ax.set_title(
        f"{title}\n"
        f"{len(pcd.points):,} points"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {output_path.relative_to(BASE_DIR)}")


def plot_normals(pcd, output_path):
    """Visualize a point cloud and a subset of its normal vectors."""

    if pcd.is_empty() or not pcd.has_normals():
        print("Cannot plot normals: cloud is empty or normals are missing.")
        return

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    indices = get_display_indices(pcd)
    display_points = points[indices]

    rng = np.random.default_rng(RANDOM_SEED)

    normal_count = min(
        NORMAL_DISPLAY_POINTS,
        len(display_points),
    )

    selected = rng.choice(
        len(display_points),
        size=normal_count,
        replace=False,
    )

    normal_points = display_points[selected]

    # Map the selected display points back to their original indices.
    original_indices = indices[selected]
    normal_vectors = normals[original_indices]

    arrow_length = (
        get_diagonal(pcd) * NORMAL_ARROW_LENGTH_RATIO
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        display_points[:, 0],
        display_points[:, 1],
        display_points[:, 2],
        s=1,
    )

    ax.quiver(
        normal_points[:, 0],
        normal_points[:, 1],
        normal_points[:, 2],
        normal_vectors[:, 0],
        normal_vectors[:, 1],
        normal_vectors[:, 2],
        length=arrow_length,
        normalize=True,
    )

    ax.set_title(
        f"Point Cloud with Estimated Normals\n"
        f"{len(pcd.points):,} points; "
        f"{normal_count} normals displayed"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved normals visualization: {output_path.relative_to(BASE_DIR)}")


def plot_comparison(
    original,
    cleaned,
    cropped,
    output_path,
):
    """Save a side-by-side comparison of the pipeline stages."""

    clouds = [
        ("Original", original),
        ("Cleaned", cleaned),
        ("ROI Cropped", cropped),
    ]

    fig = plt.figure(figsize=(16, 5))

    for position, (title, pcd) in enumerate(clouds, start=1):
        ax = fig.add_subplot(
            1,
            3,
            position,
            projection="3d",
        )

        if not pcd.is_empty():
            points = np.asarray(pcd.points)
            indices = get_display_indices(pcd)
            display_points = points[indices]

            if pcd.has_colors():
                colors = np.asarray(pcd.colors)[indices]
                ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    display_points[:, 2],
                    c=colors,
                    s=1,
                )
            else:
                ax.scatter(
                    display_points[:, 0],
                    display_points[:, 1],
                    display_points[:, 2],
                    s=1,
                )

        ax.set_title(
            f"{title}\n{len(pcd.points):,} points"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved pipeline comparison: {output_path.relative_to(BASE_DIR)}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("WEEK 3 — DAY 3: NORMALS + ROI + CLEANING PIPELINE")
    print("=" * 60)

    create_directories()

    # 1. Load
    original, cloud_name = load_point_cloud()
    print_cloud_info("ORIGINAL POINT CLOUD", original)

    save_point_cloud(
        original,
        RESULTS_DIR / f"{cloud_name}_original.ply",
    )

    # 2. Voxel downsampling
    downsampled, voxel_size = voxel_downsample(original)

    save_point_cloud(
        downsampled,
        CLEANED_DIR / f"{cloud_name}_voxel.ply",
    )

    # 3. Statistical filtering
    statistical = statistical_filter(downsampled)

    save_point_cloud(
        statistical,
        CLEANED_DIR / f"{cloud_name}_statistical.ply",
    )

    # 4. Radius filtering
    cleaned, radius = radius_filter(statistical)

    save_point_cloud(
        cleaned,
        CLEANED_DIR / f"{cloud_name}_cleaned.ply",
    )

    print_cloud_info("CLEANED POINT CLOUD", cleaned)

    # 5. Estimate normals
    with_normals, normal_radius = estimate_normals(cleaned)

    save_point_cloud(
        with_normals,
        NORMALS_DIR / f"{cloud_name}_with_normals.ply",
    )

    # 6. Crop ROI
    cropped, roi_bbox = crop_roi(with_normals)

    save_point_cloud(
        cropped,
        CROPPED_DIR / f"{cloud_name}_roi.ply",
    )

    print_cloud_info("CROPPED ROI", cropped)

    # Re-estimate normals after cropping to obtain normals
    # based on the retained region and its local neighborhood.
    cropped_with_normals, _ = estimate_normals(cropped)

    save_point_cloud(
        cropped_with_normals,
        CROPPED_DIR / f"{cloud_name}_roi_with_normals.ply",
    )

    # 7. Visualizations
    plot_point_cloud(
        original,
        "Original Point Cloud",
        RESULTS_DIR / "01_original.png",
    )

    plot_point_cloud(
        cleaned,
        "Cleaned Point Cloud",
        CLEANED_DIR / "02_cleaned.png",
    )

    plot_normals(
        with_normals,
        NORMALS_DIR / "03_normals.png",
    )

    plot_point_cloud(
        cropped_with_normals,
        "ROI Cropped Point Cloud",
        CROPPED_DIR / "04_roi.png",
    )

    plot_comparison(
        original,
        cleaned,
        cropped_with_normals,
        RESULTS_DIR / "05_pipeline_comparison.png",
    )

    # 8. Save configuration and point counts
    summary_path = RESULTS_DIR / "day3_summary.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("WEEK 3 — DAY 3: POINT CLOUD CLEANING\n")
        file.write("=" * 45 + "\n\n")

        file.write(f"Input cloud: {cloud_name}\n")
        file.write(f"Original points: {len(original.points)}\n")
        file.write(f"Voxel-downsampled points: {len(downsampled.points)}\n")
        file.write(f"After statistical filtering: {len(statistical.points)}\n")
        file.write(f"After radius filtering: {len(cleaned.points)}\n")
        file.write(f"After ROI cropping: {len(cropped_with_normals.points)}\n\n")

        file.write("Parameters\n")
        file.write("-" * 30 + "\n")
        file.write(f"Voxel size: {voxel_size:.8f}\n")
        file.write(f"Statistical neighbors: {STATISTICAL_NB_NEIGHBORS}\n")
        file.write(f"Statistical std_ratio: {STATISTICAL_STD_RATIO}\n")
        file.write(f"Radius: {radius:.8f}\n")
        file.write(f"Radius minimum neighbors: {RADIUS_NB_POINTS}\n")
        file.write(f"Normal search radius: {normal_radius:.8f}\n")
        file.write(f"Normal max neighbors: {NORMAL_MAX_NN}\n")
        file.write(f"ROI minimum ratios: {ROI_MIN_RATIO}\n")
        file.write(f"ROI maximum ratios: {ROI_MAX_RATIO}\n\n")

        file.write("ROI bounding box\n")
        file.write(f"Min bound: {roi_bbox.min_bound}\n")
        file.write(f"Max bound: {roi_bbox.max_bound}\n")

    print(f"\nSummary saved: {summary_path.relative_to(BASE_DIR)}")

    print("\n" + "=" * 60)
    print("DAY 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nMain outputs:")
    print(f"  Cleaned cloud: {CLEANED_DIR}")
    print(f"  Normals:       {NORMALS_DIR}")
    print(f"  ROI:           {CROPPED_DIR}")
    print(f"  Summary:       {summary_path}")


if __name__ == "__main__":
    main()