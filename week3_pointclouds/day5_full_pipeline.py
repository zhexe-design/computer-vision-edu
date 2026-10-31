"""
WEEK 3 — DAY 5
Full Point Cloud Cleaning Pipeline

Pipeline:
1. Load point cloud.
2. Voxel downsampling.
3. Statistical outlier removal.
4. Optional radius outlier removal.
5. Estimate normals.
6. Crop region of interest (ROI).
7. Save intermediate and final results.
8. Generate visualizations.
9. Print a processing summary.

Requirements:
    pip install open3d numpy

Run from the repository root:
    python week3_pointclouds/day5_full_pipeline.py

Or run this file directly from VS Code.
"""

from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
CLEANED_DIR = RESULTS_DIR / "cleaned"
NORMALS_DIR = RESULTS_DIR / "with_normals"
CROPPED_DIR = RESULTS_DIR / "cropped"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
SUMMARY_DIR = RESULTS_DIR / "summaries"


# Cleaning parameters.
# These are starting values for the Stanford Bunny dataset.

VOXEL_SIZE = 0.01

STATISTICAL_NB_NEIGHBORS = 20
STATISTICAL_STD_RATIO = 2.0

# Radius filtering is optional.
USE_RADIUS_FILTER = True

RADIUS_NB_POINTS = 8
RADIUS_SIZE = 0.025

# Normal estimation.
NORMAL_RADIUS = 0.075
NORMAL_MAX_NN = 30

# ROI boundaries as fractions of the bounding box.
# 0.0 = minimum coordinate, 1.0 = maximum coordinate.
ROI_MIN_RATIO = 0.10
ROI_MAX_RATIO = 0.90

# Visualization.
SHOW_WINDOWS = True
SAVE_VISUALIZATIONS = True

# Limit the number of normals drawn in the visualization.
MAX_NORMALS_TO_DISPLAY = 1500

# Supported input extensions.
SUPPORTED_EXTENSIONS = {
    ".ply",
    ".pcd",
    ".xyz",
    ".xyzn",
    ".xyzrgb",
    ".pts",
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create all output directories."""

    directories = [
        DATA_DIR,
        RESULTS_DIR,
        INTERMEDIATE_DIR,
        CLEANED_DIR,
        NORMALS_DIR,
        CROPPED_DIR,
        VISUALIZATION_DIR,
        SUMMARY_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILE DISCOVERY
# ============================================================

def find_input_file():
    """
    Find a point cloud in the data directory.

    An explicit command-line path takes priority:

        python week3_pointclouds/day5_full_pipeline.py data/bunny.ply

    Absolute paths and paths relative to the current working
    directory are supported.
    """

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).expanduser()

        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path

        input_path = input_path.resolve()

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input file does not exist: {input_path}"
            )

        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {input_path.suffix}\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        return input_path

    candidates = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not candidates:
        raise FileNotFoundError(
            "\nNo point cloud was found in the data directory.\n\n"
            f"Expected location:\n  {DATA_DIR}\n\n"
            "Place a .ply, .pcd, .xyz, .xyzn, .xyzrgb or .pts file "
            "in that directory.\n\n"
            "Alternatively, provide the input file as an argument:\n"
            "  python week3_pointclouds/day5_full_pipeline.py "
            "week3_pointclouds/data/bunny.ply"
        )

    if len(candidates) > 1:
        print("\nMultiple point clouds found:")

        for index, candidate in enumerate(candidates, start=1):
            print(f"  {index}. {candidate.name}")

        print(f"\nUsing the first file: {candidates[0].name}")

    return candidates[0]


# ============================================================
# LOADING
# ============================================================

def load_point_cloud(file_path):
    """Load a point cloud and validate its contents."""

    print("\n" + "=" * 65)
    print("STEP 1 — LOAD POINT CLOUD")
    print("=" * 65)

    print(f"File: {file_path}")

    pcd = o3d.io.read_point_cloud(str(file_path))

    if pcd.is_empty():
        raise ValueError(
            "The point cloud is empty or could not be read correctly."
        )

    points = np.asarray(pcd.points)

    if not np.isfinite(points).all():
        raise ValueError(
            "The input contains NaN or infinite coordinates. "
            "Clean these values before processing."
        )

    print(f"Points: {len(pcd.points):,}")
    print(f"Has colors: {pcd.has_colors()}")
    print(f"Has normals: {pcd.has_normals()}")

    print("\nBounding box:")
    print(pcd.get_axis_aligned_bounding_box())

    return pcd


# ============================================================
# SAVING
# ============================================================

def save_point_cloud(pcd, output_path):
    """Save a point cloud and verify that the write succeeded."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pcd.is_empty():
        print(f"[WARNING] Not saving an empty point cloud: {output_path}")
        return False

    success = o3d.io.write_point_cloud(
        str(output_path),
        pcd,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    if success:
        print(f"[SAVED] {output_path}")
        return True

    print(f"[WARNING] Could not save: {output_path}")
    return False


# ============================================================
# POINT CLOUD STATISTICS
# ============================================================

def get_point_count(pcd):
    """Return the number of points in a point cloud."""

    return len(pcd.points)


def print_point_count(label, pcd):
    """Print the point count for a processing stage."""

    print(f"{label:<35} {get_point_count(pcd):>10,} points")


def calculate_removal_percentage(before, after):
    """Calculate the percentage of removed points."""

    if before == 0:
        return 0.0

    return (before - after) / before * 100.0


# ============================================================
# STEP 2 — VOXEL DOWNSAMPLING
# ============================================================

def voxel_downsampling(pcd, voxel_size=VOXEL_SIZE):
    """
    Reduce point density using a voxel grid.

    Each occupied voxel contributes a representative point.
    This reduces processing time and can make density more uniform.
    """

    print("\n" + "=" * 65)
    print("STEP 2 — VOXEL DOWNSAMPLING")
    print("=" * 65)

    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")

    before = get_point_count(pcd)

    downsampled = pcd.voxel_down_sample(
        voxel_size=voxel_size
    )

    after = get_point_count(downsampled)

    print(f"Voxel size: {voxel_size}")
    print_point_count("Before:", pcd)
    print_point_count("After:", downsampled)

    print(
        f"Points reduced: "
        f"{calculate_removal_percentage(before, after):.2f}%"
    )

    return downsampled


# ============================================================
# STEP 3 — STATISTICAL OUTLIER REMOVAL
# ============================================================

def statistical_filter(
    pcd,
    nb_neighbors=STATISTICAL_NB_NEIGHBORS,
    std_ratio=STATISTICAL_STD_RATIO,
):
    """
    Remove points whose average neighbor distance is unusually high.

    nb_neighbors:
        Number of neighbors used to estimate local distances.

    std_ratio:
        Larger values are less aggressive.
        Smaller values remove more points.
    """

    print("\n" + "=" * 65)
    print("STEP 3 — STATISTICAL OUTLIER REMOVAL")
    print("=" * 65)

    if nb_neighbors < 1:
        raise ValueError("nb_neighbors must be at least 1.")

    if std_ratio <= 0:
        raise ValueError("std_ratio must be greater than zero.")

    before = get_point_count(pcd)

    if before <= nb_neighbors:
        print(
            "Not enough points for the configured neighborhood. "
            "Skipping statistical filtering."
        )
        return pcd

    filtered, indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    after = get_point_count(filtered)

    print(f"Neighbors: {nb_neighbors}")
    print(f"Standard deviation ratio: {std_ratio}")
    print_point_count("Before:", pcd)
    print_point_count("After:", filtered)

    print(
        f"Points removed: "
        f"{calculate_removal_percentage(before, after):.2f}%"
    )

    return filtered


# ============================================================
# STEP 4 — RADIUS OUTLIER REMOVAL
# ============================================================

def radius_filter(
    pcd,
    nb_points=RADIUS_NB_POINTS,
    radius=RADIUS_SIZE,
):
    """
    Remove points with too few neighbors within a given radius.

    This stage is optional because an overly small radius can
    remove valid points in sparse areas.
    """

    print("\n" + "=" * 65)
    print("STEP 4 — RADIUS OUTLIER REMOVAL")
    print("=" * 65)

    if nb_points < 1:
        raise ValueError("nb_points must be at least 1.")

    if radius <= 0:
        raise ValueError("radius must be greater than zero.")

    before = get_point_count(pcd)

    if before <= nb_points:
        print(
            "Not enough points for the configured radius filter. "
            "Skipping radius filtering."
        )
        return pcd

    filtered, indices = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )

    after = get_point_count(filtered)

    print(f"Minimum neighbors: {nb_points}")
    print(f"Radius: {radius}")
    print_point_count("Before:", pcd)
    print_point_count("After:", filtered)

    print(
        f"Points removed: "
        f"{calculate_removal_percentage(before, after):.2f}%"
    )

    return filtered


# ============================================================
# STEP 5 — NORMAL ESTIMATION
# ============================================================

def estimate_normals(
    pcd,
    radius=NORMAL_RADIUS,
    max_nn=NORMAL_MAX_NN,
):
    """
    Estimate local surface normals for each point.

    Normals describe local surface orientation and are useful
    for point-to-plane ICP and surface reconstruction.

    Important:
        Normal directions may not be globally consistent.
        This function estimates normals but does not guarantee
        their orientation across the entire point cloud.
    """

    print("\n" + "=" * 65)
    print("STEP 5 — ESTIMATE NORMALS")
    print("=" * 65)

    if radius <= 0:
        raise ValueError("Normal radius must be greater than zero.")

    if max_nn < 3:
        raise ValueError("max_nn must be at least 3.")

    if get_point_count(pcd) < 3:
        print("Not enough points to estimate normals.")
        return pcd

    pcd_with_normals = o3d.geometry.PointCloud(pcd)

    pcd_with_normals.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )

    # Improve local consistency. This does not guarantee globally
    # consistent orientation for disconnected surfaces.
    pcd_with_normals.orient_normals_consistent_tangent_plane(
        min(30, max(3, get_point_count(pcd_with_normals) - 1))
    )

    print(f"Search radius: {radius}")
    print(f"Maximum neighbors: {max_nn}")
    print(f"Normals estimated: {pcd_with_normals.has_normals()}")

    return pcd_with_normals


# ============================================================
# STEP 6 — ROI CROPPING
# ============================================================

def crop_roi(
    pcd,
    min_ratio=ROI_MIN_RATIO,
    max_ratio=ROI_MAX_RATIO,
):
    """
    Crop a central region of interest from the axis-aligned
    bounding box.

    Ratios are calculated independently for X, Y and Z.

    Example:
        min_ratio=0.1, max_ratio=0.9

    retains the central 80% of each bounding-box dimension.

    This is a demonstration ROI, not an object-aware segmentation.
    """

    print("\n" + "=" * 65)
    print("STEP 6 — CROP REGION OF INTEREST")
    print("=" * 65)

    if not 0 <= min_ratio < max_ratio <= 1:
        raise ValueError(
            "ROI ratios must satisfy 0 <= min_ratio < max_ratio <= 1."
        )

    if pcd.is_empty():
        raise ValueError("Cannot crop an empty point cloud.")

    points = np.asarray(pcd.points)

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    dimensions = max_bound - min_bound

    roi_min = min_bound + dimensions * min_ratio
    roi_max = min_bound + dimensions * max_ratio

    # A zero-width axis cannot be cropped using a strict interior
    # ratio. Keep its complete extent instead.
    zero_axes = dimensions <= 1e-12
    roi_min[zero_axes] = min_bound[zero_axes]
    roi_max[zero_axes] = max_bound[zero_axes]

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=roi_min,
        max_bound=roi_max,
    )

    cropped = pcd.crop(bbox)

    print(f"Minimum ratio: {min_ratio}")
    print(f"Maximum ratio: {max_ratio}")
    print(f"ROI minimum: {roi_min}")
    print(f"ROI maximum: {roi_max}")
    print_point_count("Before:", pcd)
    print_point_count("After cropping:", cropped)

    if cropped.is_empty():
        print(
            "[WARNING] ROI is empty. Consider using a wider ROI "
            "or checking the point cloud."
        )

    return cropped


# ============================================================
# VISUALIZATION
# ============================================================

def prepare_visualization(pcd, normals=False):
    """Create a visualization copy with a neutral point color."""

    visual = o3d.geometry.PointCloud(pcd)

    if not visual.has_colors():
        visual.paint_uniform_color([0.65, 0.70, 0.78])

    if normals and not visual.has_normals():
        visual = estimate_normals(visual)

    return visual


def save_visualization(pcd, output_path, title, normals=False):
    """
    Save a screenshot using Open3D's visualizer.

    This requires a working graphical environment. If screenshots
    cannot be generated, the point cloud files are still saved.
    """

    if pcd.is_empty():
        print(f"[WARNING] Cannot visualize empty cloud: {title}")
        return False

    visual = prepare_visualization(pcd, normals=normals)

    visualizer = o3d.visualization.Visualizer()

    try:
        created = visualizer.create_window(
            window_name=title,
            width=1280,
            height=800,
            visible=True,
        )

        if not created:
            print(f"[WARNING] Could not create window: {title}")
            return False

        visualizer.add_geometry(visual)

        options = visualizer.get_render_option()
        options.point_size = 2.0
        options.background_color = np.asarray([0.08, 0.08, 0.10])

        if normals:
            options.point_show_normal = True

        visualizer.reset_view_point(True)

        # Give the renderer time to initialize.
        for _ in range(5):
            visualizer.poll_events()
            visualizer.update_renderer()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        visualizer.capture_screen_image(
            str(output_path),
            do_render=True,
        )

        print(f"[SAVED] Visualization: {output_path}")

        if SHOW_WINDOWS:
            print(
                "Close the visualization window to continue "
                "to the next screenshot."
            )

            while visualizer.poll_events():
                visualizer.update_renderer()

        return True

    except Exception as error:
        print(
            f"[WARNING] Visualization failed for '{title}': {error}"
        )
        return False

    finally:
        visualizer.destroy_window()


def generate_visualizations(
    original,
    cleaned,
    with_normals,
    cropped,
):
    """Save before/after, normals and ROI screenshots."""

    if not SAVE_VISUALIZATIONS:
        print("\nVisualization saving is disabled.")
        return

    print("\n" + "=" * 65)
    print("STEP 7 — SAVE VISUALIZATIONS")
    print("=" * 65)

    # Use a temporary copy for visualization only.
    # Saved processing results are not changed by this operation.

    save_visualization(
        original,
        VISUALIZATION_DIR / "01_original.png",
        "Original Point Cloud",
    )

    save_visualization(
        cleaned,
        VISUALIZATION_DIR / "02_cleaned.png",
        "Cleaned Point Cloud",
    )

    save_visualization(
        with_normals,
        VISUALIZATION_DIR / "03_with_normals.png",
        "Point Cloud with Normals",
        normals=True,
    )

    if not cropped.is_empty():
        save_visualization(
            cropped,
            VISUALIZATION_DIR / "04_cropped_roi.png",
            "Cropped Region of Interest",
        )


# ============================================================
# SUMMARY REPORT
# ============================================================

def write_summary(
    input_path,
    original,
    downsampled,
    statistical,
    cleaned,
    with_normals,
    cropped,
):
    """Write a reproducible summary of the processing pipeline."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stages = [
        ("Original", original),
        ("Voxel downsampling", downsampled),
        ("Statistical filtering", statistical),
        ("Final cleaned cloud", cleaned),
        ("Cloud with normals", with_normals),
        ("Cropped ROI", cropped),
    ]

    report_lines = [
        "WEEK 3 — DAY 5",
        "FULL POINT CLOUD CLEANING PIPELINE",
        "=" * 55,
        f"Run timestamp: {timestamp}",
        f"Input file: {input_path}",
        "",
        "PARAMETERS",
        "-" * 55,
        f"Voxel size: {VOXEL_SIZE}",
        f"Statistical nb_neighbors: {STATISTICAL_NB_NEIGHBORS}",
        f"Statistical std_ratio: {STATISTICAL_STD_RATIO}",
        f"Radius filtering enabled: {USE_RADIUS_FILTER}",
        f"Radius nb_points: {RADIUS_NB_POINTS}",
        f"Radius: {RADIUS_SIZE}",
        f"Normal radius: {NORMAL_RADIUS}",
        f"Normal max_nn: {NORMAL_MAX_NN}",
        f"ROI min ratio: {ROI_MIN_RATIO}",
        f"ROI max ratio: {ROI_MAX_RATIO}",
        "",
        "POINT COUNTS",
        "-" * 55,
    ]

    for label, cloud in stages:
        report_lines.append(
            f"{label:<30} {get_point_count(cloud):>10,}"
        )

    report_lines.extend([
        "",
        "REMOVAL PERCENTAGES",
        "-" * 55,
        (
            "Voxel downsampling: "
            f"{calculate_removal_percentage(len(original.points), len(downsampled.points)):.2f}%"
        ),
        (
            "Statistical filtering: "
            f"{calculate_removal_percentage(len(downsampled.points), len(statistical.points)):.2f}%"
        ),
        (
            "Final cleaning after downsampling: "
            f"{calculate_removal_percentage(len(downsampled.points), len(cleaned.points)):.2f}%"
        ),
        "",
        "NOTES",
        "-" * 55,
        "- Normal estimation does not change the point count.",
        "- ROI cropping uses an axis-aligned bounding box.",
        "- ROI cropping is based on coordinate ratios, not object semantics.",
        "- Inspect visualizations before using the cloud for ICP or meshing.",
        "",
    ])

    report = "\n".join(report_lines)

    summary_path = SUMMARY_DIR / "day5_pipeline_summary.txt"
    summary_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 65)
    print("PROCESSING SUMMARY")
    print("=" * 65)
    print(report)
    print(f"\nSummary saved to: {summary_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Execute the complete point cloud cleaning pipeline."""

    print("=" * 65)
    print("WEEK 3 — DAY 5: FULL POINT CLOUD PIPELINE")
    print("=" * 65)

    create_directories()

    input_path = find_input_file()
    original = load_point_cloud(input_path)

    # --------------------------------------------------------
    # Stage 1: Voxel downsampling
    # --------------------------------------------------------

    downsampled = voxel_downsampling(
        original,
        voxel_size=VOXEL_SIZE,
    )

    save_point_cloud(
        downsampled,
        INTERMEDIATE_DIR / "01_voxel_downsampled.ply",
    )

    # --------------------------------------------------------
    # Stage 2: Statistical filtering
    # --------------------------------------------------------

    statistical = statistical_filter(
        downsampled,
        nb_neighbors=STATISTICAL_NB_NEIGHBORS,
        std_ratio=STATISTICAL_STD_RATIO,
    )

    save_point_cloud(
        statistical,
        INTERMEDIATE_DIR / "02_statistical_filtered.ply",
    )

    # --------------------------------------------------------
    # Stage 3: Optional radius filtering
    # --------------------------------------------------------

    if USE_RADIUS_FILTER:
        cleaned = radius_filter(
            statistical,
            nb_points=RADIUS_NB_POINTS,
            radius=RADIUS_SIZE,
        )
    else:
        print("\nRadius filtering is disabled.")
        cleaned = o3d.geometry.PointCloud(statistical)

    save_point_cloud(
        cleaned,
        CLEANED_DIR / "cleaned_pointcloud.ply",
    )

    # --------------------------------------------------------
    # Stage 4: Estimate normals
    # --------------------------------------------------------

    with_normals = estimate_normals(
        cleaned,
        radius=NORMAL_RADIUS,
        max_nn=NORMAL_MAX_NN,
    )

    save_point_cloud(
        with_normals,
        NORMALS_DIR / "pointcloud_with_normals.ply",
    )

    # --------------------------------------------------------
    # Stage 5: Crop ROI
    # --------------------------------------------------------

    cropped = crop_roi(
        with_normals,
        min_ratio=ROI_MIN_RATIO,
        max_ratio=ROI_MAX_RATIO,
    )

    save_point_cloud(
        cropped,
        CROPPED_DIR / "cropped_roi.ply",
    )

    # --------------------------------------------------------
    # Stage 6: Save all visualizations
    # --------------------------------------------------------

    generate_visualizations(
        original=original,
        cleaned=cleaned,
        with_normals=with_normals,
        cropped=cropped,
    )

    # --------------------------------------------------------
    # Stage 7: Save summary
    # --------------------------------------------------------

    write_summary(
        input_path=input_path,
        original=original,
        downsampled=downsampled,
        statistical=statistical,
        cleaned=cleaned,
        with_normals=with_normals,
        cropped=cropped,
    )

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED")
    print("=" * 65)

    print("\nFinal outputs:")
    print(f"  Cleaned cloud: {CLEANED_DIR}")
    print(f"  Normals:       {NORMALS_DIR}")
    print(f"  Cropped ROI:   {CROPPED_DIR}")
    print(f"  Visualizations:{VISUALIZATION_DIR}")
    print(f"  Summary:       {SUMMARY_DIR}")

    print("\nWeek 3 Day 5 pipeline finished successfully.")


if __name__ == "__main__":
    try:
        main()

    except (FileNotFoundError, ValueError) as error:
        print(f"\n[ERROR] {error}")
        sys.exit(1)

    except Exception as error:
        print(f"\n[UNEXPECTED ERROR] {error}")
        raise