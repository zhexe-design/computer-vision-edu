"""
WEEK 3 — DAY 2: POINT CLOUD FILTERING

Methods:
1. Voxel Downsampling
2. Statistical Outlier Removal
3. Radius Outlier Removal

Outputs:
- Filtered point clouds (.ply)
- Before/after comparison images (.png)
- Experiment summary (.txt)

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

STATISTICAL_DIR = RESULTS_DIR / "statistical"
RADIUS_DIR = RESULTS_DIR / "radius"

# Voxel size for downsampling.
VOXEL_SIZE = 0.01

# Add synthetic outliers to make the experiment more challenging.
# Set to 0 to disable.
OUTLIER_RATIO = 0.03

# Number of points displayed in comparison images.
MAX_PLOT_POINTS = 5000

# Random seed for reproducible experiments.
RANDOM_SEED = 42

# Supported file formats.
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
    """Create directories for experiment results."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    STATISTICAL_DIR.mkdir(parents=True, exist_ok=True)
    RADIUS_DIR.mkdir(parents=True, exist_ok=True)

    print("Output directories are ready.")


# ============================================================
# LOAD POINT CLOUD
# ============================================================

def load_point_cloud():
    """
    Load a point cloud from the data directory.

    If no supported file is found, generate a point cloud
    from the Stanford Bunny mesh.
    """

    files = sorted(
        file
        for file in DATA_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if files:
        file_path = files[0]

        print(f"\nLoading point cloud: {file_path.name}")

        pcd = o3d.io.read_point_cloud(str(file_path))

        if pcd.is_empty():
            raise ValueError(
                f"Could not load a valid point cloud from {file_path}"
            )

        print(f"Loaded {len(pcd.points):,} points.")

        return pcd, file_path.stem

    print("\nNo point cloud found in data/.")
    print("Loading the Stanford Bunny mesh instead...")

    bunny = o3d.data.BunnyMesh()

    mesh = o3d.io.read_triangle_mesh(bunny.path)

    if mesh.is_empty():
        raise ValueError("Failed to load the Stanford Bunny mesh.")

    # Generate a point cloud from the mesh.
    np.random.seed(RANDOM_SEED)
    pcd = mesh.sample_points_uniformly(
        number_of_points=30000,
    )

    print(f"Generated {len(pcd.points):,} points.")

    return pcd, "stanford_bunny"


# ============================================================
# ADD SYNTHETIC OUTLIERS
# ============================================================

def add_outliers(pcd, ratio=OUTLIER_RATIO):
    """
    Add randomly distributed points outside the original
    cloud to simulate noise and outliers.

    Existing colors are preserved for original points.
    Synthetic outliers are colored red.
    """

    if ratio <= 0:
        return pcd

    points = np.asarray(pcd.points)

    if len(points) == 0:
        return pcd

    rng = np.random.default_rng(RANDOM_SEED)

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    extent = maximum - minimum

    # Avoid a zero-width range on any axis.
    extent[extent == 0] = 1e-6

    number_of_outliers = max(
        1,
        int(len(points) * ratio),
    )

    # Generate outliers in an expanded bounding box.
    noise_min = minimum - 0.25 * extent
    noise_max = maximum + 0.25 * extent

    noise_points = rng.uniform(
        low=noise_min,
        high=noise_max,
        size=(number_of_outliers, 3),
    )

    original_colors = np.asarray(pcd.colors)

    if len(original_colors) == len(points):
        colors = original_colors
    else:
        colors = np.full(
            (len(points), 3),
            0.7,
            dtype=float,
        )

    noise_colors = np.tile(
        np.array([[1.0, 0.0, 0.0]]),
        (number_of_outliers, 1),
    )

    all_points = np.vstack(
        [points, noise_points]
    )

    all_colors = np.vstack(
        [colors, noise_colors]
    )

    noisy_pcd = o3d.geometry.PointCloud()

    noisy_pcd.points = o3d.utility.Vector3dVector(
        all_points
    )

    noisy_pcd.colors = o3d.utility.Vector3dVector(
        all_colors
    )

    print(
        f"Added {number_of_outliers:,} synthetic outliers."
    )

    return noisy_pcd


# ============================================================
# VOXEL DOWNSAMPLING
# ============================================================

def voxel_downsample(pcd, voxel_size=VOXEL_SIZE):
    """
    Reduce point density using voxel downsampling.
    """

    print("\nApplying voxel downsampling...")

    downsampled = pcd.voxel_down_sample(
        voxel_size=voxel_size
    )

    print(f"Before: {len(pcd.points):,} points")
    print(f"After:  {len(downsampled.points):,} points")

    return downsampled


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def sample_points_for_plot(pcd, max_points=MAX_PLOT_POINTS):
    """
    Randomly select points for visualization.

    This keeps comparison images manageable.
    The actual saved point clouds are not modified.
    """

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        return points, colors

    if len(points) > max_points:
        rng = np.random.default_rng(RANDOM_SEED)

        indices = rng.choice(
            len(points),
            size=max_points,
            replace=False,
        )

        points = points[indices]

        if len(colors) == len(np.asarray(pcd.points)):
            colors = colors[indices]

    if len(colors) != len(points):
        colors = np.full(
            (len(points), 3),
            0.7,
        )

    return points, colors


def draw_point_cloud(ax, pcd, title):
    """
    Draw a point cloud using a Matplotlib 3D scatter plot.
    """

    points, colors = sample_points_for_plot(pcd)

    if len(points) == 0:
        ax.set_title(f"{title}\n(empty)")
        return

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=1,
        depthshade=False,
    )

    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Keep the coordinate scale approximately equal.
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    center = (minimum + maximum) / 2
    radius = max((maximum - minimum).max() / 2, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_comparison(original, filtered, output_path, title):
    """
    Save a before/after comparison image.
    """

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    draw_point_cloud(
        ax1,
        original,
        f"Before filtering\n{len(original.points):,} points",
    )

    draw_point_cloud(
        ax2,
        filtered,
        f"After filtering\n{len(filtered.points):,} points",
    )

    fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved comparison: {output_path.name}")


# ============================================================
# SAVE POINT CLOUD
# ============================================================

def save_point_cloud(pcd, output_path):
    """Save a point cloud to a PLY file."""

    success = o3d.io.write_point_cloud(
        str(output_path),
        pcd,
    )

    if not success:
        raise IOError(
            f"Failed to save point cloud: {output_path}"
        )

    print(f"Saved point cloud: {output_path.name}")


# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

def run_statistical_experiments(pcd):
    """
    Compare Statistical Outlier Removal with different
    neighborhood sizes and standard deviation ratios.
    """

    print("\n" + "=" * 60)
    print("STATISTICAL OUTLIER REMOVAL")
    print("=" * 60)

    experiments = [
        {
            "name": "statistical_conservative",
            "nb_neighbors": 20,
            "std_ratio": 3.0,
        },
        {
            "name": "statistical_balanced",
            "nb_neighbors": 20,
            "std_ratio": 2.0,
        },
        {
            "name": "statistical_aggressive",
            "nb_neighbors": 20,
            "std_ratio": 1.0,
        },
        {
            "name": "statistical_neighbors_10",
            "nb_neighbors": 10,
            "std_ratio": 2.0,
        },
        {
            "name": "statistical_neighbors_40",
            "nb_neighbors": 40,
            "std_ratio": 2.0,
        },
    ]

    results = []

    for config in experiments:
        name = config["name"]

        nb_neighbors = config["nb_neighbors"]
        std_ratio = config["std_ratio"]

        print(f"\nExperiment: {name}")

        filtered, indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

        removed = len(pcd.points) - len(filtered.points)

        removed_percent = (
            removed / len(pcd.points) * 100
            if len(pcd.points) > 0
            else 0
        )

        print(f"Neighbors: {nb_neighbors}")
        print(f"Std ratio: {std_ratio}")
        print(f"Points kept: {len(filtered.points):,}")
        print(f"Points removed: {removed:,}")
        print(f"Removed: {removed_percent:.2f}%")

        save_point_cloud(
            filtered,
            STATISTICAL_DIR / f"{name}.ply",
        )

        save_comparison(
            pcd,
            filtered,
            STATISTICAL_DIR / f"{name}_comparison.png",
            f"Statistical Outlier Removal: {name}",
        )

        results.append({
            "name": name,
            "method": "Statistical",
            "parameters": (
                f"nb_neighbors={nb_neighbors}, "
                f"std_ratio={std_ratio}"
            ),
            "points_before": len(pcd.points),
            "points_after": len(filtered.points),
            "removed": removed,
            "removed_percent": removed_percent,
            "cloud": filtered,
        })

    return results


# ============================================================
# RADIUS OUTLIER REMOVAL
# ============================================================

def run_radius_experiments(pcd):
    """
    Compare Radius Outlier Removal with different
    radius and minimum neighbor counts.
    """

    print("\n" + "=" * 60)
    print("RADIUS OUTLIER REMOVAL")
    print("=" * 60)

    experiments = [
        {
            "name": "radius_conservative",
            "nb_points": 8,
            "radius": 0.05,
        },
        {
            "name": "radius_balanced",
            "nb_points": 16,
            "radius": 0.05,
        },
        {
            "name": "radius_aggressive",
            "nb_points": 24,
            "radius": 0.05,
        },
        {
            "name": "radius_small",
            "nb_points": 16,
            "radius": 0.025,
        },
        {
            "name": "radius_large",
            "nb_points": 16,
            "radius": 0.10,
        },
    ]

    results = []

    for config in experiments:
        name = config["name"]

        nb_points = config["nb_points"]
        radius = config["radius"]

        print(f"\nExperiment: {name}")

        filtered, indices = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius,
        )

        removed = len(pcd.points) - len(filtered.points)

        removed_percent = (
            removed / len(pcd.points) * 100
            if len(pcd.points) > 0
            else 0
        )

        print(f"Minimum neighbors: {nb_points}")
        print(f"Radius: {radius}")
        print(f"Points kept: {len(filtered.points):,}")
        print(f"Points removed: {removed:,}")
        print(f"Removed: {removed_percent:.2f}%")

        save_point_cloud(
            filtered,
            RADIUS_DIR / f"{name}.ply",
        )

        save_comparison(
            pcd,
            filtered,
            RADIUS_DIR / f"{name}_comparison.png",
            f"Radius Outlier Removal: {name}",
        )

        results.append({
            "name": name,
            "method": "Radius",
            "parameters": (
                f"nb_points={nb_points}, "
                f"radius={radius}"
            ),
            "points_before": len(pcd.points),
            "points_after": len(filtered.points),
            "removed": removed,
            "removed_percent": removed_percent,
            "cloud": filtered,
        })

    return results


# ============================================================
# SAVE EXPERIMENT SUMMARY
# ============================================================

def save_summary(results, original_count, downsampled_count):
    """
    Save experiment parameters and point counts to a text file.

    The script does not automatically declare the most
    aggressive filter to be the best one.
    """

    output_path = RESULTS_DIR / "experiment_summary.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("WEEK 3 — DAY 2: POINT CLOUD FILTERING\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Original points: {original_count:,}\n")
        file.write(f"After voxel downsampling: {downsampled_count:,}\n")
        file.write(f"Voxel size: {VOXEL_SIZE}\n\n")

        file.write("EXPERIMENT RESULTS\n")
        file.write("-" * 60 + "\n")

        for result in results:
            file.write(f"\nName: {result['name']}\n")
            file.write(f"Method: {result['method']}\n")
            file.write(f"Parameters: {result['parameters']}\n")
            file.write(
                f"Points before: {result['points_before']:,}\n"
            )
            file.write(
                f"Points after: {result['points_after']:,}\n"
            )
            file.write(f"Points removed: {result['removed']:,}\n")
            file.write(
                f"Removed percentage: "
                f"{result['removed_percent']:.2f}%\n"
            )

        file.write("\n\nHOW TO CHOOSE THE BEST RESULT\n")
        file.write("-" * 60 + "\n")
        file.write(
            "Compare the saved images and point clouds.\n"
            "Prefer a result that removes isolated noise while\n"
            "preserving the object's shape and fine details.\n"
            "The largest percentage of removed points does not\n"
            "necessarily indicate the best result.\n"
        )

    print(f"\nSummary saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("WEEK 3 — DAY 2: POINT CLOUD FILTERING")
    print("=" * 60)

    create_directories()

    # 1. Load a point cloud.
    pcd, cloud_name = load_point_cloud()

    original_count = len(pcd.points)

    # 2. Add synthetic outliers to simulate a noisy cloud.
    noisy_pcd = add_outliers(pcd)

    print(f"\nPoint cloud: {cloud_name}")
    print(f"Points before noise: {original_count:,}")
    print(f"Points after noise: {len(noisy_pcd.points):,}")

    # Save the noisy input.
    save_point_cloud(
        noisy_pcd,
        RESULTS_DIR / "noisy_original.ply",
    )

    # 3. Voxel downsampling.
    downsampled = voxel_downsample(noisy_pcd)

    downsampled_count = len(downsampled.points)

    save_point_cloud(
        downsampled,
        RESULTS_DIR / "voxel_downsampled.ply",
    )

    save_comparison(
        noisy_pcd,
        downsampled,
        RESULTS_DIR / "voxel_comparison.png",
        "Voxel Downsampling",
    )

    # 4. Statistical Outlier Removal experiments.
    statistical_results = run_statistical_experiments(
        downsampled
    )

    # 5. Radius Outlier Removal experiments.
    radius_results = run_radius_experiments(
        downsampled
    )

    # 6. Save the experiment summary.
    all_results = (
        statistical_results + radius_results
    )

    save_summary(
        all_results,
        original_count,
        downsampled_count,
    )

    print("\n" + "=" * 60)
    print("DAY 2 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Statistical results: {STATISTICAL_DIR}")
    print(f"Radius results: {RADIUS_DIR}")

    print("\nNext steps:")
    print("1. Open the comparison PNG images.")
    print("2. Compare noise removal and preserved details.")
    print("3. Select the best parameter combination.")
    print("4. Record your conclusions in README.md.")


if __name__ == "__main__":
    main()