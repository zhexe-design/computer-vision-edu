"""
WEEK 3 — DAY 4: ADVANCED POINT CLOUD CLEANING

Methods:
1. Voxel Downsampling
2. Statistical Outlier Removal
3. Radius Outlier Removal

Experiments:
- Combination 1: moderate cleaning
- Combination 2: stronger cleaning
- Combination 3: conservative cleaning

Outputs:
- Intermediate point clouds
- Final cleaned point clouds
- Visual comparisons
- Point count report
- Best configuration report
"""

from pathlib import Path
import copy
import csv
import sys

import numpy as np
import open3d as o3d
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

CLEANED_V1_DIR = RESULTS_DIR / "cleaned_v1"
CLEANED_V2_DIR = RESULTS_DIR / "cleaned_v2"
COMPARISON_DIR = RESULTS_DIR / "comparison"

for directory in (
    DATA_DIR,
    CLEANED_V1_DIR,
    CLEANED_V2_DIR,
    COMPARISON_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# Set to a filename such as "bunny.ply" to select a specific file.
# Leave as None to search automatically.
INPUT_FILE = None

# Set True to add artificial outliers for a more challenging test.
ADD_ARTIFICIAL_NOISE = True

# Reproducible random noise.
RANDOM_SEED = 42

# Limit the number of points displayed in plots.
MAX_PLOT_POINTS = 15000


# ============================================================
# 2. LOAD POINT CLOUD
# ============================================================

def load_point_cloud():
    """Load a point cloud from data/ or use the Open3D sample."""

    supported_extensions = {".ply", ".pcd", ".xyz", ".xyzn", ".pts"}

    if INPUT_FILE is not None:
        input_path = DATA_DIR / INPUT_FILE

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        candidates = [input_path]

    else:
        candidates = sorted(
            path
            for path in DATA_DIR.iterdir()
            if path.suffix.lower() in supported_extensions
        )

    if candidates:
        input_path = candidates[0]

        print(f"Loading point cloud: {input_path.name}")

        point_cloud = o3d.io.read_point_cloud(str(input_path))

        if point_cloud.is_empty():
            raise ValueError(
                f"The point cloud is empty or could not be read: "
                f"{input_path}"
            )

        return point_cloud, input_path.stem

    print("No input point cloud found in data/.")
    print("Trying the Open3D sample point cloud...")

    try:
        sample = o3d.data.PCDPointCloud()
        point_cloud = o3d.io.read_point_cloud(sample.path)

        if point_cloud.is_empty():
            raise ValueError("The Open3D sample is empty.")

        print(f"Loaded Open3D sample: {sample.path}")

        return point_cloud, "open3d_sample"

    except Exception as error:
        raise RuntimeError(
            "Could not load a point cloud.\n"
            f"1. Put a .ply, .pcd, .xyz or .pts file in:\n"
            f"   {DATA_DIR}\n"
            "2. Or set INPUT_FILE to your file's name.\n"
            "3. The Open3D fallback may require an internet connection "
            "to download the sample."
        ) from error


# ============================================================
# 3. ADD ARTIFICIAL OUTLIERS
# ============================================================

def add_artificial_noise(point_cloud, noise_ratio=0.05):
    """
    Add randomly distributed points around the cloud's bounding box.

    This simulates isolated measurement errors.

    noise_ratio=0.05 means approximately 5% as many added points
    as the original cloud contains.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    original_points = np.asarray(point_cloud.points)

    if len(original_points) == 0:
        return point_cloud

    min_bound = original_points.min(axis=0)
    max_bound = original_points.max(axis=0)

    extent = max_bound - min_bound

    # Avoid zero-sized dimensions.
    extent = np.maximum(extent, 1e-6)

    noise_count = max(
        1,
        int(len(original_points) * noise_ratio),
    )

    # Place outliers in a larger box around the object.
    noise_points = rng.uniform(
        low=min_bound - 0.35 * extent,
        high=max_bound + 0.35 * extent,
        size=(noise_count, 3),
    )

    noisy_cloud = copy.deepcopy(point_cloud)

    combined_points = np.vstack(
        [original_points, noise_points]
    )

    noisy_cloud.points = o3d.utility.Vector3dVector(
        combined_points
    )

    # Assign neutral colors so all points remain visible.
    if point_cloud.has_colors():
        original_colors = np.asarray(point_cloud.colors)

        noise_colors = np.tile(
            np.array([[1.0, 0.2, 0.2]]),
            (noise_count, 1),
        )

        noisy_cloud.colors = o3d.utility.Vector3dVector(
            np.vstack([original_colors, noise_colors])
        )
    else:
        noisy_cloud.paint_uniform_color([0.65, 0.65, 0.65])

    print(f"Artificial outliers added: {noise_count:,}")

    return noisy_cloud


# ============================================================
# 4. SAVE POINT CLOUD
# ============================================================

def save_point_cloud(point_cloud, output_path):
    """Save a point cloud and check whether writing succeeded."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_point_cloud(
        str(output_path),
        point_cloud,
    )

    if not success:
        raise IOError(
            f"Could not save point cloud: {output_path}"
        )

    print(f"Saved: {output_path.relative_to(BASE_DIR)}")


# ============================================================
# 5. VISUALIZATION
# ============================================================

def prepare_plot_points(point_cloud, max_points=MAX_PLOT_POINTS):
    """Reduce plotting workload without modifying the actual cloud."""

    points = np.asarray(point_cloud.points)

    if len(points) > max_points:
        rng = np.random.default_rng(RANDOM_SEED)

        indices = rng.choice(
            len(points),
            size=max_points,
            replace=False,
        )

        points = points[indices]

    return points


def save_comparison_plot(
    clouds,
    titles,
    output_path,
    max_points=MAX_PLOT_POINTS,
):
    """
    Save a side-by-side 3D visualization.

    Each panel uses the same viewing angle.
    Plot sampling affects visualization only, not filtering.
    """

    count = len(clouds)

    fig = plt.figure(figsize=(6 * count, 5))

    for index, (point_cloud, title) in enumerate(
        zip(clouds, titles),
        start=1,
    ):
        ax = fig.add_subplot(
            1,
            count,
            index,
            projection="3d",
        )

        points = prepare_plot_points(
            point_cloud,
            max_points=max_points,
        )

        if len(points) > 0:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=0.5,
                alpha=0.7,
            )

        ax.set_title(
            f"{title}\n"
            f"{len(point_cloud.points):,} points"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved comparison: {output_path.name}")


# ============================================================
# 6. CLEANING PIPELINE
# ============================================================

def run_cleaning_pipeline(
    source_cloud,
    name,
    voxel_size,
    nb_neighbors,
    std_ratio,
    nb_points,
    radius,
    output_directory,
):
    """
    Execute a three-stage cleaning pipeline.

    Stage 1: Voxel Downsampling
    Stage 2: Statistical Outlier Removal
    Stage 3: Radius Outlier Removal
    """

    print("\n" + "=" * 65)
    print(f"PIPELINE: {name}")
    print("=" * 65)

    original_count = len(source_cloud.points)

    # --------------------------------------------------------
    # Stage 1: Voxel Downsampling
    # --------------------------------------------------------

    downsampled = source_cloud.voxel_down_sample(
        voxel_size=voxel_size
    )

    downsampled_count = len(downsampled.points)

    print(
        f"1. Voxel Downsampling: "
        f"{original_count:,} -> {downsampled_count:,}"
    )

    save_point_cloud(
        downsampled,
        output_directory / f"{name}_01_downsampled.ply",
    )

    # --------------------------------------------------------
    # Stage 2: Statistical Outlier Removal
    # --------------------------------------------------------

    statistical, statistical_indices = (
        downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
    )

    statistical_count = len(statistical.points)

    print(
        f"2. Statistical Removal: "
        f"{downsampled_count:,} -> {statistical_count:,}"
    )

    save_point_cloud(
        statistical,
        output_directory / f"{name}_02_statistical.ply",
    )

    # --------------------------------------------------------
    # Stage 3: Radius Outlier Removal
    # --------------------------------------------------------

    radius_cleaned, radius_indices = (
        statistical.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius,
        )
    )

    radius_count = len(radius_cleaned.points)

    print(
        f"3. Radius Removal: "
        f"{statistical_count:,} -> {radius_count:,}"
    )

    save_point_cloud(
        radius_cleaned,
        output_directory / f"{name}_03_final.ply",
    )

    print(
        f"Total points removed: "
        f"{original_count - radius_count:,}"
    )

    retention = (
        100.0 * radius_count / original_count
        if original_count
        else 0.0
    )

    print(f"Points retained: {retention:.2f}%")

    return {
        "name": name,
        "voxel_size": voxel_size,
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
        "nb_points": nb_points,
        "radius": radius,
        "original_count": original_count,
        "downsampled_count": downsampled_count,
        "statistical_count": statistical_count,
        "final_count": radius_count,
        "removed_count": original_count - radius_count,
        "retention_percent": retention,
        "downsampled": downsampled,
        "statistical": statistical,
        "final": radius_cleaned,
    }


# ============================================================
# 7. EXPERIMENT CONFIGURATIONS
# ============================================================

EXPERIMENTS = [
    {
        "name": "cleaned_v1",
        "voxel_size": 0.01,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "nb_points": 12,
        "radius": 0.04,
        "output_directory": CLEANED_V1_DIR,
    },
    {
        "name": "cleaned_v2",
        "voxel_size": 0.02,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "nb_points": 16,
        "radius": 0.05,
        "output_directory": CLEANED_V2_DIR,
    },
    {
        "name": "cleaned_v3",
        "voxel_size": 0.03,
        "nb_neighbors": 30,
        "std_ratio": 1.0,
        "nb_points": 20,
        "radius": 0.07,
        "output_directory": CLEANED_V2_DIR,
    },
]


# ============================================================
# 8. REPORTING
# ============================================================

def save_report(results, output_path):
    """Save the experiment parameters and point counts as CSV."""

    fieldnames = [
        "name",
        "voxel_size",
        "nb_neighbors",
        "std_ratio",
        "nb_points",
        "radius",
        "original_count",
        "downsampled_count",
        "statistical_count",
        "final_count",
        "removed_count",
        "retention_percent",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result[field]
                for field in fieldnames
            })

    print(f"\nReport saved: {output_path}")


def choose_best_result(results):
    """
    Choose a conservative candidate using point retention.

    This is a heuristic, not a ground-truth quality measurement.
    Prefer the least aggressive pipeline that removes enough
    obvious noise. Confirm the choice visually.
    """

    # The result with the most retained points is not automatically
    # the best. Here we use the second-most aggressive result as a
    # starting candidate and still inspect the visual comparison.
    ranked = sorted(
        results,
        key=lambda result: result["retention_percent"],
        reverse=True,
    )

    return ranked[1] if len(ranked) >= 2 else ranked[0]


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("WEEK 3 — DAY 4: ADVANCED CLEANING")
    print("=" * 65)

    # Load source cloud.
    point_cloud, source_name = load_point_cloud()

    print(f"Source cloud: {source_name}")
    print(f"Original points: {len(point_cloud.points):,}")

    save_point_cloud(
        point_cloud,
        RESULTS_DIR / "original.ply",
    )

    # Add noise for a more challenging experiment.
    if ADD_ARTIFICIAL_NOISE:
        noisy_cloud = add_artificial_noise(point_cloud)

        save_point_cloud(
            noisy_cloud,
            RESULTS_DIR / "noisy_input.ply",
        )
    else:
        noisy_cloud = point_cloud

    print(f"Input points for experiments: {len(noisy_cloud.points):,}")

    # Save a visual comparison of original and noisy input.
    save_comparison_plot(
        [point_cloud, noisy_cloud],
        ["Original", "Input with noise"],
        COMPARISON_DIR / "original_vs_noisy.png",
    )

    # Run all experiments on the same input.
    results = []

    for experiment in EXPERIMENTS:
        result = run_cleaning_pipeline(
            source_cloud=noisy_cloud,
            **experiment,
        )

        results.append(result)

    # Compare final outputs.
    comparison_clouds = [noisy_cloud] + [
        result["final"] for result in results
    ]

    comparison_titles = ["Noisy input"] + [
        result["name"] for result in results
    ]

    save_comparison_plot(
        comparison_clouds,
        comparison_titles,
        COMPARISON_DIR / "all_pipelines_comparison.png",
    )

    # Compare intermediate stages of the second pipeline.
    reference = results[1]

    save_comparison_plot(
        [
            noisy_cloud,
            reference["downsampled"],
            reference["statistical"],
            reference["final"],
        ],
        [
            "Noisy input",
            "Voxel Downsampling",
            "Statistical Removal",
            "Radius Removal",
        ],
        COMPARISON_DIR / "cleaning_stages.png",
    )

    # Save metrics.
    report_path = COMPARISON_DIR / "cleaning_report.csv"

    save_report(results, report_path)

    # Choose a candidate and save a separate copy.
    best = choose_best_result(results)

    best_path = RESULTS_DIR / "best_cleaned_point_cloud.ply"

    save_point_cloud(best["final"], best_path)

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)

    for result in results:
        print(
            f"{result['name']}: "
            f"{result['final_count']:,} points remaining, "
            f"{result['retention_percent']:.2f}% retained"
        )

    print("\nCandidate configuration:")
    print(f"Name: {best['name']}")
    print(f"Voxel size: {best['voxel_size']}")
    print(f"Statistical neighbors: {best['nb_neighbors']}")
    print(f"Statistical std_ratio: {best['std_ratio']}")
    print(f"Radius neighbors: {best['nb_points']}")
    print(f"Radius: {best['radius']}")

    print(
        "\nIMPORTANT: This candidate is selected using a simple "
        "retention heuristic."
    )
    print(
        "Inspect the saved comparison images before declaring "
        "the final configuration the best."
    )

    print("\nWEEK 3 — DAY 4 COMPLETED")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise