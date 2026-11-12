"""
Week 4 - Day 3: Global Registration and Experiments
===================================================

Pipeline:
1. Load two point clouds from week4/data (or use Stanford Bunny fallback).
2. Downsample, estimate normals and compute FPFH features.
3. Global registration with feature matching + RANSAC.
4. Refine with point-to-point and point-to-plane ICP.
5. Run three experiments:
   - Initial rotation robustness (0..90 degrees), with/without global registration.
   - RMSE convergence: custom ICP, Open3D point-to-point, Open3D point-to-plane.
   - Voxel-size speed/accuracy trade-off (0.02, 0.05, 0.1).
6. Save CSV reports, plots and aligned point clouds.

Run from the repository root:
    python week4/day3_global_registration.py

Recommended: Python 3.11/3.12 with a compatible Open3D build.
"""

from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

WEEK_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEEK_DIR.parent
DATA_DIR = WEEK_DIR / "data"
RESULTS_DIR = WEEK_DIR / "results" / "day3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VOXEL_SIZE = 0.05
VOXEL_SIZES = (0.02, 0.05, 0.10)
ROTATION_ANGLES = list(range(0, 91, 10))
ICP_MAX_ITERATIONS = 50
RANSAC_MAX_ITERATIONS = 100000
RANSAC_CONFIDENCE = 0.999

# A successful registration must have low RMSE and a reasonable overlap.
SUCCESS_RMSE_FACTOR = 0.5
MIN_FITNESS = 0.30

SUPPORTED_EXTENSIONS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    try:
        o3d.utility.random.seed(seed)
    except (AttributeError, TypeError):
        # Some older Open3D builds do not expose this function.
        pass


def find_point_cloud_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

def load_demo_point_cloud():
    """
    Load Stanford Bunny using different Open3D dataset APIs.
    Fall back to a procedurally generated asymmetric 3D model.
    """

    # Option 1: Stanford Bunny mesh
    if hasattr(o3d.data, "BunnyMesh"):
        try:
            print("[INFO] Loading Stanford Bunny mesh...")

            bunny = o3d.data.BunnyMesh()
            mesh = o3d.io.read_triangle_mesh(bunny.path)

            if not mesh.is_empty():
                cloud = mesh.sample_points_poisson_disk(
                    number_of_points=20000
                )

                print("[INFO] Stanford Bunny loaded successfully.")
                return cloud

        except Exception as e:
            print(f"[WARNING] BunnyMesh unavailable: {e}")

    # Option 2: PLY point cloud
    if hasattr(o3d.data, "PLYPointCloud"):
        try:
            print("[INFO] Trying PLYPointCloud...")

            dataset = o3d.data.PLYPointCloud()
            cloud = o3d.io.read_point_cloud(dataset.path)

            if not cloud.is_empty():
                print("[INFO] PLY point cloud loaded.")
                return cloud

        except Exception as e:
            print(f"[WARNING] PLYPointCloud unavailable: {e}")

    # Option 3: Synthetic asymmetric 3D model
    print("[INFO] Creating synthetic 3D point cloud...")

    mesh = o3d.geometry.TriangleMesh()

    # Main body
    body = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.5,
        resolution=30
    )

    body.scale(1.0, center=(0, 0, 0))
    body.translate((0, 0, 0))

    # Additional asymmetric components
    head = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.25,
        resolution=20
    )
    head.translate((0.55, 0.1, 0.15))

    leg = o3d.geometry.TriangleMesh.create_box(
        width=0.3,
        height=0.25,
        depth=0.7
    )
    leg.translate((-0.4, -0.3, -0.6))

    detail = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.15,
        resolution=15
    )
    detail.translate((-0.4, 0.45, 0.3))

    mesh += body
    mesh += head
    mesh += leg
    mesh += detail

    mesh.compute_vertex_normals()

    cloud = mesh.sample_points_poisson_disk(
        number_of_points=20000
    )

    print(
        f"[INFO] Synthetic point cloud generated: "
        f"{len(cloud.points)} points"
    )

    return cloud

def load_point_clouds() -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, str]:
    """
    Prefer two local point clouds in week4/data.
    If only one local cloud exists, create a reproducible transformed copy.
    If no local cloud exists, use Open3D's Stanford Bunny and create a pair.
    """
    files = find_point_cloud_files()

    if len(files) >= 2:
        source = o3d.io.read_point_cloud(str(files[0]))
        target = o3d.io.read_point_cloud(str(files[1]))
        description = f"Local pair: {files[0].name} -> {files[1].name}"
    elif len(files) == 1:
        source = o3d.io.read_point_cloud(str(files[0]))
        target = copy.deepcopy(source)
        description = f"Local cloud with synthetic target: {files[0].name}"
        target.transform(make_transform(7.0, np.array([0.025, -0.015, 0.010])))
    else:
        print("[INFO] No local point clouds found.")
        print("[INFO] Loading demonstration dataset...")

        source = load_demo_point_cloud()

        target = copy.deepcopy(source)

        description = "Demo point cloud with synthetic transformation"

        target.transform(
            make_transform(
                7.0,
                np.array([0.025, -0.015, 0.010])
            )
        )

    if source.is_empty() or target.is_empty():
        raise RuntimeError("Could not load valid source and target point clouds.")

    # Remove non-finite points, if present. 
    source = source.select_by_index(
        np.flatnonzero(np.isfinite(np.asarray(source.points)).all(axis=1)).tolist()
    )
    target = target.select_by_index(
        np.flatnonzero(np.isfinite(np.asarray(target.points)).all(axis=1)).tolist()
    )

    if len(source.points) < 50 or len(target.points) < 50:
        raise RuntimeError("Point clouds are too small for FPFH/RANSAC registration.")

    print(f"[DATA] {description}")
    print(f"[DATA] Source points: {len(source.points):,}")
    print(f"[DATA] Target points: {len(target.points):,}")
    return source, target, description


def make_transform(angle_degrees: float, translation: np.ndarray) -> np.ndarray:
    """Rigid transform rotating around the Z axis and then translating."""
    a = np.deg2rad(angle_degrees)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def rotate_cloud_about_centroid(
    cloud: o3d.geometry.PointCloud, angle_degrees: float
) -> o3d.geometry.PointCloud:
    """Create a rotated copy while keeping the cloud centroid fixed."""
    result = copy.deepcopy(cloud)
    center = result.get_center()
    R = make_transform(angle_degrees, np.zeros(3))[:3, :3]
    result.rotate(R, center=center)
    return result


def preprocess_point_cloud(
    cloud: o3d.geometry.PointCloud, voxel_size: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Voxel downsample, estimate normals and compute FPFH descriptors."""
    down = cloud.voxel_down_sample(voxel_size)

    normal_radius = voxel_size * 2.0
    feature_radius = voxel_size * 5.0

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )

    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius, max_nn=100
        )
    )
    return down, feature


def global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    Coarse registration using FPFH feature correspondences and RANSAC.
    RANSAC correspondence threshold follows the Open3D tutorial heuristic:
    voxel_size * 1.5.
    """
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    distance_threshold = voxel_size * 1.5
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold
        ),
    ]

    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        RANSAC_MAX_ITERATIONS, RANSAC_CONFIDENCE
    )

    start = time.perf_counter()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,  # mutual_filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        checkers,
        criteria,
    )
    elapsed = time.perf_counter() - start

    return result.transformation, float(result.fitness), float(result.inlier_rmse), elapsed


def refine_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    voxel_size: float,
    method: str = "point_to_plane",
    max_iterations: int = ICP_MAX_ITERATIONS,
):
    """Run one Open3D ICP refinement."""
    threshold = voxel_size * 0.4
    source_work = copy.deepcopy(source)
    target_work = copy.deepcopy(target)

    if method == "point_to_plane":
        radius = voxel_size * 2.0
        for cloud in (source_work, target_work):
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
    elif method == "point_to_point":
        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
    else:
        raise ValueError("method must be 'point_to_point' or 'point_to_plane'")

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations
    )

    start = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        source_work,
        target_work,
        threshold,
        initial_transform,
        estimation,
        criteria,
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def iterative_open3d_rmse(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    voxel_size: float,
    method: str,
    iterations: int = 30,
) -> list[float]:
    """
    Record an RMSE curve by running Open3D ICP one iteration at a time.
    This is slower than a single ICP call but exposes per-iteration metrics.
    """
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    threshold = voxel_size * 0.4

    if method == "point_to_plane":
        radius = voxel_size * 2.0
        for cloud in (src, tgt):
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    current = np.asarray(initial_transform, dtype=float).copy()
    history = []

    for _ in range(iterations):
        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            threshold,
            current,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
        )
        current = result.transformation
        history.append(float(result.inlier_rmse))
        if len(history) > 1 and abs(history[-2] - history[-1]) < 1e-8:
            break

    return history


def load_custom_icp():
    """Import Day 2's own NumPy/SciPy ICP implementation if available."""
    src_dir = WEEK_DIR / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    try:
        from my_icp import icp as custom_icp
        return custom_icp
    except Exception as exc:
        print(f"[WARNING] Could not import src/my_icp.py: {exc}")
        print("[WARNING] The custom ICP curve will be omitted.")
        return None


def run_angle_experiment(
    reference: o3d.geometry.PointCloud,
    voxel_size: float,
) -> list[dict]:
    """
    Compare local ICP from identity against global RANSAC + ICP
    for initial Z rotations from 0 to 90 degrees.
    """
    print("\n[EXPERIMENT 1] Initial rotation robustness")
    rows = []
    threshold = voxel_size * 0.4
    success_rmse = voxel_size * SUCCESS_RMSE_FACTOR

    for angle in ROTATION_ANGLES:
        source_rotated = rotate_cloud_about_centroid(reference, angle)

        # Local ICP: no global initialization.
        local_result, local_time = refine_icp(
            source_rotated, reference, np.eye(4), voxel_size,
            method="point_to_point"
        )
        local_success = (
            local_result.inlier_rmse < success_rmse
            and local_result.fitness >= MIN_FITNESS
        )

        # Global registration, followed by point-to-plane ICP.
        try:
            global_T, global_fitness, global_rmse, ransac_time = global_registration(
                source_rotated, reference, voxel_size
            )
            global_result, refine_time = refine_icp(
                source_rotated, reference, global_T, voxel_size,
                method="point_to_plane"
            )
            global_success = (
                global_result.inlier_rmse < success_rmse
                and global_result.fitness >= MIN_FITNESS
            )
            global_final_rmse = float(global_result.inlier_rmse)
            global_final_fitness = float(global_result.fitness)
            total_global_time = ransac_time + refine_time
        except Exception as exc:
            print(f"[WARNING] RANSAC failed at {angle} degrees: {exc}")
            global_success = False
            global_final_rmse = float("inf")
            global_final_fitness = 0.0
            total_global_time = float("nan")

        rows.append({
            "angle_degrees": angle,
            "local_icp_success": int(local_success),
            "local_icp_rmse": float(local_result.inlier_rmse),
            "local_icp_fitness": float(local_result.fitness),
            "local_icp_time_seconds": float(local_time),
            "global_plus_icp_success": int(global_success),
            "global_plus_icp_rmse": global_final_rmse,
            "global_plus_icp_fitness": global_final_fitness,
            "global_plus_icp_time_seconds": total_global_time,
        })

        print(
            f"Angle {angle:>2}° | local={local_success} "
            f"(RMSE {local_result.inlier_rmse:.5f}) | "
            f"global+ICP={global_success} (RMSE {global_final_rmse:.5f})"
        )

    write_csv(RESULTS_DIR / "rotation_robustness.csv", rows)
    plot_rotation_success(rows)
    return rows


def plot_rotation_success(rows: list[dict]) -> None:
    angles = [r["angle_degrees"] for r in rows]
    local = [r["local_icp_success"] for r in rows]
    global_ = [r["global_plus_icp_success"] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(angles, local, marker="o", linewidth=2, label="ICP from identity")
    plt.plot(angles, global_, marker="s", linewidth=2, label="RANSAC + ICP")
    plt.yticks([0, 1], ["Failure", "Success"])
    plt.ylim(-0.15, 1.15)
    plt.xlabel("Initial rotation around Z (degrees)")
    plt.ylabel("Registration result")
    plt.title("Registration robustness to initial rotation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = RESULTS_DIR / "rotation_success.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"[SAVE] {path}")


def run_rmse_experiment(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
) -> None:
    print("\n[EXPERIMENT 2] Point-to-point vs point-to-plane")
    global_T, _, _, _ = global_registration(source, target, voxel_size)

    p2p_history = iterative_open3d_rmse(
        source, target, global_T, voxel_size, "point_to_point", 30
    )
    p2plane_history = iterative_open3d_rmse(
        source, target, global_T, voxel_size, "point_to_plane", 30
    )

    custom_history = None
    custom_icp = load_custom_icp()
    if custom_icp is not None:
        # Day 2's custom ICP starts at identity. Pre-align source with RANSAC,
        # then measure its incremental point-to-point refinement.
        source_pre_aligned = np.asarray(source.points) @ global_T[:3, :3].T + global_T[:3, 3]
        target_points = np.asarray(target.points)
        _, _, custom_history = custom_icp(
            source_pre_aligned,
            target_points,
            max_iter=30,
            tol=1e-8,
            max_correspondence_distance=voxel_size * 0.4,
        )

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(p2p_history) + 1), p2p_history,
             marker="o", label="Open3D point-to-point")
    plt.plot(range(1, len(p2plane_history) + 1), p2plane_history,
             marker="s", label="Open3D point-to-plane")
    if custom_history is not None:
        plt.plot(range(1, len(custom_history) + 1), custom_history,
                 marker="^", label="Custom NumPy ICP (point-to-point)")
    plt.xlabel("Iteration")
    plt.ylabel("Inlier RMSE")
    plt.title("ICP convergence: point-to-point vs point-to-plane")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = RESULTS_DIR / "icp_rmse_comparison.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"[SAVE] {path}")

    rows = []
    for name, history in [
        ("Open3D point-to-point", p2p_history),
        ("Open3D point-to-plane", p2plane_history),
        ("Custom NumPy ICP", custom_history or []),
    ]:
        if history:
            rows.append({
                "method": name,
                "iterations_recorded": len(history),
                "initial_rmse": float(history[0]),
                "final_rmse": float(history[-1]),
            })
    write_csv(RESULTS_DIR / "icp_method_comparison.csv", rows)


def run_voxel_experiment(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> list[dict]:
    print("\n[EXPERIMENT 3] Voxel size: speed vs accuracy")
    rows = []

    for voxel in VOXEL_SIZES:
        start = time.perf_counter()
        try:
            global_T, global_fitness, global_rmse, ransac_time = global_registration(
                source, target, voxel
            )
            result, icp_time = refine_icp(
                source, target, global_T, voxel, method="point_to_plane"
            )
            total_time = time.perf_counter() - start

            row = {
                "voxel_size": voxel,
                "source_downsampled_points": len(source.voxel_down_sample(voxel).points),
                "target_downsampled_points": len(target.voxel_down_sample(voxel).points),
                "ransac_fitness": global_fitness,
                "ransac_rmse": global_rmse,
                "final_fitness": float(result.fitness),
                "final_rmse": float(result.inlier_rmse),
                "ransac_time_seconds": ransac_time,
                "icp_time_seconds": icp_time,
                "total_time_seconds": total_time,
            }
        except Exception as exc:
            print(f"[WARNING] Voxel {voxel} failed: {exc}")
            row = {
                "voxel_size": voxel,
                "source_downsampled_points": len(source.voxel_down_sample(voxel).points),
                "target_downsampled_points": len(target.voxel_down_sample(voxel).points),
                "ransac_fitness": 0.0,
                "ransac_rmse": float("nan"),
                "final_fitness": 0.0,
                "final_rmse": float("nan"),
                "ransac_time_seconds": float("nan"),
                "icp_time_seconds": float("nan"),
                "total_time_seconds": float("nan"),
            }

        rows.append(row)
        print(
            f"Voxel {voxel:.2f} | points={row['source_downsampled_points']} | "
            f"RMSE={row['final_rmse']:.6f} | "
            f"time={row['total_time_seconds']:.3f}s"
        )

    write_csv(RESULTS_DIR / "voxel_experiment.csv", rows)
    plot_voxel_results(rows)
    return rows


def plot_voxel_results(rows: list[dict]) -> None:
    valid = [r for r in rows if np.isfinite(r["final_rmse"])]
    if not valid:
        print("[WARNING] No valid voxel experiment results to plot.")
        return

    voxels = [r["voxel_size"] for r in valid]
    times = [r["total_time_seconds"] for r in valid]
    rmses = [r["final_rmse"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(voxels, times, marker="o", linewidth=2, label="Runtime (s)")
    ax1.set_xlabel("Voxel size")
    ax1.set_ylabel("Total runtime (seconds)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(voxels, rmses, marker="s", linewidth=2, label="Final RMSE")
    ax2.set_ylabel("Final inlier RMSE")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Voxel size: runtime vs registration accuracy")
    fig.tight_layout()
    path = RESULTS_DIR / "voxel_speed_accuracy.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"[SAVE] {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {path}")


def main() -> None:
    set_random_seed()
    print("=" * 72)
    print("WEEK 4 - DAY 3: GLOBAL REGISTRATION AND EXPERIMENTS")
    print("=" * 72)

    source, target, description = load_point_clouds()
    print(f"\nDataset: {description}")
    print(f"Base voxel size: {VOXEL_SIZE}")

    # Full coarse-to-fine pipeline on the selected pair.
    print("\n[PIPELINE] FPFH + RANSAC + point-to-plane ICP")
    global_T, fitness, ransac_rmse, ransac_time = global_registration(
        source, target, VOXEL_SIZE
    )
    refined, icp_time = refine_icp(
        source, target, global_T, VOXEL_SIZE, method="point_to_plane"
    )

    print(f"RANSAC fitness: {fitness:.6f}")
    print(f"RANSAC inlier RMSE: {ransac_rmse:.6f}")
    print(f"Refined fitness: {refined.fitness:.6f}")
    print(f"Refined inlier RMSE: {refined.inlier_rmse:.6f}")
    print(f"RANSAC time: {ransac_time:.3f}s")
    print(f"ICP time: {icp_time:.3f}s")

    aligned = copy.deepcopy(source)
    aligned.transform(refined.transformation)
    o3d.io.write_point_cloud(str(RESULTS_DIR / "aligned_global_icp.ply"), aligned)
    np.save(RESULTS_DIR / "global_transformation.npy", global_T)
    np.save(RESULTS_DIR / "refined_transformation.npy", refined.transformation)

    # Experiment suite.
    run_angle_experiment(source, VOXEL_SIZE)
    run_rmse_experiment(source, target, VOXEL_SIZE)
    run_voxel_experiment(source, target)

    print("\n" + "=" * 72)
    print("DAY 3 REGISTRATION EXPERIMENTS COMPLETED")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
