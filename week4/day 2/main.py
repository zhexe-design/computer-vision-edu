"""
Week 4 - Day 2
Custom ICP Implementation using NumPy and SciPy.

Features:
1. Best-fit rigid transformation using SVD.
2. Custom point-to-point ICP algorithm.
3. Synthetic transformation recovery test.
4. Comparison with Open3D ICP.
5. RMSE convergence visualization.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

MAX_ITERATIONS = 50
TOLERANCE = 1e-6


# ============================================================
# 1. TRANSFORMATION UTILITIES
# ============================================================

def create_rotation_matrix(angle_degrees, axis="z"):
    """
    Create a 3D rotation matrix.

    Parameters
    ----------
    angle_degrees : float
        Rotation angle in degrees.
    axis : str
        Rotation axis: x, y or z.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """

    angle = np.radians(angle_degrees)

    c = np.cos(angle)
    s = np.sin(angle)

    if axis == "x":
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    elif axis == "y":
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    elif axis == "z":
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    raise ValueError("Axis must be x, y or z.")


def apply_transformation(points, R, t):
    """
    Apply rigid transformation to a point cloud.

    Convention:
        points_new = points @ R.T + t
    """

    return points @ R.T + t


# ============================================================
# 2. BEST FIT TRANSFORMATION USING SVD
# ============================================================

def best_fit_transform(A, B):
    """
    Calculate optimal rigid transformation between
    two corresponding point sets using SVD.

    Finds R and t such that:

        B = A @ R.T + t

    Parameters
    ----------
    A : np.ndarray
        Source points (N, 3).
    B : np.ndarray
        Target points (N, 3).

    Returns
    -------
    R : np.ndarray
        Rotation matrix (3, 3).
    t : np.ndarray
        Translation vector (3,).
    """

    assert A.shape == B.shape
    assert A.shape[1] == 3

    # Step 1: Calculate centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Step 2: Center the point clouds
    AA = A - centroid_A
    BB = B - centroid_B

    # Step 3: Covariance matrix
    H = AA.T @ BB

    # Step 4: Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Calculate rotation
    R = Vt.T @ U.T

    # Step 6: Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 7: Calculate translation
    t = centroid_B - R @ centroid_A

    return R, t


# ============================================================
# 3. SYNTHETIC TEST
# ============================================================

def test_best_fit_transform():
    """
    Test transformation recovery using 500 random points.

    Expected:
        Rotation error < 0.01 degrees.
    """

    print("\n" + "=" * 60)
    print("SYNTHETIC TRANSFORMATION TEST")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Generate random 3D points
    source = np.random.randn(500, 3)

    # Ground truth transformation
    true_angle = 15.0

    true_R = create_rotation_matrix(
        true_angle,
        axis="z"
    )

    true_t = np.array([0.5, -0.3, 0.8])

    # Transform source points
    target = apply_transformation(
        source,
        true_R,
        true_t
    )

    # Recover transformation
    estimated_R, estimated_t = best_fit_transform(
        source,
        target
    )

    # Rotation error
    relative_R = estimated_R @ true_R.T

    trace = np.clip(
        (np.trace(relative_R) - 1) / 2,
        -1.0,
        1.0
    )

    angle_error = np.degrees(
        np.arccos(trace)
    )

    # Translation error
    translation_error = np.linalg.norm(
        estimated_t - true_t
    )

    print(f"Points: {len(source)}")

    print(f"True rotation: {true_angle:.6f} degrees")

    print(
        f"Rotation error: {angle_error:.10f} degrees"
    )

    print(
        f"Translation error: {translation_error:.10f}"
    )

    print("\nEstimated rotation matrix:")
    print(estimated_R)

    print("\nEstimated translation:")
    print(estimated_t)

    passed = angle_error < 0.01

    print(
        "\nTEST:",
        "PASSED" if passed else "FAILED"
    )

    # Save results
    with open(
        RESULTS_DIR / "synthetic_test.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("SYNTHETIC TRANSFORMATION TEST\n")
        f.write("=" * 40 + "\n")

        f.write(f"Points: {len(source)}\n")
        f.write(f"True angle: {true_angle}\n")
        f.write(f"Rotation error: {angle_error}\n")
        f.write(f"Translation error: {translation_error}\n")
        f.write(f"Test passed: {passed}\n")

    return passed


# ============================================================
# 4. CUSTOM ICP ALGORITHM
# ============================================================

def icp(
    source,
    target,
    max_iter=50,
    tol=1e-6,
    max_correspondence_distance=np.inf
):
    """
    Iterative Closest Point algorithm.

    Parameters
    ----------
    source : np.ndarray
        Source point cloud (N, 3).
    target : np.ndarray
        Target point cloud (M, 3).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    max_correspondence_distance : float
        Maximum allowed nearest-neighbor distance.

    Returns
    -------
    T : np.ndarray
        Final 4x4 transformation matrix.
    aligned : np.ndarray
        Transformed source point cloud.
    rmse_history : list
        RMSE at each iteration.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Source must have shape (N, 3).")

    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Target must have shape (M, 3).")

    if len(source) < 3 or len(target) < 3:
        raise ValueError("Point clouds must contain at least 3 points.")

    # Initial alignment
    aligned = source.copy()

    # Total transformation
    T = np.eye(4)

    # KD-tree is built ONCE because target is static
    tree = cKDTree(target)

    rmse_history = []

    print("\nStarting custom ICP...")

    for iteration in range(max_iter):

        # Step 1: Find nearest neighbors
        distances, indices = tree.query(
            aligned,
            k=1
        )

        # Step 2: Filter correspondences
        mask = distances < max_correspondence_distance

        if np.count_nonzero(mask) < 3:
            raise RuntimeError(
                "Not enough valid correspondences."
            )

        matched_source = aligned[mask]
        matched_target = target[indices[mask]]

        # Step 3: Estimate incremental transformation
        R, t = best_fit_transform(
            matched_source,
            matched_target
        )

        # Step 4: Apply transformation
        aligned = apply_transformation(
            aligned,
            R,
            t
        )

        # Step 5: Update total transformation
        delta_T = np.eye(4)

        delta_T[:3, :3] = R
        delta_T[:3, 3] = t

        T = delta_T @ T

        # Step 6: Calculate RMSE after alignment
        new_distances, _ = tree.query(
            aligned,
            k=1
        )

        new_distances = new_distances[
            new_distances < max_correspondence_distance
        ]

        if len(new_distances) < 3:
            raise RuntimeError(
                "Not enough valid correspondences after alignment."
            )

        rmse = np.sqrt(
            np.mean(new_distances ** 2)
        )

        rmse_history.append(rmse)

        print(
            f"Iteration {iteration + 1:02d} | "
            f"RMSE: {rmse:.8f}"
        )

        # Step 7: Check convergence
        if len(rmse_history) > 1:

            improvement = abs(
                rmse_history[-2] - rmse_history[-1]
            )

            if improvement < tol:

                print(
                    f"\nConverged after {iteration + 1} iterations."
                )

                break

    print("\nICP finished.")

    return T, aligned, rmse_history


# ============================================================
# 5. RMSE VISUALIZATION
# ============================================================

def plot_rmse(rmse_history):

    plt.figure(figsize=(10, 6))

    iterations = np.arange(
        1,
        len(rmse_history) + 1
    )

    plt.plot(
        iterations,
        rmse_history,
        marker="o",
        linewidth=2
    )

    plt.title("Custom ICP - RMSE Convergence")

    plt.xlabel("Iteration")

    plt.ylabel("RMSE")

    plt.grid(True, alpha=0.3)

    plt.xticks(iterations)

    plt.tight_layout()

    output_path = RESULTS_DIR / "rmse_iterations.png"

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"\nRMSE graph saved: {output_path}")


# ============================================================
# 6. OPEN3D COMPARISON
# ============================================================

def compare_with_open3d(
    source,
    target,
    custom_T,
    custom_rmse,
    custom_time,
    max_correspondence_distance=0.5
):
    """
    Compare custom ICP with Open3D ICP.

    Both algorithms start from the same initial alignment.
    """

    try:
        import open3d as o3d

    except ImportError:

        print(
            "\nOpen3D is not installed. "
            "Skipping comparison."
        )

        return

    print("\n" + "=" * 60)
    print("OPEN3D COMPARISON")
    print("=" * 60)

    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()

    source_cloud.points = o3d.utility.Vector3dVector(
        source
    )

    target_cloud.points = o3d.utility.Vector3dVector(
        target
    )

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=MAX_ITERATIONS
    )

    start = time.perf_counter()

    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )

    open3d_time = time.perf_counter() - start

    open3d_T = result.transformation

    print("\nCUSTOM ICP:")
    print(f"RMSE: {custom_rmse[-1]:.8f}")
    print(f"Iterations: {len(custom_rmse)}")
    print(f"Time: {custom_time:.6f} seconds")

    print("\nOPEN3D ICP:")
    print(f"RMSE: {result.inlier_rmse:.8f}")
    print(f"Fitness: {result.fitness:.6f}")
    print(f"Time: {open3d_time:.6f} seconds")

    print("\nCUSTOM TRANSFORMATION:")
    print(custom_T)

    print("\nOPEN3D TRANSFORMATION:")
    print(open3d_T)

    # Save comparison
    with open(
        RESULTS_DIR / "comparison.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("CUSTOM ICP VS OPEN3D\n")
        f.write("=" * 50 + "\n\n")

        f.write("CUSTOM ICP\n")
        f.write(f"RMSE: {custom_rmse[-1]:.8f}\n")
        f.write(f"Iterations: {len(custom_rmse)}\n")
        f.write(f"Time: {custom_time:.6f}\n\n")

        f.write("OPEN3D ICP\n")
        f.write(f"RMSE: {result.inlier_rmse:.8f}\n")
        f.write(f"Fitness: {result.fitness:.6f}\n")
        f.write(f"Time: {open3d_time:.6f}\n\n")

        f.write("CUSTOM TRANSFORMATION:\n")
        f.write(str(custom_T) + "\n\n")

        f.write("OPEN3D TRANSFORMATION:\n")
        f.write(str(open3d_T))

    np.save(
        RESULTS_DIR / "open3d_transformation.npy",
        open3d_T
    )

    print("\nComparison saved.")


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 60)
    print("WEEK 4 - DAY 2: CUSTOM ICP")
    print("=" * 60)

    # Test mathematical implementation
    test_passed = test_best_fit_transform()

    if not test_passed:
        raise RuntimeError(
            "SVD transformation test failed."
        )

    # Generate synthetic point cloud
    np.random.seed(RANDOM_SEED)

    source = np.random.randn(1000, 3)

    # Ground truth transformation
    true_R = create_rotation_matrix(
        3.0,
        axis="z"
    )

    true_t = np.array([
        0.02,
        -0.01,
        0.015
    ])

    target = apply_transformation(
        source,
        true_R,
        true_t
    )

    print("\nSynthetic point clouds generated.")

    print(f"Source points: {len(source)}")
    print(f"Target points: {len(target)}")

    # Run custom ICP
    start = time.perf_counter()

    T, aligned, rmse_history = icp(
        source,
        target,
        max_iter=MAX_ITERATIONS,
        tol=TOLERANCE,
        max_correspondence_distance=0.5
    )

    custom_time = time.perf_counter() - start

    # Save transformation matrix
    np.save(
        RESULTS_DIR / "transformation_matrix.npy",
        T
    )

    # Save RMSE graph
    plot_rmse(rmse_history)

    # Compare with Open3D
    compare_with_open3d(
        source,
        target,
        T,
        rmse_history,
        custom_time,
        max_correspondence_distance=0.5
    )

    # Final report
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Iterations: {len(rmse_history)}")
    print(f"Final RMSE: {rmse_history[-1]:.8f}")
    print(f"Execution time: {custom_time:.6f} seconds")

    print("\nEstimated transformation:")
    print(T)

    print("\nAll results saved to:")
    print(RESULTS_DIR)

    print("\nDAY 2 COMPLETED.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()