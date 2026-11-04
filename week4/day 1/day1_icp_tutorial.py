
"""
Week 4 — Day 1
ICP Registration with Open3D

Experiments:
1. Load and visualize source and target point clouds.
2. Evaluate initial registration.
3. Point-to-Point ICP.
4. Point-to-Plane ICP.
5. Compare four distance thresholds.
6. Save metrics, plots and registered point clouds.
"""

from pathlib import Path
import json
import csv

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
ICP_RESULTS = RESULTS / "icp"
METRICS_DIR = RESULTS / "metrics"
PLOTS_DIR = RESULTS / "plots"

for folder in [ICP_RESULTS, METRICS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.01, 0.02, 0.05, 0.1]

MAX_ITERATIONS = 200

# Initial transformation from the Open3D ICP tutorial.
INITIAL_TRANSFORMATION = np.array([
    [0.862, 0.011, -0.507, 0.5],
    [-0.139, 0.967, -0.215, 0.7],
    [0.487, 0.255, 0.835, -1.4],
    [0.0, 0.0, 0.0, 1.0]
])


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_point_clouds():
    print("\n[1] Loading Open3D demonstration dataset...")

    dataset = o3d.data.DemoICPPointClouds()

    source = o3d.io.read_point_cloud(dataset.paths[0])
    target = o3d.io.read_point_cloud(dataset.paths[1])

    print(f"Source points: {len(source.points)}")
    print(f"Target points: {len(target.points)}")

    if source.is_empty() or target.is_empty():
        raise RuntimeError("Could not load demonstration point clouds.")

    return source, target


# ============================================================
# 2. VISUALIZATION
# ============================================================

def visualize_clouds(source, target, title="Point Clouds"):
    source_vis = o3d.geometry.PointCloud(source)
    target_vis = o3d.geometry.PointCloud(target)

    source_vis.paint_uniform_color([1.0, 0.0, 0.0])
    target_vis.paint_uniform_color([0.0, 0.7, 1.0])

    print(f"\nVisualizing: {title}")

    o3d.visualization.draw_geometries(
        [source_vis, target_vis],
        window_name=title
    )


# ============================================================
# 3. PREPARE NORMALS
# ============================================================

def estimate_normals(source, target):
    print("\n[2] Estimating surface normals...")

    source_copy = o3d.geometry.PointCloud(source)
    target_copy = o3d.geometry.PointCloud(target)

    search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,
        max_nn=30
    )

    source_copy.estimate_normals(search_param)
    target_copy.estimate_normals(search_param)

    print("Normals estimated.")

    return source_copy, target_copy


# ============================================================
# 4. EVALUATE INITIAL REGISTRATION
# ============================================================

def evaluate_initial(source, target):
    print("\n[3] Evaluating initial registration...")

    results = []

    for threshold in THRESHOLDS:
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            threshold,
            INITIAL_TRANSFORMATION
        )

        results.append({
            "threshold": threshold,
            "fitness": evaluation.fitness,
            "inlier_rmse": evaluation.inlier_rmse
        })

        print(
            f"Threshold: {threshold:.3f} | "
            f"Fitness: {evaluation.fitness:.5f} | "
            f"RMSE: {evaluation.inlier_rmse:.5f}"
        )

    return results


# ============================================================
# 5. POINT-TO-POINT ICP
# ============================================================

def run_point_to_point(source, target):
    print("\n[4] Running Point-to-Point ICP...")

    results = []

    for threshold in THRESHOLDS:

        registration = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            INITIAL_TRANSFORMATION,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=MAX_ITERATIONS
            )
        )

        result = {
            "method": "Point-to-Point",
            "threshold": threshold,
            "fitness": registration.fitness,
            "inlier_rmse": registration.inlier_rmse,
            "transformation": registration.transformation.tolist()
        }

        results.append(result)

        print(
            f"Threshold: {threshold:.3f} | "
            f"Fitness: {registration.fitness:.5f} | "
            f"RMSE: {registration.inlier_rmse:.5f}"
        )

        if threshold == 0.05:
            save_registration(
                source,
                target,
                registration,
                "point_to_point"
            )

    return results


# ============================================================
# 6. POINT-TO-PLANE ICP
# ============================================================

def run_point_to_plane(source, target):
    print("\n[5] Running Point-to-Plane ICP...")

    results = []

    for threshold in THRESHOLDS:

        registration = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            INITIAL_TRANSFORMATION,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=MAX_ITERATIONS
            )
        )

        result = {
            "method": "Point-to-Plane",
            "threshold": threshold,
            "fitness": registration.fitness,
            "inlier_rmse": registration.inlier_rmse,
            "transformation": registration.transformation.tolist()
        }

        results.append(result)

        print(
            f"Threshold: {threshold:.3f} | "
            f"Fitness: {registration.fitness:.5f} | "
            f"RMSE: {registration.inlier_rmse:.5f}"
        )

        if threshold == 0.05:
            save_registration(
                source,
                target,
                registration,
                "point_to_plane"
            )

    return results


# ============================================================
# 7. SAVE REGISTERED POINT CLOUDS
# ============================================================

def save_registration(source, target, registration, name):
    aligned = o3d.geometry.PointCloud(source)

    aligned.transform(registration.transformation)

    aligned.paint_uniform_color([1.0, 0.0, 0.0])

    target_copy = o3d.geometry.PointCloud(target)
    target_copy.paint_uniform_color([0.0, 0.7, 1.0])

    o3d.io.write_point_cloud(
        str(ICP_RESULTS / f"{name}_aligned.ply"),
        aligned
    )

    o3d.io.write_point_cloud(
        str(ICP_RESULTS / f"{name}_target.ply"),
        target_copy
    )

    print(f"Saved: {name}")


# ============================================================
# 8. SAVE METRICS
# ============================================================

def save_metrics(initial_results, p2p_results, p2plane_results):

    all_results = []

    for result in initial_results:
        all_results.append({
            "method": "Initial",
            **result
        })

    all_results.extend(p2p_results)
    all_results.extend(p2plane_results)

    csv_path = METRICS_DIR / "day1_icp_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "threshold",
                "fitness",
                "inlier_rmse"
            ]
        )

        writer.writeheader()

        for result in all_results:
            writer.writerow({
                "method": result["method"],
                "threshold": result["threshold"],
                "fitness": result["fitness"],
                "inlier_rmse": result["inlier_rmse"]
            })

    json_path = METRICS_DIR / "day1_icp_metrics.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print("\nMetrics saved.")


# ============================================================
# 9. EXPERIMENT GRAPHS
# ============================================================

def plot_metrics(initial_results, p2p_results, p2plane_results):

    thresholds = THRESHOLDS

    methods = {
        "Initial": initial_results,
        "Point-to-Point": p2p_results,
        "Point-to-Plane": p2plane_results
    }

    # Fitness plot
    plt.figure(figsize=(9, 5))

    for name, results in methods.items():
        fitness = [r["fitness"] for r in results]

        plt.plot(
            thresholds,
            fitness,
            marker="o",
            label=name
        )

    plt.xlabel("Distance Threshold")
    plt.ylabel("Fitness")
    plt.title("ICP Fitness Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "day1_fitness.png", dpi=200)
    plt.close()

    # RMSE plot
    plt.figure(figsize=(9, 5))

    for name, results in methods.items():
        rmse = [r["inlier_rmse"] for r in results]

        plt.plot(
            thresholds,
            rmse,
            marker="o",
            label=name
        )

    plt.xlabel("Distance Threshold")
    plt.ylabel("Inlier RMSE")
    plt.title("ICP Inlier RMSE Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "day1_rmse.png", dpi=200)
    plt.close()

    print("Experiment graphs saved.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("WEEK 4 — DAY 1: ICP REGISTRATION")
    print("=" * 55)

    source, target = load_point_clouds()

    # Initial visualization
    visualize_clouds(
        source,
        target,
        "Initial Source and Target"
    )

    # Initial registration evaluation
    initial_results = evaluate_initial(source, target)

    # Estimate normals for Point-to-Plane
    source_normals, target_normals = estimate_normals(
        source,
        target
    )

    # ICP experiments
    p2p_results = run_point_to_point(source, target)

    p2plane_results = run_point_to_plane(
        source_normals,
        target_normals
    )

    # Save metrics and graphs
    save_metrics(
        initial_results,
        p2p_results,
        p2plane_results
    )

    plot_metrics(
        initial_results,
        p2p_results,
        p2plane_results
    )

    print("\n" + "=" * 55)
    print("DAY 1 ICP EXPERIMENTS COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
