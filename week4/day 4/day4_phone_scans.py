#!/usr/bin/env python
"""
Week 4 — Day 4: Register two smartphone scans.

Example:
  python day4_phone_scans.py --source data/scan1.obj --target data/scan2.obj
  python day4_phone_scans.py --source data/scan1.glb --target data/scan2.glb --voxel-size 0.02

Dependencies:
  pip install open3d numpy scipy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_as_point_cloud(path: Path, points: int) -> o3d.geometry.PointCloud:
    """Load OBJ/GLB mesh or a point-cloud file and return a point cloud."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".obj", ".glb", ".gltf", ".stl", ".fbx", ".off"}:
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise ValueError(
                f"Could not read a triangle mesh from {path}. "
                "Try exporting it from Scaniverse as OBJ or GLB."
            )
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        return mesh.sample_points_poisson_disk(number_of_points=points)

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Could not read a point cloud from {path}")
    return cloud


def clean_cloud(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(pcd)
    pcd.remove_non_finite_points()
    pcd.remove_duplicated_points()
    if len(pcd.points) < 30:
        raise ValueError("Too few points after loading; check the input scan.")
    # Remove isolated points conservatively before downsampling.
    if len(pcd.points) > 100:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        pcd = pcd.select_by_index(ind)
    pcd = pcd.voxel_down_sample(voxel)
    if len(pcd.points) < 30:
        raise ValueError("Too few points after voxel downsampling. Reduce --voxel-size.")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=30)
    )
    pcd.normalize_normals()
    return pcd


def prepare_fpfh(pcd: o3d.geometry.PointCloud, voxel: float):
    radius_normal = voxel * 2.0
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return feature


def global_registration(source, target, voxel):
    source_fpfh = prepare_fpfh(source, voxel)
    target_fpfh = prepare_fpfh(target, voxel)
    distance = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance,
        estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def custom_icp(source_points, target_points, initial, max_distance, max_iter=60, tol=1e-7):
    """Simple point-to-point ICP using cKDTree and an SVD rigid transform."""
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)
    tree = cKDTree(dst)
    transform = initial.copy()
    history = []
    previous = np.inf

    for _ in range(max_iter):
        moved = (transform[:3, :3] @ src.T).T + transform[:3, 3]
        distances, indices = tree.query(moved, k=1, workers=-1)
        mask = distances < max_distance
        if int(mask.sum()) < 6:
            break

        a, b = moved[mask], dst[indices[mask]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        translation = cb - rotation @ ca

        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = translation
        transform = delta @ transform

        moved_after = (transform[:3, :3] @ src.T).T + transform[:3, 3]
        d_after, _ = tree.query(moved_after, k=1, workers=-1)
        valid = d_after < max_distance
        rmse = float(np.sqrt(np.mean(d_after[valid] ** 2))) if valid.any() else None
        if rmse is not None:
            history.append(rmse)
            if abs(previous - rmse) < tol:
                break
            previous = rmse
    return transform, history


def evaluate(source, target, transform, threshold):
    return o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, transform
    )


def save_screenshot(source, target, output: Path, title: str):
    a = o3d.geometry.PointCloud(source)
    b = o3d.geometry.PointCloud(target)
    a.paint_uniform_color([0.95, 0.35, 0.15])
    b.paint_uniform_color([0.15, 0.55, 0.95])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900, visible=False)
    vis.add_geometry(a)
    vis.add_geometry(b)
    vis.get_render_option().point_size = 2.0
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output), do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Register two smartphone 3D scans.")
    parser.add_argument("--source", required=True, help="First scan: OBJ/GLB/PLY/PCD")
    parser.add_argument("--target", required=True, help="Second scan: OBJ/GLB/PLY/PCD")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="Voxel size in scan units. Default: 1%% of combined bounding-box diagonal.")
    parser.add_argument("--sample-points", type=int, default=80000,
                        help="Poisson-disk samples per input mesh (default: 80000).")
    parser.add_argument("--output", default="results/day4",
                        help="Output directory (default: results/day4).")
    args = parser.parse_args()

    source_path, target_path = Path(args.source), Path(args.target)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading scans...")
    source_raw = load_as_point_cloud(source_path, args.sample_points)
    target_raw = load_as_point_cloud(target_path, args.sample_points)

    combined = np.vstack([np.asarray(source_raw.points), np.asarray(target_raw.points)])
    diagonal = float(np.linalg.norm(combined.max(axis=0) - combined.min(axis=0)))
    voxel = args.voxel_size if args.voxel_size is not None else max(diagonal * 0.01, 1e-5)
    if voxel <= 0:
        raise ValueError("--voxel-size must be positive.")

    print(f"[INFO] Raw points: source={len(source_raw.points):,}, target={len(target_raw.points):,}")
    print(f"[INFO] Bounding-box diagonal={diagonal:.5f}; voxel_size={voxel:.5f}")

    print("[2/6] Cleaning, downsampling and estimating normals...")
    source = clean_cloud(source_raw, voxel)
    target = clean_cloud(target_raw, voxel)
    o3d.io.write_point_cloud(str(output / "source_processed.ply"), source)
    o3d.io.write_point_cloud(str(output / "target_processed.ply"), target)

    threshold = voxel * 2.0
    before = evaluate(source, target, np.eye(4), threshold)

    print("[3/6] Global registration: FPFH + RANSAC...")
    start = time.perf_counter()
    ransac = global_registration(source, target, voxel)
    ransac_seconds = time.perf_counter() - start
    after_ransac = evaluate(source, target, ransac.transformation, threshold)

    print("[4/6] Local refinement: point-to-plane ICP...")
    source_for_icp = o3d.geometry.PointCloud(source)
    source_for_icp.transform(ransac.transformation)
    start = time.perf_counter()
    icp = o3d.pipelines.registration.registration_icp(
        source_for_icp, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    icp_seconds = time.perf_counter() - start
    final_transform = icp.transformation @ ransac.transformation
    final_eval = evaluate(source, target, final_transform, threshold)

    print("[5/6] Running custom NumPy/SciPy ICP from the RANSAC initialization...")
    start = time.perf_counter()
    my_transform, my_history = custom_icp(
        np.asarray(source.points), np.asarray(target.points),
        ransac.transformation, threshold, max_iter=60
    )
    my_seconds = time.perf_counter() - start
    my_eval = evaluate(source, target, my_transform, threshold)

    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(final_transform)
    merged = aligned + target
    merged = merged.voxel_down_sample(voxel * 0.5)
    o3d.io.write_point_cloud(str(output / "merged.ply"), merged)

    # Save a combined before/after view as two images.
    save_screenshot(source, target, output / "before.png", "Before registration")
    save_screenshot(aligned, target, output / "after.png", "After registration")

    def metrics(result):
        return {
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
        }

    report = {
        "inputs": {"source": str(source_path), "target": str(target_path)},
        "units_note": "voxel_size and thresholds use the input scan coordinate units.",
        "parameters": {
            "voxel_size": voxel,
            "ransac_distance_threshold": voxel * 1.5,
            "icp_distance_threshold": threshold,
            "source_points_after_preprocessing": len(source.points),
            "target_points_after_preprocessing": len(target.points),
        },
        "metrics": {
            "before": metrics(before),
            "after_ransac": metrics(after_ransac),
            "ransac_plus_open3d_point_to_plane": metrics(final_eval),
            "ransac_plus_custom_point_to_point": metrics(my_eval),
        },
        "runtime_seconds": {
            "ransac": ransac_seconds,
            "open3d_icp": icp_seconds,
            "custom_icp": my_seconds,
        },
        "custom_icp_rmse_history": my_history,
        "transformations": {
            "ransac": ransac.transformation.tolist(),
            "open3d_final": final_transform.tolist(),
            "custom_final": my_transform.tolist(),
        },
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[6/6] Done.")
    print(f"[RESULT] Open3D: fitness={final_eval.fitness:.4f}, RMSE={final_eval.inlier_rmse:.6f}")
    print(f"[RESULT] Custom: fitness={my_eval.fitness:.4f}, RMSE={my_eval.inlier_rmse:.6f}")
    print(f"[SAVED] {output.resolve()}")
    print("Files: merged.ply, metrics.json, before.png, after.png, processed scans")


if __name__ == "__main__":
    main()
