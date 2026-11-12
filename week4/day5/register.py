#!/usr/bin/env python
"""
Week 4 — Day 5: One-command point-cloud registration pipeline.

Examples:
  python register.py --source data/scan1.obj --target data/scan2.obj --method p2l
  python register.py --source data/scan1.ply --target data/scan2.ply --method p2p --voxel-size 0.02
  python register.py --source data/scan1.obj --target data/scan2.obj --method my_icp --make-gif

Dependencies:
  pip install open3d numpy scipy imageio pillow
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_cloud(path: Path, sample_points: int) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".obj", ".glb", ".gltf", ".stl", ".fbx", ".off"}:
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise ValueError(f"Could not load a triangle mesh: {path}")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        cloud = mesh.sample_points_poisson_disk(number_of_points=sample_points)
    else:
        cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Could not load point cloud: {path}")
    cloud.remove_non_finite_points()
    cloud.remove_duplicated_points()
    return cloud


def preprocess(cloud, voxel):
    cloud = o3d.geometry.PointCloud(cloud)
    if len(cloud.points) > 100:
        _, indices = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        cloud = cloud.select_by_index(indices)
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) < 10:
        raise ValueError("Preprocessing left too few points; lower --voxel-size.")
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    cloud.normalize_normals()
    return cloud


def fpfh(cloud, voxel):
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )


def global_init(source, target, voxel):
    distance = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, fpfh(source, voxel), fpfh(target, voxel),
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


def rigid_transform(a, b):
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = cb - r @ ca
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = r, t
    return matrix


def numpy_icp(source, target, initial, threshold, max_iter=80, tolerance=1e-7):
    src = np.asarray(source.points)
    dst = np.asarray(target.points)
    tree = cKDTree(dst)
    transform = initial.copy()
    history = []
    previous = np.inf
    for _ in range(max_iter):
        moved = (transform[:3, :3] @ src.T).T + transform[:3, 3]
        distances, indices = tree.query(moved, k=1, workers=-1)
        mask = distances < threshold
        if mask.sum() < 6:
            break
        delta = rigid_transform(moved[mask], dst[indices[mask]])
        transform = delta @ transform
        updated = (transform[:3, :3] @ src.T).T + transform[:3, 3]
        d, _ = tree.query(updated, k=1, workers=-1)
        valid = d < threshold
        if not valid.any():
            break
        rmse = float(np.sqrt(np.mean(d[valid] ** 2)))
        history.append(rmse)
        if abs(previous - rmse) < tolerance:
            break
        previous = rmse
    return transform, history


def make_gif(source, target, output_path: Path, frames=36):
    """Capture a rotating, two-colour view using Open3D Visualizer."""
    src = o3d.geometry.PointCloud(source)
    tgt = o3d.geometry.PointCloud(target)
    src.paint_uniform_color([0.95, 0.35, 0.15])
    tgt.paint_uniform_color([0.15, 0.55, 0.95])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ICP registration", width=1000, height=750, visible=False)
    vis.add_geometry(src)
    vis.add_geometry(tgt)
    vis.get_render_option().point_size = 2.0
    view = vis.get_view_control()
    images = []
    for _ in range(frames):
        vis.poll_events()
        vis.update_renderer()
        images.append(np.asarray(vis.capture_screen_float_buffer(do_render=True))[:, :, :3])
        view.rotate(10.0, 0.0)
    vis.destroy_window()
    imageio.mimsave(output_path, [(np.clip(frame, 0, 1) * 255).astype(np.uint8)
                                  for frame in images], duration=0.08, loop=0)


def main():
    parser = argparse.ArgumentParser(description="Command-line ICP registration pipeline.")
    parser.add_argument("--source", required=True, help="Source scan: OBJ/GLB/PLY/PCD")
    parser.add_argument("--target", required=True, help="Target scan: OBJ/GLB/PLY/PCD")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="Voxel size in scan units; default is 1%% of bounding-box diagonal.")
    parser.add_argument("--method", choices=["p2p", "p2l", "my_icp"], default="p2l",
                        help="Final local refinement method.")
    parser.add_argument("--output", default="results",
                        help="Output folder (default: results).")
    parser.add_argument("--sample-points", type=int, default=80000,
                        help="Number of Poisson-disk samples for each mesh.")
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--no-global", action="store_true",
                        help="Skip FPFH+RANSAC; use identity as initial alignment.")
    parser.add_argument("--make-gif", action="store_true",
                        help="Save a rotating two-colour GIF of the final alignment.")
    args = parser.parse_args()

    source_path, target_path = Path(args.source), Path(args.target)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading input scans...")
    source_raw = load_cloud(source_path, args.sample_points)
    target_raw = load_cloud(target_path, args.sample_points)
    all_points = np.vstack([np.asarray(source_raw.points), np.asarray(target_raw.points)])
    diagonal = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
    voxel = args.voxel_size if args.voxel_size is not None else max(diagonal * 0.01, 1e-5)
    if voxel <= 0:
        parser.error("--voxel-size must be greater than zero.")
    threshold = voxel * 2.0

    print("[2/5] Preprocessing...")
    source, target = preprocess(source_raw, voxel), preprocess(target_raw, voxel)
    o3d.io.write_point_cloud(str(output / "source_processed.ply"), source)
    o3d.io.write_point_cloud(str(output / "target_processed.ply"), target)

    registration = o3d.pipelines.registration
    identity = np.eye(4)
    before = registration.evaluate_registration(source, target, threshold, identity)

    print("[3/5] Initial alignment...")
    start = time.perf_counter()
    if args.no_global:
        initial = identity
        global_result = None
    else:
        global_result = global_init(source, target, voxel)
        initial = global_result.transformation
    global_seconds = time.perf_counter() - start

    print(f"[4/5] Refinement: {args.method}...")
    start = time.perf_counter()
    history = []
    if args.method == "p2p":
        result = registration.registration_icp(
            source, target, threshold, initial,
            registration.TransformationEstimationPointToPoint(),
            registration.ICPConvergenceCriteria(max_iteration=args.max_iterations),
        )
        transform = result.transformation
    elif args.method == "p2l":
        result = registration.registration_icp(
            source, target, threshold, initial,
            registration.TransformationEstimationPointToPlane(),
            registration.ICPConvergenceCriteria(max_iteration=args.max_iterations),
        )
        transform = result.transformation
    else:
        transform, history = numpy_icp(
            source, target, initial, threshold, max_iter=args.max_iterations
        )
        result = registration.evaluate_registration(source, target, threshold, transform)
    refinement_seconds = time.perf_counter() - start

    final_eval = registration.evaluate_registration(source, target, threshold, transform)
    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(transform)
    merged = aligned + target
    merged = merged.voxel_down_sample(voxel * 0.5)

    o3d.io.write_point_cloud(str(output / "aligned_source.ply"), aligned)
    o3d.io.write_point_cloud(str(output / "merged.ply"), merged)

    # Save a static preview; the optional GIF is generated below.
    preview_a = o3d.geometry.PointCloud(aligned)
    preview_b = o3d.geometry.PointCloud(target)
    preview_a.paint_uniform_color([0.95, 0.35, 0.15])
    preview_b.paint_uniform_color([0.15, 0.55, 0.95])
    o3d.visualization.draw_geometries  # keep Open3D visualization import explicit

    def metric(obj):
        return {"fitness": float(obj.fitness), "inlier_rmse": float(obj.inlier_rmse)}

    report = {
        "source": str(source_path),
        "target": str(target_path),
        "method": args.method,
        "global_registration": "disabled" if args.no_global else "FPFH + RANSAC",
        "parameters": {
            "voxel_size": voxel,
            "icp_distance_threshold": threshold,
            "ransac_distance_threshold": voxel * 1.5,
            "source_points": len(source.points),
            "target_points": len(target.points),
            "max_iterations": args.max_iterations,
        },
        "metrics": {
            "before": metric(before),
            "after": metric(final_eval),
            "after_global_registration": (
                None if global_result is None else metric(
                    registration.evaluate_registration(
                        source, target, threshold, initial
                    )
                )
            ),
        },
        "runtime_seconds": {
            "global_registration": global_seconds,
            "refinement": refinement_seconds,
        },
        "transformation": transform.tolist(),
        "custom_icp_rmse_history": history,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.make_gif:
        print("[GIF] Capturing rotating preview...")
        make_gif(aligned, target, output / "icp_demo.gif")

    print("[5/5] Complete.")
    print(f"[RESULT] fitness={final_eval.fitness:.4f}; inlier RMSE={final_eval.inlier_rmse:.6f}")
    print(f"[SAVED] {output.resolve()}")
    print("Outputs: merged.ply, aligned_source.ply, metrics.json, processed scans")
    if args.make_gif:
        print("         icp_demo.gif")


if __name__ == "__main__":
    main()
