"""
Day 2 — Calibration Quality Analysis

Run from any working directory:
    python path/to/day2_quality_analysis/main.py

Expected input:
    calibration_images/  - chessboard images

Outputs:
    results/corners/             detected chessboard corners
    results/before_after/        calibration comparison chart
    results/error_analysis.csv   per-image reprojection errors
    results/camera_params.yaml   final camera parameters
    results/camera_params.npz    final camera parameters
    results/axes_visualization.jpg
"""

from pathlib import Path
import csv
import cv2
import numpy as np
import yaml


BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = BASE_DIR / "calibration_images"
RESULTS_DIR = BASE_DIR / "results"
CORNERS_DIR = RESULTS_DIR / "corners"
COMPARISON_DIR = RESULTS_DIR / "before_after"

CHESSBOARD_SIZE = (9, 6)  # inner corners: columns, rows
SQUARE_SIZE = 0.025       # meters
NUM_BAD_IMAGES_TO_REMOVE = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def create_directories():
    """Create output directories."""
    for folder in (RESULTS_DIR, CORNERS_DIR, COMPARISON_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def find_images(folder):
    """Return supported image paths in deterministic order."""
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_object_points():
    """Create chessboard points in the board's coordinate system."""
    points = np.zeros(
        (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
        dtype=np.float32,
    )
    points[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0],
        0:CHESSBOARD_SIZE[1],
    ].T.reshape(-1, 2)
    points *= SQUARE_SIZE
    return points


def detect_chessboards(image_paths):
    """Detect and refine chessboard corners; preserve image names and size."""
    object_template = make_object_points()
    object_points = []
    image_points = []
    image_names = []
    image_sizes = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Could not read: {image_path.name}")
            continue

        height, width = image.shape[:2]
        image_sizes.append((width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if not found:
            print(f"[SKIP] Chessboard not found: {image_path.name}")
            continue

        refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        object_points.append(object_template.copy())
        image_points.append(refined)
        image_names.append(image_path.name)

        preview = image.copy()
        cv2.drawChessboardCorners(preview, CHESSBOARD_SIZE, refined, found)
        cv2.imwrite(str(CORNERS_DIR / f"corners_{image_path.stem}.jpg"), preview)
        print(f"[OK] Chessboard found: {image_path.name}")

    if not object_points:
        raise RuntimeError(
            f"No chessboard corners found. Check CHESSBOARD_SIZE={CHESSBOARD_SIZE} "
            f"and images in: {CALIBRATION_DIR}"
        )

    # Calibration requires a consistent image resolution.
    detected_sizes = set()
    for name in image_names:
        image = cv2.imread(str(CALIBRATION_DIR / name))
        if image is not None:
            detected_sizes.add((image.shape[1], image.shape[0]))
    if len(detected_sizes) != 1:
        raise ValueError(
            "Calibration images must all have the same resolution. "
            f"Detected resolutions: {sorted(detected_sizes)}"
        )

    image_size = next(iter(detected_sizes))
    return object_points, image_points, image_names, image_size


def reprojection_errors(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Calculate mean point reprojection error for each image."""
    errors = []
    for obj, observed, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            obj, rvec, tvec, camera_matrix, dist_coeffs
        )
        observed = observed.reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        errors.append(float(np.mean(np.linalg.norm(observed - projected, axis=1))))
    return errors


def calibrate(object_points, image_points, image_size):
    """Calibrate camera and calculate per-image reprojection errors."""
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors = reprojection_errors(
        object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
    )
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs, errors, float(np.mean(errors))


def save_error_csv(path, names, errors, removed_names=None):
    """Save per-image error results to CSV."""
    removed_names = set(removed_names or [])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "reprojection_error_px", "selected_for_removal"])
        for name, error in sorted(zip(names, errors), key=lambda item: item[1], reverse=True):
            writer.writerow([name, f"{error:.6f}", "yes" if name in removed_names else "no"])


def save_parameters(camera_matrix, dist_coeffs, image_size, mean_error, image_count):
    """Save final camera parameters in YAML and NumPy formats."""
    width, height = image_size
    data = {
        "K": camera_matrix.tolist(),
        "dist": dist_coeffs.reshape(-1).tolist(),
        "image_width": int(width),
        "image_height": int(height),
        "reprojection_error": float(mean_error),
        "num_images": int(image_count),
        "chessboard_columns": int(CHESSBOARD_SIZE[0]),
        "chessboard_rows": int(CHESSBOARD_SIZE[1]),
        "square_size_m": float(SQUARE_SIZE),
    }
    with (RESULTS_DIR / "camera_params.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    np.savez(
        RESULTS_DIR / "camera_params.npz",
        K=camera_matrix,
        dist=dist_coeffs,
        image_width=width,
        image_height=height,
        reprojection_error=mean_error,
        num_images=image_count,
    )


def save_comparison_chart(before_count, before_error, after_count, after_error):
    """Save a simple before/after summary image."""
    canvas = np.full((420, 1000, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Calibration quality: before vs after filtering",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    lines = [
        f"Before: {before_count} images | mean reprojection error: {before_error:.4f} px",
        f"After:  {after_count} images | mean reprojection error: {after_error:.4f} px",
    ]
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (40, 145 + index * 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    if before_error > 0:
        improvement = (before_error - after_error) / before_error * 100
        cv2.putText(canvas, f"Change in mean error: {improvement:+.2f}%",
                    (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (25, 25, 25), 2, cv2.LINE_AA)
    cv2.imwrite(str(COMPARISON_DIR / "calibration_before_after.jpg"), canvas)


def save_axes_visualization(image_path, camera_matrix, dist_coeffs):
    """Estimate the chessboard pose and draw coordinate axes."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARNING] Could not load image for axes: {image_path.name}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if not found:
        print("[WARNING] Could not detect chessboard for axes visualization.")
        return

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    success, rvec, tvec = cv2.solvePnP(
        make_object_points(), corners, camera_matrix, dist_coeffs
    )
    if success:
        cv2.drawFrameAxes(
            image, camera_matrix, dist_coeffs, rvec, tvec, SQUARE_SIZE * 3
        )
        cv2.imwrite(str(RESULTS_DIR / "axes_visualization.jpg"), image)


def main():
    create_directories()
    print("=" * 64)
    print("DAY 2 — CALIBRATION QUALITY ANALYSIS")
    print("=" * 64)

    image_paths = find_images(CALIBRATION_DIR)
    if not image_paths:
        raise FileNotFoundError(
            f"No calibration images found in {CALIBRATION_DIR}. "
            "Create this folder and add chessboard images."
        )

    object_points, image_points, names, image_size = detect_chessboards(image_paths)
    if len(object_points) <= NUM_BAD_IMAGES_TO_REMOVE:
        raise ValueError(
            f"Only {len(object_points)} usable images were detected, but "
            f"NUM_BAD_IMAGES_TO_REMOVE={NUM_BAD_IMAGES_TO_REMOVE}. "
            "Reduce the number of images to remove."
        )

    print(f"\nUsable images: {len(names)}")
    print(f"Calibration resolution: {image_size[0]} x {image_size[1]}")

    rms_before, K_before, dist_before, rvecs_before, tvecs_before, errors_before, mean_before = (
        calibrate(object_points, image_points, image_size)
    )
    ranked = sorted(zip(names, errors_before), key=lambda item: item[1], reverse=True)
    bad_names = {name for name, _ in ranked[:NUM_BAD_IMAGES_TO_REMOVE]}

    print("\nPer-image reprojection error (highest first):")
    for name, error in ranked:
        marker = " [REMOVE]" if name in bad_names else ""
        print(f"  {name}: {error:.6f} px{marker}")

    filtered = [
        (obj, img, name)
        for obj, img, name in zip(object_points, image_points, names)
        if name not in bad_names
    ]
    filtered_objects = [item[0] for item in filtered]
    filtered_images = [item[1] for item in filtered]
    filtered_names = [item[2] for item in filtered]

    rms_after, K_after, dist_after, rvecs_after, tvecs_after, errors_after, mean_after = (
        calibrate(filtered_objects, filtered_images, image_size)
    )

    save_error_csv(
        RESULTS_DIR / "error_analysis.csv", names, errors_before, removed_names=bad_names
    )
    save_parameters(K_after, dist_after, image_size, mean_after, len(filtered_names))
    save_comparison_chart(len(names), mean_before, len(filtered_names), mean_after)

    # Use the first retained image for the coordinate axes visualization.
    save_axes_visualization(CALIBRATION_DIR / filtered_names[0], K_after, dist_after)

    improvement = ((mean_before - mean_after) / mean_before * 100) if mean_before else 0.0
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Images before filtering: {len(names)}")
    print(f"Images removed:          {len(bad_names)}")
    print(f"Images after filtering:  {len(filtered_names)}")
    print(f"Mean error before:       {mean_before:.6f} px")
    print(f"Mean error after:        {mean_after:.6f} px")
    print(f"Relative change:         {improvement:+.2f}%")
    print(f"OpenCV RMS before:       {rms_before:.6f}")
    print(f"OpenCV RMS after:        {rms_after:.6f}")
    print(f"Results saved to:        {RESULTS_DIR}")

    if mean_after >= mean_before:
        print(
            "\nNOTE: Filtering did not reduce the mean error for this run. "
            "Use the measured result in your README; do not claim an improvement "
            "unless the numbers confirm it."
        )


if __name__ == "__main__":
    main()
