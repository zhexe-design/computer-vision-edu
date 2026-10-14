"""
Day 5 — Camera Calibration

Full OpenCV calibration pipeline:
detect chessboard corners → calibrateCamera → reprojection error →
save parameters → undistort comparisons → draw frame axes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Settings
# ============================================================

CHESSBOARD_SIZE = (9, 6)  # internal corners (cols, rows)
SQUARE_SIZE = 0.025  # meters

CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = BASE_DIR / "calibration_images"
RESULTS_DIR = BASE_DIR / "results"


def build_object_points(
    board_size: tuple[int, int],
    square_size: float,
) -> np.ndarray:
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def collect_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(folder.glob(ext))
    return sorted(paths)


def detect_corners(gray: np.ndarray) -> tuple[bool, np.ndarray | None]:
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, FIND_FLAGS)
    if not ret:
        return False, None
    corners_refined = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), CRITERIA
    )
    return True, corners_refined


def compute_reprojection_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list,
    tvecs: list,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[float, list[float]]:
    per_image: list[float] = []
    total = 0.0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )
        original = imgpoints[i].reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        err = cv2.norm(original, projected, cv2.NORM_L2) / len(projected)
        per_image.append(float(err))
        total += err
    mean_err = total / len(objpoints)
    return mean_err, per_image


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Camera calibration")
    print(f"  Board size (internal corners): {CHESSBOARD_SIZE}")
    print(f"  Square size: {SQUARE_SIZE} m")
    print()

    if not CALIBRATION_DIR.is_dir():
        print(f"Folder not found: {CALIBRATION_DIR}")
        print("Create calibration_images/ and put chessboard photos there.")
        return

    image_paths = collect_images(CALIBRATION_DIR)
    if not image_paths:
        print(f"No images found in {CALIBRATION_DIR}")
        return

    objp = build_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    successful_paths: list[Path] = []
    image_size: tuple[int, int] | None = None

    print(f"Found {len(image_paths)} image(s)")
    print()

    for path in image_paths:
        print(f"Processing: {path.name}")
        img = cv2.imread(str(path))
        if img is None:
            print("  Failed to load.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        ok, corners = detect_corners(gray)
        if not ok or corners is None:
            print("  Corners NOT found.")
            print()
            continue

        print("  Corners found.")
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        successful_paths.append(path)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, True)
        out_corners = RESULTS_DIR / f"corners_{path.name}"
        cv2.imwrite(str(out_corners), vis)
        print(f"  Saved: {out_corners.name}")
        print()

    print("=" * 40)
    print("Corner detection")
    print("=" * 40)
    print(f"Total images:     {len(image_paths)}")
    print(f"Boards detected:  {len(objpoints)}")
    print()

    if len(objpoints) < 3:
        print("Need at least 3 successful detections for calibration.")
        return

    assert image_size is not None

    # ----------------------------------------------------------
    # Calibrate
    # ----------------------------------------------------------
    print("Running cv2.calibrateCamera ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    mean_error, per_image_errors = compute_reprojection_errors(
        objpoints, imgpoints, rvecs, tvecs, K, dist
    )

    print()
    print("=" * 40)
    print("Calibration result")
    print("=" * 40)
    print(f"RMS (OpenCV):              {rms:.6f}")
    print(f"Mean reprojection error:   {mean_error:.6f} px")
    print()
    print("Per-image errors (px):")
    for i, e in enumerate(per_image_errors, start=1):
        print(f"  Image {i}: {e:.6f}")
    print()
    print("Camera matrix K:")
    print(K)
    print()
    print("Distortion coefficients [k1, k2, p1, p2, k3]:")
    print(dist.ravel())

    # ----------------------------------------------------------
    # Save parameters
    # ----------------------------------------------------------
    calib_path = RESULTS_DIR / "camera_calibration.npz"
    np.savez(
        calib_path,
        K=K,
        dist=dist,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        image_width=image_size[0],
        image_height=image_size[1],
        mean_reprojection_error=mean_error,
        rms_error=rms,
        chessboard_size=np.array(CHESSBOARD_SIZE),
        square_size=SQUARE_SIZE,
    )
    print()
    print(f"Saved: {calib_path}")

    # ----------------------------------------------------------
    # Axes on first successful image
    # ----------------------------------------------------------
    axis_img = cv2.imread(str(successful_paths[0]))
    if axis_img is not None:
        gray = cv2.cvtColor(axis_img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_corners(gray)
        if ok and corners is not None:
            cv2.drawFrameAxes(axis_img, K, dist, rvecs[0], tvecs[0], 0.1)
            axes_path = RESULTS_DIR / "axes_visualization.jpg"
            cv2.imwrite(str(axes_path), axis_img)
            print(f"Saved: {axes_path}")

    # ----------------------------------------------------------
    # Undistort comparisons (first 3 successful images)
    # ----------------------------------------------------------
    print()
    print("=" * 40)
    print("Undistortion comparisons")
    print("=" * 40)

    for i, path in enumerate(successful_paths[:3], start=1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        undistorted = cv2.undistort(img, K, dist, None, K)

        left = img.copy()
        right = undistorted.copy()
        cv2.putText(left, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(right, "Undistorted", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        comparison = np.hstack([left, right])

        out = RESULTS_DIR / f"comparison_{i}.jpg"
        cv2.imwrite(str(out), comparison)
        print(f"Saved: {out.name}")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print()
    print("=" * 40)
    print("FINAL CALIBRATION PARAMETERS")
    print("=" * 40)
    print(f"1. Mean reprojection error: {mean_error:.6f} px")
    print()
    print("2. Camera matrix K:")
    print(K)
    print()
    print("3. Distortion coefficients:")
    print(dist.ravel())
    print()
    print(f"Results folder: {RESULTS_DIR}")
    print("  - camera_calibration.npz")
    print("  - corners_*.jpg")
    print("  - axes_visualization.jpg")
    print("  - comparison_*.jpg")


if __name__ == "__main__":
    main()