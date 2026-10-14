"""
Day 4 — Chessboard Corner Detection

Detect internal chessboard corners in calibration images,
refine them with cornerSubPix, and save 3D/2D point pairs
for the next step (camera calibration).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Settings
# ============================================================

CHESSBOARD_SIZE = (9, 6)  # internal corners: (cols, rows)
SQUARE_SIZE = 0.025  # meters (2.5 cm)

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
IMAGES_DIR = BASE_DIR / "calibration_images"
OUTPUT_DIR = BASE_DIR / "results"


def build_object_points(
    board_size: tuple[int, int],
    square_size: float,
) -> np.ndarray:
    """3D coordinates of chessboard corners on the Z = 0 plane."""
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


def process_image(
    path: Path,
    objp: np.ndarray,
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
) -> bool:
    """
    Detect and refine chessboard corners.
    Returns True if corners were found.
    """
    img = cv2.imread(str(path))
    if img is None:
        print(f"  Failed to load: {path.name}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, FIND_FLAGS)

    if not ret:
        print(f"  Corners NOT found: {path.name}")
        return False

    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        CRITERIA,
    )

    objpoints.append(objp.copy())
    imgpoints.append(corners_refined)

    # Draw and save visualization
    vis = img.copy()
    cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners_refined, ret)
    out_path = OUTPUT_DIR / f"{path.stem}_corners.jpg"
    cv2.imwrite(str(out_path), vis)

    print(f"  Corners found → {out_path.name}")

    cv2.imshow("Chessboard Corners", vis)
    cv2.waitKey(500)

    return True


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Chessboard corner detection")
    print(f"  Board size (internal corners): {CHESSBOARD_SIZE}")
    print(f"  Square size: {SQUARE_SIZE} m")
    print()

    if not IMAGES_DIR.is_dir():
        print(f"Folder not found: {IMAGES_DIR}")
        print("Create calibration_images/ and put chessboard photos there.")
        return

    image_paths = collect_images(IMAGES_DIR)
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        print("Supported formats: jpg, jpeg, png")
        return

    objp = build_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    print(f"Found {len(image_paths)} image(s)")
    print(f"Object points per board: {len(objp)}")
    print()

    for path in image_paths:
        print(f"Processing: {path.name}")
        process_image(path, objp, objpoints, imgpoints)

    cv2.destroyAllWindows()

    print()
    print("=" * 40)
    print("Result")
    print("=" * 40)
    print(f"Total images:              {len(image_paths)}")
    print(f"Boards detected:           {len(objpoints)}")

    if objpoints:
        # Save for Day 5 (calibration)
        np.save(OUTPUT_DIR / "objpoints.npy", np.array(objpoints, dtype=object))
        np.save(OUTPUT_DIR / "imgpoints.npy", np.array(imgpoints, dtype=object))
        print(f"Saved: {OUTPUT_DIR / 'objpoints.npy'}")
        print(f"Saved: {OUTPUT_DIR / 'imgpoints.npy'}")
        print("Ready for camera calibration (Day 5).")
    else:
        print("No chessboard found in any image.")
        print("Check CHESSBOARD_SIZE and photo quality/angles.")


if __name__ == "__main__":
    main()