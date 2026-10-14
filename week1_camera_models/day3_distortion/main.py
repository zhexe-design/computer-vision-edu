"""
Day 3 — Lens Distortion

Apply artificial radial (barrel) distortion to photos and remove it with OpenCV.
Produces Original | Distorted | Undistorted comparison images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "results"

# ============================================================
# Camera / distortion settings
# ============================================================

FX = 800.0
FY = 800.0

# OpenCV order: k1, k2, p1, p2, k3
# Negative k1 → barrel distortion
DIST_COEFFS = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def make_camera_matrix(width: int, height: int) -> np.ndarray:
    """Intrinsic matrix K for the given image size."""
    return np.array(
        [
            [FX, 0, width / 2],
            [0, FY, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def apply_approx_distortion(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Approximate *adding* distortion by applying the inverse of the
    undistortion map to a clean image (common learning demo).
    """
    h, w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        K,
        (w, h),
        cv2.CV_32FC1,
    )
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)


def remove_distortion(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Remove distortion with cv2.undistort."""
    return cv2.undistort(img, K, dist)


def label_image(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def resize_for_display(image: np.ndarray, max_height: int = 600) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= max_height:
        return image
    scale = max_height / h
    return cv2.resize(
        image,
        (int(w * scale), max_height),
        interpolation=cv2.INTER_AREA,
    )


def collect_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(folder.glob(ext))
    return sorted(paths)


def process_image(image_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read: {image_path}")
        return
    if img.ndim != 3:
        print(f"Skip non-color image: {image_path}")
        return

    h, w = img.shape[:2]
    name = image_path.stem
    print(f"Processing: {image_path.name}  ({w} x {h})")

    K = make_camera_matrix(w, h)

    # 1) Approximate add distortion
    distorted = apply_approx_distortion(img, K, DIST_COEFFS)

    # 2) Remove distortion
    undistorted = remove_distortion(distorted, K, DIST_COEFFS)

    # 3) Comparison strip
    original_l = label_image(img, "Original")
    distorted_l = label_image(distorted, "Distorted")
    undistorted_l = label_image(undistorted, "Undistorted")
    combined = np.hstack([original_l, distorted_l, undistorted_l])

    # 4) Save
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_original.jpg"), img)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_distorted.jpg"), distorted)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_undistorted.jpg"), undistorted)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_comparison.jpg"), combined)

    # 5) Show (scaled for screen)
    cv2.imshow(
        f"Original | Distorted | Undistorted — {image_path.name}",
        resize_for_display(combined),
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Images folder not found: {INPUT_DIR}")

    image_paths = collect_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        print("Put .jpg / .png files into the images/ folder and run again.")
        return

    print(f"Found {len(image_paths)} image(s)")
    print(f"Distortion coeffs (k1,k2,p1,p2,k3): {DIST_COEFFS.tolist()}")
    print()

    for path in image_paths:
        process_image(path)

    print()
    print("=" * 40)
    print("All images processed.")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 40)


if __name__ == "__main__":
    main()