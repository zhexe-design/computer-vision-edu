"""
Day 6 — Calibration Validation & Distance Measurement

Two modes:
  auto  — measure known chessboard corner distances vs ground truth
  click — click two points on a photo and get distance in mm

IMPORTANT: measurements are valid only for points that lie in the SAME
plane as the chessboard (e.g. an object flat on the same table).

Usage:
    python main.py auto
    python main.py click --image test_images/board_01.jpg --known 85.6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Settings
# ============================================================

CHESSBOARD_SIZE = (9, 6)  # inner corners (cols, rows)
SQUARE_SIZE = 0.025  # meters
MAX_WINDOW_WIDTH = 1200

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
IMAGES_DIR = BASE_DIR / "test_images"
RESULTS_DIR = BASE_DIR / "results"

CALIB_CANDIDATES = [
    BASE_DIR / "camera_calibration.npz",
    BASE_DIR.parent / "day5_calibration" / "results" / "camera_calibration.npz",
]


# ============================================================
# Calibration & geometry
# ============================================================

def resolve_calib_path() -> Path:
    for path in CALIB_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "camera_calibration.npz not found.\n"
        f"  Tried: {[str(p) for p in CALIB_CANDIDATES]}\n"
        "Put the file in day6_testing/ or run Day 5 first."
    )


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    size = (int(data["image_width"]), int(data["image_height"]))
    return K, dist, size


def make_object_points() -> np.ndarray:
    """3D chessboard corners on Z = 0, in meters."""
    cols, rows = CHESSBOARD_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * SQUARE_SIZE


def find_pose(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Chessboard pose relative to the camera. Returns rvec, tvec, corners."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, FIND_FLAGS)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    objp = make_object_points()
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        return None
    return rvec, tvec, corners


def pixel_to_plane(
    pixel: tuple[float, float] | np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    Map one image pixel to a 3D point on the chessboard plane (Z = 0).
    Returns coordinates in the chessboard frame, in meters.
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    pt = np.array([[pixel]], dtype=np.float32)
    x, y = cv2.undistortPoints(pt, K, dist).reshape(2)
    ray = np.array([x, y, 1.0], dtype=np.float64)

    normal = R[:, 2].astype(np.float64)
    denom = float(normal.dot(ray))
    if abs(denom) < 1e-8:
        raise ValueError("Ray is parallel to the chessboard plane")
    s = float(normal.dot(t)) / denom
    if s <= 0:
        raise ValueError("Intersection is behind the camera")

    point_cam = s * ray
    return (R.T @ (point_cam - t)).astype(np.float64)


def measure_distance_m(
    p1: tuple[float, float] | np.ndarray,
    p2: tuple[float, float] | np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """Real-world distance between two pixels (meters), on the board plane."""
    a = pixel_to_plane(p1, K, dist, rvec, tvec)
    b = pixel_to_plane(p2, K, dist, rvec, tvec)
    return float(np.linalg.norm(a - b))


def check_image_size(img: np.ndarray, calib_size: tuple[int, int]) -> None:
    h, w = img.shape[:2]
    if (w, h) != calib_size:
        print(
            f"WARNING: image size is {w}x{h}, "
            f"calibration was for {calib_size[0]}x{calib_size[1]}. "
            "Results may be wrong."
        )


def collect_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(folder.glob(ext))
    return sorted(paths)


# ============================================================
# Mode: auto
# ============================================================

def run_auto_test(
    K: np.ndarray,
    dist: np.ndarray,
    calib_size: tuple[int, int],
) -> None:
    if not IMAGES_DIR.is_dir():
        print(f"Folder not found: {IMAGES_DIR}")
        return

    images = collect_images(IMAGES_DIR)
    if not images:
        print(f"No images in {IMAGES_DIR}")
        return

    cols, rows = CHESSBOARD_SIZE
    pairs = [
        ((0, 0), (1, 0)),
        ((0, 0), (4, 0)),
        ((0, 0), (cols - 1, 0)),
        ((0, 0), (0, rows - 1)),
        ((0, 0), (cols - 1, rows - 1)),
    ]

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "auto_test.csv"
    all_errors: list[float] = []
    first_vis_saved = False

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image", "pair", "true_mm", "measured_mm", "error_mm", "error_percent"]
        )

        for path in images:
            img = cv2.imread(str(path))
            if img is None:
                continue
            check_image_size(img, calib_size)

            pose = find_pose(img, K, dist)
            if pose is None:
                print(f"{path.name}: chessboard not found, skipped")
                continue

            rvec, tvec, corners = pose
            corners = corners.reshape(-1, 2)

            print(f"\n{path.name}")
            for (c1, r1), (c2, r2) in pairs:
                i1 = r1 * cols + c1
                i2 = r2 * cols + c2
                true_mm = SQUARE_SIZE * float(np.hypot(c2 - c1, r2 - r1)) * 1000
                measured_mm = (
                    measure_distance_m(corners[i1], corners[i2], K, dist, rvec, tvec)
                    * 1000
                )
                error_mm = measured_mm - true_mm
                error_pct = error_mm / true_mm * 100
                all_errors.append(abs(error_mm))

                name = f"({c1},{r1})-({c2},{r2})"
                print(
                    f"  {name:>14}: true {true_mm:7.2f} mm | "
                    f"measured {measured_mm:7.2f} mm | "
                    f"error {error_mm:+.2f} mm ({error_pct:+.2f}%)"
                )
                writer.writerow(
                    [
                        path.name,
                        name,
                        f"{true_mm:.2f}",
                        f"{measured_mm:.2f}",
                        f"{error_mm:.2f}",
                        f"{error_pct:.2f}",
                    ]
                )

            if not first_vis_saved:
                vis = img.copy()
                for (c1, r1), (c2, r2) in pairs:
                    i1 = r1 * cols + c1
                    i2 = r2 * cols + c2
                    p1 = tuple(corners[i1].astype(int))
                    p2 = tuple(corners[i2].astype(int))
                    cv2.line(vis, p1, p2, (0, 255, 0), 2)
                    cv2.circle(vis, p1, 5, (0, 0, 255), -1)
                    cv2.circle(vis, p2, 5, (0, 0, 255), -1)
                vis_path = RESULTS_DIR / f"auto_{path.stem}.jpg"
                cv2.imwrite(str(vis_path), vis)
                print(f"  Saved visualization: {vis_path.name}")
                first_vis_saved = True

    if all_errors:
        print()
        print(f"Mean absolute error: {np.mean(all_errors):.2f} mm")
        print(f"Max absolute error:  {np.max(all_errors):.2f} mm")
        print(f"CSV saved to: {csv_path}")
    else:
        print("No successful measurements.")


# ============================================================
# Mode: click
# ============================================================

def run_click_test(
    image_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    calib_size: tuple[int, int],
    known_mm: float | None = None,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read image: {image_path}")
        return
    check_image_size(img, calib_size)

    pose = find_pose(img, K, dist)
    if pose is None:
        print("Chessboard not found. The board must be visible in the image.")
        return
    rvec, tvec, _ = pose

    scale = min(1.0, MAX_WINDOW_WIDTH / img.shape[1])
    display = cv2.resize(img, None, fx=scale, fy=scale)
    window = "Click two points (ESC to quit)"
    points: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x / scale, y / scale))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window, display)

    cv2.imshow(window, display)
    cv2.setMouseCallback(window, on_mouse)

    while len(points) < 2:
        if cv2.waitKey(20) == 27:
            cv2.destroyAllWindows()
            return

    distance_mm = measure_distance_m(points[0], points[1], K, dist, rvec, tvec) * 1000
    print(f"Measured distance: {distance_mm:.2f} mm")
    if known_mm is not None:
        error = distance_mm - known_mm
        print(f"Known distance:    {known_mm:.2f} mm")
        print(f"Error:             {error:+.2f} mm ({error / known_mm * 100:+.2f}%)")

    p1 = (int(points[0][0]), int(points[0][1]))
    p2 = (int(points[1][0]), int(points[1][1]))
    thickness = max(2, img.shape[1] // 400)
    cv2.line(img, p1, p2, (0, 255, 0), thickness)
    cv2.circle(img, p1, thickness * 2, (0, 0, 255), -1)
    cv2.circle(img, p2, thickness * 2, (0, 0, 255), -1)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 15)
    cv2.putText(
        img,
        f"{distance_mm:.1f} mm",
        mid,
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, thickness / 2),
        (255, 0, 0),
        thickness,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"measure_{image_path.stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")

    cv2.imshow(window, cv2.resize(img, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distance measurement on a calibrated photo (plane of the chessboard)"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("auto", help="Measure known chessboard distances vs ground truth")

    click = sub.add_parser("click", help="Click two points on a photo")
    click.add_argument("--image", required=True, help="Path to the photo")
    click.add_argument(
        "--known",
        type=float,
        default=None,
        help="Real distance in mm (optional, to report error)",
    )

    args = parser.parse_args()

    calib_path = resolve_calib_path()
    print(f"Using calibration: {calib_path}")
    K, dist, calib_size = load_calibration(calib_path)

    if args.mode == "auto":
        run_auto_test(K, dist, calib_size)
    else:
        run_click_test(
            Path(args.image),
            K,
            dist,
            calib_size,
            args.known,
        )


if __name__ == "__main__":
    main()