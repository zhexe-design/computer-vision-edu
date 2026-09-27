"""
Day 6 - Distance measurement on a calibrated photo.

Two tests:
  1. auto  - measure known distances between chessboard corners and compare
             them with the real values (sanity check of the calibration).
  2. click - click two points on a photo and get the real distance in mm.
             Optionally compare it with a known real size (--known).

IMPORTANT: the measurement works only for points that lie in the SAME PLANE
as the chessboard (for example, an object lying flat on the same table).

Usage:
    python measure_distance.py auto
    python measure_distance.py click --image test_images/card.jpg --known 85.6
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

# ---------------- Settings ----------------
CHESSBOARD_SIZE = (9, 6)      # inner corners (columns, rows)
SQUARE_SIZE = 0.025           # meters (2.5 cm)
CALIB_FILE = "camera_calibration.npz"
IMAGES_FOLDER = "test_images"
RESULTS_FOLDER = "results"
MAX_WINDOW_WIDTH = 1200       # only for showing the image on the screen

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ---------------- Helpers ----------------
def load_calibration(path):
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    size = (int(data["image_width"]), int(data["image_height"]))
    return K, dist, size


def make_object_points():
    """3D points of the chessboard corners (Z = 0), in meters."""
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    return objp * SQUARE_SIZE


def find_pose(img, K, dist):
    """Find the chessboard and estimate the camera pose relative to it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

    objp = make_object_points()
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        return None
    return rvec, tvec, corners


def pixel_to_plane(pixel, K, dist, rvec, tvec):
    """
    Convert one image pixel to a 3D point on the chessboard plane (Z = 0).
    Returns the point in chessboard coordinates, in meters.
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    # 1. remove distortion and get the ray direction in camera coordinates
    pt = np.array([[pixel]], dtype=np.float32)          # shape (1, 1, 2)
    x, y = cv2.undistortPoints(pt, K, dist).reshape(2)
    ray = np.array([x, y, 1.0])

    # 2. intersect the ray with the chessboard plane
    normal = R[:, 2]                                    # plane normal in camera coords
    s = normal.dot(t) / normal.dot(ray)
    point_cam = s * ray

    # 3. camera coordinates -> chessboard coordinates
    return R.T @ (point_cam - t)


def measure(p1, p2, K, dist, rvec, tvec):
    """Real distance between two pixels (meters)."""
    a = pixel_to_plane(p1, K, dist, rvec, tvec)
    b = pixel_to_plane(p2, K, dist, rvec, tvec)
    return float(np.linalg.norm(a - b))


def check_image_size(img, calib_size):
    h, w = img.shape[:2]
    if (w, h) != calib_size:
        print(f"WARNING: image size is {w}x{h}, but calibration was made for "
              f"{calib_size[0]}x{calib_size[1]}. The result will be wrong.")


# ---------------- Test 1: automatic ----------------
def run_auto_test(K, dist, calib_size):
    images = sorted(
        glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg"))
        + glob.glob(os.path.join(IMAGES_FOLDER, "*.jpeg"))
        + glob.glob(os.path.join(IMAGES_FOLDER, "*.png"))
    )
    if not images:
        print(f"No images found in {IMAGES_FOLDER}/")
        return

    cols, rows = CHESSBOARD_SIZE
    # pairs of corners: ((col1, row1), (col2, row2))
    pairs = [
        ((0, 0), (1, 0)),
        ((0, 0), (4, 0)),
        ((0, 0), (cols - 1, 0)),
        ((0, 0), (0, rows - 1)),
        ((0, 0), (cols - 1, rows - 1)),
    ]

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    csv_path = os.path.join(RESULTS_FOLDER, "auto_test.csv")
    all_errors = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "pair", "true_mm", "measured_mm", "error_mm", "error_percent"])

        for path in images:
            img = cv2.imread(path)
            if img is None:
                continue
            check_image_size(img, calib_size)
            pose = find_pose(img, K, dist)
            if pose is None:
                print(f"{os.path.basename(path)}: chessboard not found, skipped")
                continue
            rvec, tvec, corners = pose
            corners = corners.reshape(-1, 2)

            print(f"\n{os.path.basename(path)}")
            for (c1, r1), (c2, r2) in pairs:
                i1 = r1 * cols + c1
                i2 = r2 * cols + c2
                true_mm = SQUARE_SIZE * np.hypot(c2 - c1, r2 - r1) * 1000
                measured_mm = measure(corners[i1], corners[i2], K, dist, rvec, tvec) * 1000
                error_mm = measured_mm - true_mm
                error_pct = error_mm / true_mm * 100
                all_errors.append(abs(error_mm))

                name = f"({c1},{r1})-({c2},{r2})"
                print(f"  {name:>12}: true {true_mm:7.2f} mm | "
                      f"measured {measured_mm:7.2f} mm | error {error_mm:+.2f} mm ({error_pct:+.2f}%)")
                writer.writerow([os.path.basename(path), name, f"{true_mm:.2f}",
                                 f"{measured_mm:.2f}", f"{error_mm:.2f}", f"{error_pct:.2f}"])

    if all_errors:
        print(f"\nMean absolute error: {np.mean(all_errors):.2f} mm")
        print(f"Max absolute error:  {np.max(all_errors):.2f} mm")
        print(f"Table saved to: {csv_path}")


# ---------------- Test 2: click two points ----------------
def run_click_test(image_path, K, dist, calib_size, known_mm=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return
    check_image_size(img, calib_size)

    pose = find_pose(img, K, dist)
    if pose is None:
        print("Chessboard not found in this image. The board must be visible.")
        return
    rvec, tvec, _ = pose

    scale = min(1.0, MAX_WINDOW_WIDTH / img.shape[1])
    display = cv2.resize(img, None, fx=scale, fy=scale)
    window = "Click two points (ESC to quit)"
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x / scale, y / scale))     # back to original pixels
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window, display)

    cv2.imshow(window, display)
    cv2.setMouseCallback(window, on_mouse)

    while len(points) < 2:
        if cv2.waitKey(20) == 27:
            cv2.destroyAllWindows()
            return

    distance_mm = measure(points[0], points[1], K, dist, rvec, tvec) * 1000
    print(f"Measured distance: {distance_mm:.2f} mm")
    if known_mm:
        error = distance_mm - known_mm
        print(f"Known distance:    {known_mm:.2f} mm")
        print(f"Error:             {error:+.2f} mm ({error / known_mm * 100:+.2f}%)")

    # draw the result on the full-size image and save it
    p1 = tuple(int(v) for v in points[0])
    p2 = tuple(int(v) for v in points[1])
    thickness = max(2, img.shape[1] // 400)
    cv2.line(img, p1, p2, (0, 255, 0), thickness)
    cv2.circle(img, p1, thickness * 2, (0, 0, 255), -1)
    cv2.circle(img, p2, thickness * 2, (0, 0, 255), -1)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 15)
    cv2.putText(img, f"{distance_mm:.1f} mm", mid, cv2.FONT_HERSHEY_SIMPLEX,
                thickness / 2, (255, 0, 0), thickness)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(RESULTS_FOLDER, f"measure_{name}.jpg")
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")

    cv2.imshow(window, cv2.resize(img, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Distance measurement on a calibrated photo")
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("auto", help="measure known chessboard distances")

    click = sub.add_parser("click", help="click two points on a photo")
    click.add_argument("--image", required=True, help="path to the photo")
    click.add_argument("--known", type=float, default=None,
                       help="real distance in mm (optional, to calculate the error)")

    args = parser.parse_args()

    if not os.path.exists(CALIB_FILE):
        print(f"Calibration file not found: {CALIB_FILE}")
        return
    K, dist, calib_size = load_calibration(CALIB_FILE)

    if args.mode == "auto":
        run_auto_test(K, dist, calib_size)
    else:
        run_click_test(args.image, K, dist, calib_size, args.known)


if __name__ == "__main__":
    main()