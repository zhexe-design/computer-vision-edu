import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MATCHES_DIR = ROOT / "results" / "matches"
EPILINES_DIR = ROOT / "results" / "epilines"
METRICS_DIR = ROOT / "results" / "metrics"

PAIRS = ["A", "B", "C"]

MATCH_TYPE = "sift"

RANSAC_THRESHOLD = 1.0
RANSAC_CONFIDENCE = 0.999

NUM_EPILINES = 25

RANDOM_SEED = 0

for folder in [EPILINES_DIR, METRICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

cv2.setRNGSeed(RANDOM_SEED)


# ============================================================
# EPIPOLAR ERROR
# ============================================================

def epipolar_error(F, pts1, pts2):
    """
    Calculate point-to-epipolar-line distance in pixels.

    Measures the distance from points in image 2
    to their corresponding epipolar lines.
    """

    pts1 = pts1.reshape(-1, 1, 2)
    pts2 = pts2.reshape(-1, 2)

    lines = cv2.computeCorrespondEpilines(
        pts1, 1, F
    ).reshape(-1, 3)

    numerator = np.abs(
        lines[:, 0] * pts2[:, 0]
        + lines[:, 1] * pts2[:, 1]
        + lines[:, 2]
    )

    denominator = np.sqrt(
        lines[:, 0] ** 2 + lines[:, 1] ** 2
    )

    return numerator / np.maximum(denominator, 1e-12)


# ============================================================
# DRAW EPIPOLAR LINES
# ============================================================

def draw_line(img, line, color):

    h, w = img.shape[:2]

    a, b, c = line

    points = []

    if abs(b) > 1e-10:
        points.append((0, int(-c / b)))
        points.append((w, int(-(c + a * w) / b)))

    if abs(a) > 1e-10:
        points.append((int(-c / a), 0))
        points.append((int(-(c + b * h) / a), h))

    valid = [
        p for p in points
        if 0 <= p[0] <= w and 0 <= p[1] <= h
    ]

    if len(valid) >= 2:
        cv2.line(
            img,
            valid[0],
            valid[1],
            color,
            1,
            cv2.LINE_AA
        )


def visualize_epilines(
    img1,
    img2,
    pts1,
    pts2,
    F,
    inliers,
    pair_name
):
    """
    Draw corresponding epipolar lines
    on both images.
    """

    img1 = img1.copy()
    img2 = img2.copy()

    indices = np.where(inliers)[0]

    if len(indices) == 0:
        print(f"[WARNING] {pair_name}: no inliers to visualize.")
        return

    # Select up to NUM_EPILINES evenly distributed inliers.
    selected = np.linspace(
        0,
        len(indices) - 1,
        min(NUM_EPILINES, len(indices)),
        dtype=int
    )

    selected = indices[selected]

    p1 = pts1[selected].reshape(-1, 1, 2)
    p2 = pts2[selected].reshape(-1, 1, 2)

    # Lines in image 2 from points in image 1.
    lines2 = cv2.computeCorrespondEpilines(
        p1, 1, F
    ).reshape(-1, 3)

    # Lines in image 1 from points in image 2.
    lines1 = cv2.computeCorrespondEpilines(
        p2, 2, F
    ).reshape(-1, 3)

    rng = np.random.default_rng(42)

    for i in range(len(selected)):

        color = tuple(
            int(x) for x in rng.integers(50, 256, 3)
        )

        draw_line(img1, lines1[i], color)
        draw_line(img2, lines2[i], color)

        x1, y1 = np.round(pts1[selected[i]]).astype(int)
        x2, y2 = np.round(pts2[selected[i]]).astype(int)

        cv2.circle(img1, (x1, y1), 5, color, -1)
        cv2.circle(img2, (x2, y2), 5, color, -1)

    # Combine images side by side.
    h = max(img1.shape[0], img2.shape[0])

    def resize_height(img, height):
        width = int(img.shape[1] * height / img.shape[0])
        return cv2.resize(img, (width, height))

    img1 = resize_height(img1, h)
    img2 = resize_height(img2, h)

    combined = np.hstack([img1, img2])

    output = EPILINES_DIR / f"{pair_name}.png"

    cv2.imwrite(str(output), combined)

    print(f"[SAVED] {output}")


# ============================================================
# HOMOGRAPHY DIAGNOSTICS
# ============================================================

def analyze_homography(pts1, pts2):

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        3.0
    )

    if H is None or mask is None:
        return 0

    return int(mask.sum())


# ============================================================
# PROCESS ONE PAIR
# ============================================================

def process_pair(pair_name):

    print("\n" + "=" * 60)
    print(f"PROCESSING PAIR {pair_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load feature matches
    # --------------------------------------------------------

    match_file = MATCHES_DIR / f"{pair_name}_{MATCH_TYPE}.npz"

    if not match_file.exists():
        print(f"[ERROR] Missing file: {match_file}")
        return None

    data = np.load(match_file)

    pts1 = data["pts1"].astype(np.float64)
    pts2 = data["pts2"].astype(np.float64)

    print(f"Total matches: {len(pts1)}")

    if len(pts1) < 8:
        print("[ERROR] At least 8 matches are required.")
        return None

    # --------------------------------------------------------
    # Load original images
    # --------------------------------------------------------

    pair_dir = ROOT / "data" / "pairs" / pair_name

    image_files = sorted([
        p for p in pair_dir.iterdir()
        if p.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".bmp"
        ]
    ])

    if len(image_files) != 2:
        print("[ERROR] Expected exactly two images.")
        return None

    img1 = cv2.imread(str(image_files[0]))
    img2 = cv2.imread(str(image_files[1]))

    if img1 is None or img2 is None:
        print("[ERROR] Could not load images.")
        return None

    # --------------------------------------------------------
    # METHOD 1: 8-POINT
    # --------------------------------------------------------

    print("\n[1] Estimating F using 8-point algorithm...")

    F8, _ = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_8POINT
    )

    if F8 is None or F8.shape != (3, 3):
        print("[ERROR] 8-point estimation failed.")
        return None

    error8 = epipolar_error(F8, pts1, pts2)

    print(f"Mean epipolar error: {error8.mean():.4f} px")

    # --------------------------------------------------------
    # METHOD 2: RANSAC
    # --------------------------------------------------------

    print("\n[2] Estimating F using RANSAC...")

    cv2.setRNGSeed(RANDOM_SEED)

    Fr, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        RANSAC_THRESHOLD,
        RANSAC_CONFIDENCE
    )

    if Fr is None or mask is None or Fr.shape != (3, 3):
        print("[ERROR] RANSAC estimation failed.")
        return None

    inliers = mask.ravel().astype(bool)

    num_inliers = int(inliers.sum())

    if num_inliers == 0:
        print("[ERROR] RANSAC found no inliers.")
        return None

    error_ransac = epipolar_error(
        Fr,
        pts1[inliers],
        pts2[inliers]
    )

    print(f"Inliers: {num_inliers}/{len(pts1)}")
    print(f"Inlier ratio: {num_inliers / len(pts1):.2%}")
    print(f"Mean epipolar error: {error_ransac.mean():.4f} px")

    # --------------------------------------------------------
    # HOMOGRAPHY DIAGNOSTICS
    # --------------------------------------------------------

    homography_inliers = analyze_homography(pts1, pts2)

    print("\n[3] Homography diagnostics")

    print(f"Homography inliers: {homography_inliers}/{len(pts1)}")
    print(f"Fundamental matrix inliers: {num_inliers}/{len(pts1)}")

    if homography_inliers / len(pts1) > 0.85:
        print("[WARNING] Homography dominates.")
        print("Possible planar scene or low parallax.")

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    print("\n[4] Drawing epipolar lines...")

    visualize_epilines(
        img1,
        img2,
        pts1,
        pts2,
        Fr,
        inliers,
        pair_name
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    np.savez(
        METRICS_DIR / f"{pair_name}_F.npz",
        F=Fr,
        F8=F8,
        inliers=inliers,
        pts1=pts1,
        pts2=pts2,
        error8=error8,
        error_ransac=error_ransac
    )

    result = {
        "Pair": pair_name,
        "Total matches": len(pts1),
        "8-point inliers": len(pts1),
        "8-point error (px)": round(float(error8.mean()), 4),
        "RANSAC inliers": num_inliers,
        "RANSAC ratio (%)": round(
            num_inliers / len(pts1) * 100, 2
        ),
        "RANSAC error (px)": round(
            float(error_ransac.mean()), 4
        ),
        "Homography inliers": homography_inliers
    }

    print("\n[RESULT]")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    for pair in PAIRS:

        result = process_pair(pair)

        if result is not None:
            results.append(result)

    if not results:
        print("\n[ERROR] No pairs were processed.")
        return

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_file = METRICS_DIR / "fundamental_comparison.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("DAY 2 COMPLETED")
    print("=" * 60)

    print(f"\nMetrics saved to: {csv_file}")
    print(f"Epipolar lines: {EPILINES_DIR}")
    print(f"F matrices: {METRICS_DIR}")


if __name__ == "__main__":
    main()