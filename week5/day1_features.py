"""
Week 5 — Day 1: Feature Detection and Matching
===============================================

Pipeline:
1. Load camera calibration from calib.yaml
2. Load image pairs A, B, C
3. Undistort both images
4. Detect keypoints using SIFT and ORB
5. Match descriptors using Lowe's ratio test
6. Visualize feature matches
7. Save matched points as .npz
8. Generate comparison metrics as CSV

Author: Computer Vision Learning Project
"""

import cv2
import numpy as np
import yaml
import time
import csv

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PAIRS_DIR = DATA_DIR / "pairs"
CALIB_PATH = DATA_DIR / "calib.yaml"

RESULTS_DIR = ROOT / "results"

MATCHES_DIR = RESULTS_DIR / "matches"
METRICS_DIR = RESULTS_DIR / "metrics"
UNDISTORTED_DIR = RESULTS_DIR / "undistorted"

PAIR_NAMES = ["A", "B", "C"]

RATIO_THRESHOLD = 0.75
MAX_FEATURES = 5000

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    directories = [
        MATCHES_DIR,
        METRICS_DIR,
        UNDISTORTED_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# CAMERA CALIBRATION
# ============================================================

def load_calibration(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}"
        )

    with open(path, "r") as file:
        calibration = yaml.safe_load(file)

    K = np.array(
        calibration["K"],
        dtype=np.float64
    )

    dist = np.array(
        calibration["dist"],
        dtype=np.float64
    )

    width = int(calibration["image_width"])
    height = int(calibration["image_height"])

    return K, dist, width, height


def scale_camera_matrix(K, original_size, new_size):

    original_width, original_height = original_size
    new_width, new_height = new_size

    sx = new_width / original_width
    sy = new_height / original_height

    K_scaled = K.copy()

    K_scaled[0, :] *= sx
    K_scaled[1, :] *= sy

    return K_scaled


def undistort_image(image, K, dist, calibration_size):

    height, width = image.shape[:2]

    K_scaled = scale_camera_matrix(
        K,
        calibration_size,
        (width, height)
    )

    undistorted = cv2.undistort(
        image,
        K_scaled,
        dist
    )

    return undistorted


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_pair(pair_name):

    pair_dir = PAIRS_DIR / pair_name

    if not pair_dir.exists():
        raise FileNotFoundError(
            f"Pair directory not found: {pair_dir}"
        )

    images = sorted([
        path for path in pair_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if len(images) != 2:
        raise ValueError(
            f"Pair {pair_name} must contain exactly "
            f"2 images. Found: {len(images)}"
        )

    img1 = cv2.imread(str(images[0]))
    img2 = cv2.imread(str(images[1]))

    if img1 is None or img2 is None:
        raise ValueError(
            f"Could not read images in pair {pair_name}"
        )

    print(f"\nPair {pair_name}:")
    print(f"  Image 1: {images[0].name}")
    print(f"  Image 2: {images[1].name}")

    return img1, img2


# ============================================================
# FEATURE DETECTION AND MATCHING
# ============================================================

def detect_and_match(
    img1,
    img2,
    method="sift",
    ratio=RATIO_THRESHOLD
):

    if method == "sift":

        detector = cv2.SIFT_create(
            nfeatures=MAX_FEATURES
        )

        norm = cv2.NORM_L2

    elif method == "orb":

        detector = cv2.ORB_create(
            nfeatures=MAX_FEATURES
        )

        norm = cv2.NORM_HAMMING

    else:
        raise ValueError(
            f"Unknown detector: {method}"
        )

    start_time = time.perf_counter()

    # Detect keypoints and descriptors

    kp1, des1 = detector.detectAndCompute(
        img1, None
    )

    kp2, des2 = detector.detectAndCompute(
        img2, None
    )

    # Handle images without detectable features

    if (
        des1 is None
        or des2 is None
        or len(des1) < 2
        or len(des2) < 2
    ):

        elapsed = time.perf_counter() - start_time

        return {
            "kp1": kp1,
            "kp2": kp2,
            "good": [],
            "pts1": np.empty((0, 2), dtype=np.float32),
            "pts2": np.empty((0, 2), dtype=np.float32),
            "time": elapsed
        }

    # Brute Force Matcher

    matcher = cv2.BFMatcher(
        norm,
        crossCheck=False
    )

    # K-nearest neighbors

    knn_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    # Lowe's ratio test

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    # Extract corresponding coordinates

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 2)

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 2)

    elapsed = time.perf_counter() - start_time

    return {
        "kp1": kp1,
        "kp2": kp2,
        "good": good_matches,
        "pts1": pts1,
        "pts2": pts2,
        "time": elapsed
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_match_visualization(
    img1,
    img2,
    result,
    pair_name,
    method
):

    kp1 = result["kp1"]
    kp2 = result["kp2"]
    matches = result["good"]

    # Display up to 200 matches

    visualization = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:200],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    output_path = (
        MATCHES_DIR /
        f"{pair_name}_{method}.png"
    )

    cv2.imwrite(
        str(output_path),
        visualization
    )

    print(f"  Saved visualization: {output_path.name}")


# ============================================================
# SAVE MATCHED POINTS
# ============================================================

def save_matched_points(
    result,
    pair_name,
    method
):

    output_path = (
        MATCHES_DIR /
        f"{pair_name}_{method}.npz"
    )

    np.savez_compressed(
        output_path,

        pts1=result["pts1"],
        pts2=result["pts2"],

        num_keypoints1=len(result["kp1"]),
        num_keypoints2=len(result["kp2"]),

        num_matches=len(result["good"]),

        detector=method,

        ratio=RATIO_THRESHOLD
    )

    print(f"  Saved points: {output_path.name}")


# ============================================================
# SAVE UNDISTORTED IMAGES
# ============================================================

def save_undistorted_images(
    img1,
    img2,
    pair_name
):

    cv2.imwrite(
        str(UNDISTORTED_DIR / f"{pair_name}_1.png"),
        img1
    )

    cv2.imwrite(
        str(UNDISTORTED_DIR / f"{pair_name}_2.png"),
        img2
    )


# ============================================================
# METRICS
# ============================================================

def save_metrics(metrics):

    output_path = (
        METRICS_DIR / "day1_comparison.csv"
    )

    fieldnames = [
        "Pair",
        "Detector",
        "Keypoints_1",
        "Keypoints_2",
        "Good_Matches",
        "Time_seconds"
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(metrics)

    print(f"\nMetrics saved: {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("WEEK 5 — DAY 1: FEATURE MATCHING")
    print("=" * 60)

    create_directories()

    # Load calibration

    K, dist, calib_width, calib_height = (
        load_calibration(CALIB_PATH)
    )

    calibration_size = (
        calib_width,
        calib_height
    )

    print("\nCamera calibration loaded.")
    print(f"Calibration resolution: {calib_width}x{calib_height}")

    metrics = []

    # Process all image pairs

    for pair_name in PAIR_NAMES:

        print("\n" + "=" * 60)
        print(f"PROCESSING PAIR {pair_name}")
        print("=" * 60)

        img1, img2 = load_image_pair(pair_name)

        # Undistortion

        img1 = undistort_image(
            img1,
            K,
            dist,
            calibration_size
        )

        img2 = undistort_image(
            img2,
            K,
            dist,
            calibration_size
        )

        save_undistorted_images(
            img1,
            img2,
            pair_name
        )

        # Convert to grayscale

        gray1 = cv2.cvtColor(
            img1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            img2,
            cv2.COLOR_BGR2GRAY
        )

        # Compare SIFT and ORB

        for method in ["sift", "orb"]:

            print(f"\n  Detector: {method.upper()}")

            result = detect_and_match(
                gray1,
                gray2,
                method=method
            )

            kp1 = result["kp1"]
            kp2 = result["kp2"]

            matches = result["good"]

            print(f"  Keypoints image 1: {len(kp1)}")
            print(f"  Keypoints image 2: {len(kp2)}")
            print(f"  Good matches: {len(matches)}")
            print(f"  Processing time: {result['time']:.4f} s")

            # Save visualization

            save_match_visualization(
                img1,
                img2,
                result,
                pair_name,
                method
            )

            # Save corresponding points

            save_matched_points(
                result,
                pair_name,
                method
            )

            # Collect metrics

            metrics.append({
                "Pair": pair_name,
                "Detector": method.upper(),

                "Keypoints_1": len(kp1),
                "Keypoints_2": len(kp2),

                "Good_Matches": len(matches),

                "Time_seconds": round(
                    result["time"], 4
                )
            })

    # Save comparison table

    save_metrics(metrics)

    # Print final summary

    print("\n" + "=" * 60)
    print("DAY 1 COMPLETED")
    print("=" * 60)

    print("\nFINAL COMPARISON:")

    print(
        f"{'Pair':<10}"
        f"{'Method':<10}"
        f"{'Keypoints':<15}"
        f"{'Matches':<12}"
        f"{'Time (s)':<10}"
    )

    for row in metrics:

        keypoints = (
            f"{row['Keypoints_1']}/"
            f"{row['Keypoints_2']}"
        )

        print(
            f"{row['Pair']:<10}"
            f"{row['Detector']:<10}"
            f"{keypoints:<15}"
            f"{row['Good_Matches']:<12}"
            f"{row['Time_seconds']:<10}"
        )

    print("\nAll results saved successfully.")


if __name__ == "__main__":
    main()