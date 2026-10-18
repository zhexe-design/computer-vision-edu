"""
Day 3 — Apply Camera Calibration


Outputs:
    results/undistorted_*.jpg
    results/comparison_*.jpg

This script does NOT recalibrate the camera, rescale the camera matrix
automatically, or crop the output. It preserves the original photo dimensions.
"""

from pathlib import Path
import cv2
import numpy as np
import yaml


BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test_images"
RESULTS_DIR = BASE_DIR / "results"

# Default location when both projects are inside week2_camera_calibration/.
PARAMS_FILE = (
    BASE_DIR.parent
    / "quality_analysis"
    / "results"
    / "camera_params.yaml"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(folder):
    """Return supported image paths in deterministic order."""
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_parameters(path):
    """Load K, distortion coefficients, and calibration image dimensions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Camera parameter file was not found: {path}\n"
            "Run Day 2 first, or update PARAMS_FILE in this script."
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict) or "K" not in data or "dist" not in data:
        raise ValueError(
            f"{path} must contain 'K' and 'dist' entries."
        )

    camera_matrix = np.asarray(data["K"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist"], dtype=np.float64).reshape(-1, 1)

    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 camera matrix K, got {camera_matrix.shape}.")

    calibration_width = data.get("image_width")
    calibration_height = data.get("image_height")
    calibration_size = None
    if calibration_width is not None and calibration_height is not None:
        calibration_size = (int(calibration_width), int(calibration_height))

    return camera_matrix, dist_coeffs, calibration_size


def add_label(image, text):
    """Return a labelled copy of an image."""
    output = image.copy()
    cv2.rectangle(output, (10, 10), (190, 52), (0, 0, 0), thickness=-1)
    cv2.putText(
        output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    return output


def create_comparison(original, undistorted):
    """Create side-by-side comparison without cropping or resizing either image."""
    original_labelled = add_label(original, "Original")
    undistorted_labelled = add_label(undistorted, "Undistorted")
    return np.hstack((original_labelled, undistorted_labelled))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(TEST_DIR)
    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in {TEST_DIR}. "
            "Add ordinary photos (not calibration chessboard images) and run again."
        )

    camera_matrix, dist_coeffs, calibration_size = load_parameters(PARAMS_FILE)

    print("=" * 64)
    print("DAY 3 — APPLY CAMERA CALIBRATION")
    print("=" * 64)
    print(f"Parameters: {PARAMS_FILE}")
    if calibration_size:
        print(
            "Calibration resolution: "
            f"{calibration_size[0]} x {calibration_size[1]}"
        )
    else:
        print("Calibration resolution: not recorded in the parameter file")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Could not read {image_path.name}; skipping.")
            continue

        height, width = image.shape[:2]
        photo_size = (width, height)
        print(f"\n{image_path.name}: photo resolution = {width} x {height}")

        if calibration_size and photo_size != calibration_size:
            print(
                "[WARNING] Photo resolution differs from calibration resolution. "
                "The script will NOT scale K automatically. For reliable results, "
                "use photos at the calibration resolution or deliberately adapt "
                "the camera matrix after verifying the camera/image pipeline."
            )

        # Keep the original camera matrix, output dimensions, and full image.
        # No automatic new camera matrix and no ROI crop are applied.
        undistorted = cv2.undistort(
            image,
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
        )

        undistorted_path = RESULTS_DIR / f"undistorted_{index:02d}_{image_path.stem}.jpg"
        comparison_path = RESULTS_DIR / f"comparison_{index:02d}_{image_path.stem}.jpg"

        if not cv2.imwrite(str(undistorted_path), undistorted):
            print(f"[WARNING] Failed to save: {undistorted_path}")
            continue

        comparison = create_comparison(image, undistorted)
        if not cv2.imwrite(str(comparison_path), comparison):
            print(f"[WARNING] Failed to save: {comparison_path}")
            continue

        print(f"Saved undistorted image: {undistorted_path}")
        print(f"Saved comparison:        {comparison_path}")

    print("\nFinished. Results are saved in:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
