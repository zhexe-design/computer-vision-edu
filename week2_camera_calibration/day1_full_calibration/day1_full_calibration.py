import os
import glob
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_FOLDER = os.path.join(
    BASE_DIR,
    "calibration_images"
)

RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

CORNERS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "corners"
)

BEFORE_AFTER_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "before_after"
)

PARAMS_FILE = os.path.join(
    BASE_DIR,
    "camera_params.yaml"
)


# Create result folders
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CORNERS_FOLDER, exist_ok=True)
os.makedirs(BEFORE_AFTER_FOLDER, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Folder where this Python file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Calibration images
CALIBRATION_FOLDER = os.path.join(
    BASE_DIR,
    "calibration_images"
)

# Normal test images
TEST_FOLDER = os.path.join(
    BASE_DIR,
    "test_images"
)

# Results
RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

# Corners results
CORNERS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "corners"
)

# Before / After results
BEFORE_AFTER_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "before_after"
)

# Camera parameters
PARAMS_FILE = os.path.join(
    BASE_DIR,
    "camera_params.yaml"
)


# Chessboard inner corners
CHESSBOARD_SIZE = (9, 6)

# Size of one chessboard square
SQUARE_SIZE = 0.025


# Create result folders
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CORNERS_FOLDER, exist_ok=True)
os.makedirs(BEFORE_AFTER_FOLDER, exist_ok=True)


# ============================================================
# STEP 1. CREATE 3D OBJECT POINTS
# ============================================================

# Create coordinates for the chessboard corners
objp = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

# Convert chessboard coordinates to real-world units
objp *= SQUARE_SIZE


# Lists for 3D points and 2D image points
objpoints = []
imgpoints = []

# ============================================================
# STEP 2. LOAD CALIBRATION IMAGES
# ============================================================

# Get the absolute path of the calibration folder
CALIBRATION_FOLDER = os.path.abspath(CALIBRATION_FOLDER)

print("\nCalibration folder:")
print(CALIBRATION_FOLDER)

# Search for common image formats
image_paths = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(
        glob.glob(
            os.path.join(CALIBRATION_FOLDER, extension)
        )
    )

# Remove duplicate paths
image_paths = list(set(image_paths))

# Sort files for consistent order
image_paths.sort()

print(f"\nFound {len(image_paths)} calibration images.")

if len(image_paths) == 0:
    print("\nERROR: No images were found.")
    print("Please check that your calibration images are located here:")
    print(CALIBRATION_FOLDER)
    exit()

gray = None
successful_images = 0


# ============================================================
# STEP 3. FIND CHESSBOARD CORNERS
# ============================================================

for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        None
    )

    if found:

        # Improve corner accuracy
        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Save object points and image points
        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)

        successful_images += 1

        print(
            f"Chessboard found: "
            f"{os.path.basename(image_path)}"
        )

        # Draw detected corners
        preview = image.copy()

        cv2.drawChessboardCorners(
            preview,
            CHESSBOARD_SIZE,
            corners_refined,
            found
        )

        preview_name = os.path.join(
            CORNERS_FOLDER,
            f"corners_{successful_images}.jpg"
        )

        cv2.imwrite(preview_name, preview)

    else:
        print(
            f"Chessboard NOT found: "
            f"{os.path.basename(image_path)}"
        )


# ============================================================
# CHECK CALIBRATION DATA
# ============================================================

if len(objpoints) == 0:
    print("\nNo chessboard corners were found.")
    print("Check your calibration images and CHESSBOARD_SIZE.")
    exit()


print(
    f"\nSuccessfully processed "
    f"{successful_images} images."
)


# ============================================================
# STEP 4. CAMERA CALIBRATION
# ============================================================

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)


# ============================================================
# STEP 5. PRINT CALIBRATION RESULTS
# ============================================================

print("\n========================================")
print("CALIBRATION RESULTS")
print("========================================")

print("\nCamera matrix K:")
print(K)

print("\nDistortion coefficients:")
print(dist)

print(f"\nRMS calibration error: {ret:.6f} pixels")


# ============================================================
# STEP 6. CALCULATE REPROJECTION ERROR
# ============================================================

total_error = 0.0
per_image_errors = []

for i in range(len(objpoints)):

    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        K,
        dist
    )

    # Convert both arrays to the same shape
    actual_points = imgpoints[i].reshape(-1, 2)
    projected_points = projected_points.reshape(-1, 2)

    # Calculate Euclidean error for every corner
    errors = np.linalg.norm(
        actual_points - projected_points,
        axis=1
    )

    # Mean error for this image
    image_error = np.mean(errors)

    per_image_errors.append(image_error)
    total_error += image_error


mean_error = total_error / len(objpoints)


print("\n========================================")
print("REPROJECTION ERROR")
print("========================================")

for i, error in enumerate(per_image_errors):

    print(
        f"Image {i + 1}: "
        f"{error:.6f} pixels"
    )


print(
    f"\nMean reprojection error: "
    f"{mean_error:.6f} pixels"
)


# ============================================================
# STEP 7. SAVE CAMERA PARAMETERS
# ============================================================

fs = cv2.FileStorage(
    PARAMS_FILE,
    cv2.FILE_STORAGE_WRITE
)

fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)

fs.release()

print(
    f"\nCamera parameters saved to:"
    f"\n{PARAMS_FILE}"
)


# ============================================================
# STEP 8. DRAW CAMERA AXES ON A CHESSBOARD
# ============================================================

# Use the first successful calibration image
first_image = cv2.imread(image_paths[0])

if first_image is not None:

    first_gray = cv2.cvtColor(
        first_image,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCorners(
        first_gray,
        CHESSBOARD_SIZE,
        None
    )

    if found:

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv2.cornerSubPix(
            first_gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Solve camera pose
        success, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            K,
            dist
        )

        if success:

            axes_image = first_image.copy()

            cv2.drawFrameAxes(
                axes_image,
                K,
                dist,
                rvec,
                tvec,
                SQUARE_SIZE * 3
            )

            axes_path = os.path.join(
                RESULTS_FOLDER,
                "axes_visualization.jpg"
            )

            cv2.imwrite(
                axes_path,
                axes_image
            )

            print(
                f"Axes visualization saved to:"
                f"\n{axes_path}"
            )


# ============================================================
# STEP 9. UNDISTORT TEST IMAGES
# ============================================================

test_images = glob.glob(
    os.path.join(TEST_FOLDER, "*.*")
)


if len(test_images) == 0:

    print(
        "\nNo test images found."
        "\nCreate the 'test_images' folder and "
        "put 4-6 normal photos there."
    )

else:

    print("\n========================================")
    print("UNDISTORTING TEST IMAGES")
    print("========================================")

    for i, image_path in enumerate(test_images):

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        # Calculate an optimal new camera matrix
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K,
            dist,
            (width, height),
            1,
            (width, height)
        )

        # Remove lens distortion
        undistorted = cv2.undistort(
            image,
            K,
            dist,
            None,
            new_K
        )

        # Crop the result using ROI
        x, y, w, h = roi

        if w > 0 and h > 0:
            undistorted = undistorted[
                y:y + h,
                x:x + w
            ]

        # Resize original image to the same size
        # as the undistorted image
        original_resized = cv2.resize(
            image,
            (
                undistorted.shape[1],
                undistorted.shape[0]
            )
        )

        # Add labels
        cv2.putText(
            original_resized,
            "Before",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            undistorted,
            "After",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Create side-by-side comparison
        comparison = np.hstack(
            (original_resized, undistorted)
        )

        comparison_path = os.path.join(
            BEFORE_AFTER_FOLDER,
            f"comparison_{i + 1}.jpg"
        )

        cv2.imwrite(
            comparison_path,
            comparison
        )

        print(
            f"Saved: {comparison_path}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("CALIBRATION COMPLETE")
print("========================================")

print(f"Calibration images: {len(image_paths)}")
print(f"Successful images: {successful_images}")
print(f"Mean reprojection error: {mean_error:.6f} px")

print("\nCamera matrix:")
print(K)

print("\nDistortion coefficients:")
print(dist)

print("\nResults saved in:")
print(RESULTS_FOLDER)