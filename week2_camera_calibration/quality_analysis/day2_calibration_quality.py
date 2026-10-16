import os
import glob
import cv2
import numpy as np
import yaml


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_FOLDER = os.path.join(
    BASE_DIR,
    "calibration_images"
)

TEST_FOLDER = os.path.join(
    BASE_DIR,
    "test_images"
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

# Chessboard inner corners
CHESSBOARD_SIZE = (9, 6)

# Size of one chessboard square in meters
SQUARE_SIZE = 0.025

# Number of worst images to remove
# You can change this to 2, 3, 4 or 5
NUM_BAD_IMAGES_TO_REMOVE = 4


# ============================================================
# STEP 1. CREATE 3D OBJECT POINTS
# ============================================================

objp = np.zeros(
    (
        CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1],
        3
    ),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


# ============================================================
# STEP 2. LOAD CALIBRATION IMAGES
# ============================================================

print("\n========================================")
print("LOADING CALIBRATION IMAGES")
print("========================================")

image_paths = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(
        glob.glob(
            os.path.join(
                CALIBRATION_FOLDER,
                extension
            )
        )
    )

image_paths = list(set(image_paths))
image_paths.sort()

print(f"\nCalibration folder:")
print(CALIBRATION_FOLDER)

print(f"\nFound {len(image_paths)} calibration images.")


if len(image_paths) == 0:
    print("\nERROR: No calibration images were found.")
    print("Please check:")
    print(CALIBRATION_FOLDER)
    exit()


# ============================================================
# STEP 3. FIND CHESSBOARD CORNERS
# ============================================================

print("\n========================================")
print("FINDING CHESSBOARD CORNERS")
print("========================================")


objpoints = []
imgpoints = []
image_names = []

gray = None

successful_images = 0


for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        print(
            f"Could not read: "
            f"{os.path.basename(image_path)}"
        )
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        None
    )

    if found:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
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

        # Save calibration points
        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)

        # Save image name
        image_names.append(
            os.path.basename(image_path)
        )

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

        cv2.imwrite(
            preview_name,
            preview
        )

    else:

        print(
            f"Chessboard NOT found: "
            f"{os.path.basename(image_path)}"
        )


if len(objpoints) == 0:

    print("\nERROR: No chessboard corners were found.")
    print("Check CHESSBOARD_SIZE and your images.")
    exit()


print(
    f"\nSuccessfully processed "
    f"{successful_images} images."
)


# ============================================================
# FUNCTION: CALCULATE REPROJECTION ERRORS
# ============================================================

def calculate_reprojection_errors(
    objpoints,
    imgpoints,
    rvecs,
    tvecs,
    K,
    dist
):

    errors = []

    for i in range(len(objpoints)):

        projected_points, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            K,
            dist
        )

        actual_points = imgpoints[i].reshape(
            -1,
            2
        )

        projected_points = projected_points.reshape(
            -1,
            2
        )

        point_errors = np.linalg.norm(
            actual_points - projected_points,
            axis=1
        )

        image_error = np.mean(
            point_errors
        )

        errors.append(
            image_error
        )

    return errors


# ============================================================
# FUNCTION: CALIBRATE CAMERA
# ============================================================

def calibrate_camera(
    objpoints,
    imgpoints,
    image_size
):

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    errors = calculate_reprojection_errors(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        K,
        dist
    )

    mean_error = np.mean(errors)

    return (
        ret,
        K,
        dist,
        rvecs,
        tvecs,
        errors,
        mean_error
    )


# ============================================================
# STEP 4. INITIAL CALIBRATION
# ============================================================

print("\n========================================")
print("INITIAL CALIBRATION")
print("========================================")


image_size = gray.shape[::-1]


(
    ret_old,
    K_old,
    dist_old,
    rvecs_old,
    tvecs_old,
    errors_old,
    mean_error_old
) = calibrate_camera(
    objpoints,
    imgpoints,
    image_size
)


print("\nCamera matrix K:")
print(K_old)

print("\nDistortion coefficients:")
print(dist_old)

print(
    f"\nRMS calibration error: "
    f"{ret_old:.6f} pixels"
)


# ============================================================
# STEP 5. PRINT ERROR FOR EVERY IMAGE
# ============================================================

print("\n========================================")
print("REPROJECTION ERROR PER IMAGE")
print("========================================")


for name, error in zip(
    image_names,
    errors_old
):

    print(
        f"{name} -> "
        f"{error:.6f} pixels"
    )


print(
    f"\nMean reprojection error: "
    f"{mean_error_old:.6f} pixels"
)


# ============================================================
# STEP 6. FIND WORST IMAGES
# ============================================================

print("\n========================================")
print("PROBLEMATIC IMAGES")
print("========================================")


# Create pairs:
# image name + error
image_error_pairs = list(
    zip(
        image_names,
        errors_old
    )
)


# Sort from highest error to lowest
image_error_pairs.sort(
    key=lambda x: x[1],
    reverse=True
)


print(
    "\nImages sorted from highest "
    "error to lowest:"
)


for name, error in image_error_pairs:

    print(
        f"{name} -> "
        f"{error:.6f} pixels"
    )


# Select the worst images
bad_images = [
    name
    for name, error
    in image_error_pairs[
        :NUM_BAD_IMAGES_TO_REMOVE
    ]
]


print("\n========================================")
print("IMAGES SELECTED FOR REMOVAL")
print("========================================")


for name in bad_images:

    error = dict(
        image_error_pairs
    )[name]

    print(
        f"{name} -> "
        f"{error:.6f} pixels"
    )


print(
    "\nThese images will not be used "
    "for the final calibration."
)


# ============================================================
# STEP 7. REMOVE BAD IMAGES
# ============================================================

filtered_objpoints = []
filtered_imgpoints = []
filtered_names = []


for i, name in enumerate(image_names):

    if name not in bad_images:

        filtered_objpoints.append(
            objpoints[i]
        )

        filtered_imgpoints.append(
            imgpoints[i]
        )

        filtered_names.append(
            name
        )


print("\n========================================")
print("CALIBRATION DATA AFTER FILTERING")
print("========================================")

print(
    f"\nOriginal images: "
    f"{len(objpoints)}"
)

print(
    f"Removed images: "
    f"{len(bad_images)}"
)

print(
    f"Final images: "
    f"{len(filtered_objpoints)}"
)


# ============================================================
# STEP 8. IMPROVED CALIBRATION
# ============================================================

print("\n========================================")
print("IMPROVED CALIBRATION")
print("========================================")


(
    ret_new,
    K_new,
    dist_new,
    rvecs_new,
    tvecs_new,
    errors_new,
    mean_error_new
) = calibrate_camera(
    filtered_objpoints,
    filtered_imgpoints,
    image_size
)


print("\nFinal camera matrix K:")
print(K_new)

print("\nFinal distortion coefficients:")
print(dist_new)

print(
    f"\nFinal RMS calibration error: "
    f"{ret_new:.6f} pixels"
)


# ============================================================
# STEP 9. COMPARE OLD AND NEW CALIBRATION
# ============================================================

print("\n========================================")
print("CALIBRATION COMPARISON")
print("========================================")


print(
    f"\nBefore removing bad images:"
)

print(
    f"Images: {len(objpoints)}"
)

print(
    f"Mean reprojection error: "
    f"{mean_error_old:.6f} pixels"
)


print(
    f"\nAfter removing bad images:"
)

print(
    f"Images: {len(filtered_objpoints)}"
)

print(
    f"Mean reprojection error: "
    f"{mean_error_new:.6f} pixels"
)


difference = (
    mean_error_old
    - mean_error_new
)


print(
    f"\nError difference: "
    f"{difference:.6f} pixels"
)


if mean_error_new < mean_error_old:

    improvement = (
        difference
        / mean_error_old
        * 100
    )

    print(
        f"Improvement: "
        f"{improvement:.2f}%"
    )

else:

    increase = (
        abs(difference)
        / mean_error_old
        * 100
    )

    print(
        f"Error increased by: "
        f"{increase:.2f}%"
    )

    print(
        "\nRemoving these images did not "
        "improve the calibration."
    )


# ============================================================
# STEP 10. PRINT FINAL ERROR PER IMAGE
# ============================================================

print("\n========================================")
print("FINAL ERROR PER IMAGE")
print("========================================")


for name, error in zip(
    filtered_names,
    errors_new
):

    print(
        f"{name} -> "
        f"{error:.6f} pixels"
    )


# ============================================================
# STEP 11. SAVE FINAL CAMERA PARAMETERS
# ============================================================

print("\n========================================")
print("SAVING FINAL CAMERA PARAMETERS")
print("========================================")


data = {
    "K": K_new.tolist(),
    "dist": dist_new.tolist(),
    "image_width": int(image_size[0]),
    "image_height": int(image_size[1]),
    "reprojection_error": float(mean_error_new),
    "num_images": len(filtered_objpoints)
}


with open(
    PARAMS_FILE,
    "w"
) as f:

    yaml.dump(
        data,
        f,
        sort_keys=False
    )


print(
    f"\nFinal camera parameters saved to:"
    f"\n{PARAMS_FILE}"
)


# ============================================================
# STEP 12. DRAW CAMERA AXES
# ============================================================

print("\n========================================")
print("AXES VISUALIZATION")
print("========================================")


# Use the first final calibration image
axes_image_name = filtered_names[0]

axes_image_path = os.path.join(
    CALIBRATION_FOLDER,
    axes_image_name
)

axes_image = cv2.imread(
    axes_image_path
)


if axes_image is not None:

    axes_gray = cv2.cvtColor(
        axes_image,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCorners(
        axes_gray,
        CHESSBOARD_SIZE,
        None
    )

    if found:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv2.cornerSubPix(
            axes_gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        success, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            K_new,
            dist_new
        )

        if success:

            axes_result = axes_image.copy()

            cv2.drawFrameAxes(
                axes_result,
                K_new,
                dist_new,
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
                axes_result
            )

            print(
                f"Axes visualization saved to:"
                f"\n{axes_path}"
            )


# ============================================================
# STEP 13. UNDISTORT TEST IMAGES
# ============================================================

print("\n========================================")
print("UNDISTORTING TEST IMAGES")
print("========================================")


test_images = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png"
]:

    test_images.extend(
        glob.glob(
            os.path.join(
                TEST_FOLDER,
                extension
            )
        )
    )


test_images.sort()


if len(test_images) == 0:

    print(
        "\nNo test images found."
    )

    print(
        "Put 4-6 normal photos into:"
    )

    print(
        TEST_FOLDER
    )

else:

    for i, image_path in enumerate(
        test_images
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        # Calculate optimal camera matrix
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K_new,
            dist_new,
            (width, height),
            1,
            (width, height)
        )

        # Remove distortion
        undistorted = cv2.undistort(
            image,
            K_new,
            dist_new,
            None,
            new_K
        )

        # Crop using ROI
        x, y, w, h = roi

        if w > 0 and h > 0:

            undistorted = undistorted[
                y:y + h,
                x:x + w
            ]

        # Resize original to the same size
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

        # Create comparison
        comparison = np.hstack(
            (
                original_resized,
                undistorted
            )
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
            f"Saved: "
            f"{comparison_path}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("DAY 2 CALIBRATION COMPLETE")
print("========================================")


print(
    f"\nInitial calibration:"
)

print(
    f"Images: {len(objpoints)}"
)

print(
    f"Mean error: "
    f"{mean_error_old:.6f} pixels"
)


print(
    f"\nFinal calibration:"
)

print(
    f"Images: {len(filtered_objpoints)}"
)

print(
    f"Mean error: "
    f"{mean_error_new:.6f} pixels"
)


print(
    f"\nRemoved images: "
    f"{len(bad_images)}"
)


print(
    "\nFinal parameters:"
)

print(
    PARAMS_FILE
)


print(
    "\nResults:"
)

print(
    RESULTS_FOLDER
)

print(
    "\n========================================"
)