import cv2
import numpy as np
import glob
import os


# ==========================================
# SETTINGS
# ==========================================

CHESSBOARD_SIZE = (9, 6)

# Size of one square in meters
SQUARE_SIZE = 0.025

# Folders
CALIBRATION_FOLDER = r"Y:\1\week1_camera_models\week1_camera_models\day5_calibration\calibration_images"
RESULTS_FOLDER = r"Y:\1\week1_camera_models\week1_camera_models\day5_calibration\results"

# Criteria for refining corner positions
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ==========================================
# CREATE RESULTS FOLDER
# ==========================================

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ==========================================
# PREPARING 3D POINTS
# ==========================================

objp = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


# Points will be stored here
objpoints = []  # 3D points
imgpoints = []  # 2D points


print("Program started.")
print(f"Chessboard size: {CHESSBOARD_SIZE}")
print(f"Square size: {SQUARE_SIZE} m")
print()


# ==========================================
# SEARCHING FOR CALIBRATION IMAGES
# ==========================================

images = (
    glob.glob(os.path.join(CALIBRATION_FOLDER, "*.jpg"))
    + glob.glob(os.path.join(CALIBRATION_FOLDER, "*.jpeg"))
    + glob.glob(os.path.join(CALIBRATION_FOLDER, "*.png"))
)

images.sort()


if len(images) == 0:
    print("No calibration images found.")
    print(f"Please put chessboard images into '{CALIBRATION_FOLDER}/'")
    exit()


print(f"Found calibration images: {len(images)}")
print()


# ==========================================
# PROCESSING CALIBRATION IMAGES
# ==========================================

image_size = None

for fname in images:

    print(f"Processing: {fname}")

    img = cv2.imread(fname)

    if img is None:
        print("  Error loading image.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save image size
    if image_size is None:
        image_size = gray.shape[::-1]

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        None
    )

    if ret:
        print("  Corners found!")

        # Refine corner positions
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Save points
        objpoints.append(objp.copy())
        imgpoints.append(corners2)

        # Draw detected corners
        cv2.drawChessboardCorners(
            img,
            CHESSBOARD_SIZE,
            corners2,
            ret
        )

        # Save image with detected corners
        filename = os.path.basename(fname)
        output_path = os.path.join(
            RESULTS_FOLDER,
            "corners_" + filename
        )

        cv2.imwrite(output_path, img)

    else:
        print("  Corners NOT found.")

    print()


# ==========================================
# CHECK DETECTION RESULT
# ==========================================

print("================================")
print("Corner Detection Result")
print("================================")

print(f"Total images: {len(images)}")
print(f"Successfully detected boards: {len(objpoints)}")
print()


if len(objpoints) < 3:
    print("Not enough calibration images.")
    print("Please add more chessboard images.")
    exit()


# ==========================================
# CAMERA CALIBRATION
# ==========================================

print("Starting camera calibration...")
print()

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)


# ==========================================
# PRINT CALIBRATION RESULTS
# ==========================================

print("================================")
print("Calibration Result")
print("================================")

print(f"Calibration RMS error: {ret}")

print()
print("Camera matrix K:")
print(K)

print()
print("Distortion coefficients:")
print(dist)


# ==========================================
# CALCULATE REPROJECTION ERROR
# ==========================================

total_error = 0
per_image_errors = []

print()
print("================================")
print("Reprojection Errors")
print("================================")

for i in range(len(objpoints)):

    # Project 3D points back to the image
    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        K,
        dist
    )

    # Convert both point arrays to the same shape
    original_points = imgpoints[i].reshape(-1, 2)
    projected_points = projected_points.reshape(-1, 2)

    # Calculate error for this image
    error = cv2.norm(
        original_points,
        projected_points,
        cv2.NORM_L2
    ) / len(projected_points)

    per_image_errors.append(error)
    total_error += error

    print(f"Image {i + 1}: {error:.6f} pixels")


# Calculate mean error
mean_error = total_error / len(objpoints)

print()
print(f"Mean reprojection error: {mean_error:.6f} pixels")


# ==========================================
# SAVE CALIBRATION PARAMETERS
# ==========================================

calibration_file = os.path.join(
    RESULTS_FOLDER,
    "camera_calibration.npz"
)

np.savez(
    calibration_file,
    K=K,
    dist=dist,
    image_width=image_size[0],
    image_height=image_size[1]
)

print()
print(f"Calibration parameters saved to:")
print(calibration_file)


# ==========================================
# DRAW COORDINATE AXES ON ONE CALIBRATION IMAGE
# ==========================================

axis_image = cv2.imread(images[0])

if axis_image is not None:

    # Detect corners again
    gray = cv2.cvtColor(axis_image, cv2.COLOR_BGR2GRAY)

    ret_axes, corners_axes = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        None
    )

    if ret_axes:

        corners_axes = cv2.cornerSubPix(
            gray,
            corners_axes,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Draw coordinate axes
        cv2.drawFrameAxes(
            axis_image,
            K,
            dist,
            rvecs[0],
            tvecs[0],
            0.1
        )

        axes_path = os.path.join(
            RESULTS_FOLDER,
            "axes_visualization.jpg"
        )

        cv2.imwrite(
            axes_path,
            axis_image
        )

        print(f"Axes visualization saved to:")
        print(axes_path)


# ==========================================
# UNDISTORT CALIBRATION IMAGES
# ==========================================
# These images are only used as an example.
# You can also put normal photos into the
# "results" folder and undistort them using
# the saved calibration parameters.


print()
print("================================")
print("Creating Undistorted Images")
print("================================")


# Use first few calibration images for demonstration
# of the undistortion process.

for i, fname in enumerate(images[:3]):

    img = cv2.imread(fname)

    if img is None:
        continue

    # Undistort image
    undistorted = cv2.undistort(
        img,
        K,
        dist,
        None,
        K
    )

    # Create comparison image:
    # Original | Undistorted

    comparison = np.hstack(
        (img, undistorted)
    )

    comparison_path = os.path.join(
        RESULTS_FOLDER,
        f"comparison_{i + 1}.jpg"
    )

    cv2.imwrite(
        comparison_path,
        comparison
    )

    print(f"Saved: {comparison_path}")

# ==========================================
# FINAL CALIBRATION PARAMETERS
# ==========================================

print()
print("================================")
print("FINAL CALIBRATION PARAMETERS")
print("================================")

# 1. Mean reprojection error
print()
print("1. Mean reprojection error:")
print(f"{mean_error:.6f} pixels")

# 2. Camera matrix K
print()
print("2. Camera matrix K:")
print(K)

# 3. Distortion coefficients
print()
print("3. Distortion coefficients:")
print(dist)

print()
print("================================")
print("End of program")
print("================================")

# ==========================================
# FINAL RESULT
# ==========================================

print()
print("================================")
print("Finished")
print("================================")

print(f"Mean reprojection error: {mean_error:.6f} pixels")
print()
print("Results folder contains:")
print("- camera_calibration.npz")
print("- images with detected corners")
print("- axes_visualization.jpg")
print("- comparison images")