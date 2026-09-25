import cv2
import numpy as np
import glob
import os

# ==========================================
# SETTINGS
# ==========================================

CHESSBOARD_SIZE = (9, 6)

# Size of one square in meters
# This value is not very important for now
SQUARE_SIZE = 0.025

# Criteria for refining corner positions
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

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

# ==========================================
# SEARCHING FOR IMAGES
# ==========================================

images = (
    glob.glob("calibration_images/*.jpg")
    + glob.glob("calibration_images/*.png")
)

# ==========================================
# IF THERE ARE NO IMAGES YET
# ==========================================

if len(images) == 0:
    print()
    print("No images found.")
    print("The calibration_images/ folder is empty.")
    print()
    print("3D points preparation completed successfully.")
    print(f"Number of points on one board: {len(objp)}")

else:
    # ==========================================
    # PROCESSING IMAGES
    # ==========================================

    print(f"Found images: {len(images)}")
    print()

    for fname in images:

        print(f"Processing: {fname}")

        img = cv2.imread(fname)

        if img is None:
            print("  Error loading image")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Finding chessboard corners
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
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw detected corners
            cv2.drawChessboardCorners(
                img,
                CHESSBOARD_SIZE,
                corners2,
                ret
            )

            # Display image
            cv2.imshow("Corners", img)
            cv2.waitKey(500)

        else:
            print("  Corners NOT found.")

    cv2.destroyAllWindows()

    # ==========================================
    # RESULT
    # ==========================================

    print()
    print("================================")
    print("Result")
    print("================================")

    print(f"Total images: {len(images)}")
    print(f"Successfully detected boards: {len(objpoints)}")

    if len(objpoints) > 0:
        print("You can proceed to camera calibration.")
    else:
        print("No chessboard was found in any image.")