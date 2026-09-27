import cv2
import numpy as np
import os
import glob

# ==========================================
# FOLDER WITH PHOTOS
# ==========================================
input_folder = r"Y:\1\week1_camera_models\day3_distortion**\i**mages"
# Folder where the results will be saved
output_folder = os.path.join(
    r"Y:\1\week1_camera_models\day3_distortion",
    "results"
)
os.makedirs(output_folder, exist_ok=True)

# ==========================================
# FIND ALL PHOTOS
# ==========================================
extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]
image_paths = []
for extension in extensions:
    image_paths.extend(
        glob.glob(os.path.join(input_folder, extension))
    )

if len(image_paths) == 0:
    print("❌ No photos found!")
    print(f"Check the folder:")
    print(input_folder)
    exit()

print(f"Photos found: {len(image_paths)}")
print()

# ==========================================
# PROCESS EACH PHOTO
# ==========================================
for image_path in image_paths:
    # --------------------------------------
    # Load the image
    # --------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to open: {image_path}")
        continue
    h, w = img.shape[:2]
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    print(f"Processing: {filename}")
    print(f"Size: {w} x {h}")

    # ======================================
    # CREATE CAMERA MATRIX
    # ======================================
    fx = 800.0
    fy = 800.0
    cx = w / 2
    cy = h / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    # ======================================
    # DISTORTION COEFFICIENTS
    # ======================================
    # k1, k2, p1, p2, k3
    dist_coeffs = np.array(
        [-0.3, 0.1, 0.0, 0.0, 0.0],
        dtype=np.float32
    )

    # ======================================
    # CREATE ARTIFICIAL DISTORTION
    # ======================================
    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        dist_coeffs,
        None,
        K,
        (w, h),
        cv2.CV_32FC1
    )
    distorted = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )

    # ======================================
    # CORRECT THE DISTORTION
    # ======================================
    undistorted = cv2.undistort(
        distorted,
        K,
        dist_coeffs
    )

    # ======================================
    # CREATE IMAGE FOR COMPARISON
    # ======================================
    original_labeled = img.copy()
    distorted_labeled = distorted.copy()
    undistorted_labeled = undistorted.copy()

    # Labels
    cv2.putText(
        original_labeled,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        distorted_labeled,
        "Distorted",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        undistorted_labeled,
        "Undistorted",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Combine three images
    combined = np.hstack([
        original_labeled,
        distorted_labeled,
        undistorted_labeled
    ])

    # ======================================
    # SAVE RESULTS
    # ======================================
    cv2.imwrite(
        os.path.join(
            output_folder,
            f"{name}_original.jpg"
        ),
        img
    )
    cv2.imwrite(
        os.path.join(
            output_folder,
            f"{name}_distorted.jpg"
        ),
        distorted
    )
    cv2.imwrite(
        os.path.join(
            output_folder,
            f"{name}_undistorted.jpg"
        ),
        undistorted
    )
    cv2.imwrite(
        os.path.join(
            output_folder,
            f"{name}_comparison.jpg"
        ),
        combined
    )

    # ======================================
    # SHOW RESULT
    # ======================================
    cv2.imshow(
        f"Original | Distorted | Undistorted - {filename}",
        combined
    )
    # Press any key to continue
    # to the next photo
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print()
print("===================================")
print("✅ All photos have been processed!")
print("===================================")
print()
print(f"Results are located here:")
print(output_folder)