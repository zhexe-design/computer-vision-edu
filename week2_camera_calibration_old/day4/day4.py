import os
import glob
import cv2
import numpy as np
import yaml


# ============================================================
# SETTINGS
# ============================================================

# Folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Normal photos that were NOT used for calibration
TEST_FOLDER = os.path.join(
    BASE_DIR,
    "test_images"
)

# Results folder
RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

# Folder for Original | Undistorted comparisons
BEFORE_AFTER_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "before_after"
)

# Saved camera parameters
PARAMS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "camera_params.yaml"
)


# ============================================================
# CREATE RESULT FOLDERS
# ============================================================

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(BEFORE_AFTER_FOLDER, exist_ok=True)


# ============================================================
# FUNCTION: LOAD CAMERA PARAMETERS
# ============================================================

def load_camera_parameters(params_file):

    if not os.path.exists(params_file):
        print("\nERROR: Camera parameters file was not found.")
        print(f"Expected file:")
        print(params_file)

        print("\nRun day1_full_calibration.py first.")
        return None, None, None

    with open(params_file, "r") as f:
        data = yaml.safe_load(f)

    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64)

    image_width = int(data["image_width"])
    image_height = int(data["image_height"])

    return K, dist, (image_width, image_height)


# ============================================================
# LOAD CAMERA PARAMETERS
# ============================================================

print("\n========================================")
print("LOADING CAMERA PARAMETERS")
print("========================================")

K, dist, calibration_size = load_camera_parameters(
    PARAMS_FILE
)

if K is None:
    exit()


print("\nCamera matrix K:")
print(K)

print("\nDistortion coefficients:")
print(dist)

print(
    f"\nCalibration image size: "
    f"{calibration_size[0]} x {calibration_size[1]}"
)

print("/n Check")


# ============================================================
# LOAD TEST IMAGES
# ============================================================

print("\n========================================")
print("LOADING TEST IMAGES")
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


test_images = sorted(list(set(test_images)))


print(f"\nTest folder:")
print(TEST_FOLDER)

print(
    f"\nFound {len(test_images)} test images."
)


if len(test_images) == 0:

    print("\nERROR: No test images were found.")

    print(
        "\nPut 5–8 normal photos into:"
    )

    print(TEST_FOLDER)

    print(
        "\nExamples:"
    )

    print("- door")
    print("- window")
    print("- book")
    print("- furniture")
    print("- street")

    exit()


if len(test_images) < 5:

    print(
        "\nWARNING: It is recommended to use "
        "at least 5 test images."
    )

elif len(test_images) > 8:

    print(
        "\nWARNING: More than 8 test images were found."
    )

    print(
        "The program will process all of them."
    )


# ============================================================
# FUNCTION: SCALE CAMERA MATRIX
# ============================================================

def scale_camera_matrix(
    K,
    original_size,
    new_size
):

    original_width, original_height = original_size
    new_width, new_height = new_size

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    K_scaled = K.copy()

    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y

    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y

    return K_scaled


# ============================================================
# PROCESS TEST IMAGES
# ============================================================

print("\n========================================")
print("UNDISTORTING TEST IMAGES")
print("========================================")


processed_images = 0


for i, image_path in enumerate(test_images):

    print(
        f"\nProcessing: "
        f"{os.path.basename(image_path)}"
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print("Could not read image.")
        continue


    # --------------------------------------------------------
    # Get image size
    # --------------------------------------------------------

    height, width = image.shape[:2]

    current_size = (
        width,
        height
    )


    print(
        f"Image size: "
        f"{width} x {height}"
    )


    # --------------------------------------------------------
    # Scale camera matrix if test image has
    # a different resolution
    # --------------------------------------------------------

    if current_size != calibration_size:

        K_current = scale_camera_matrix(
            K,
            calibration_size,
            current_size
        )

        print(
            "Camera matrix was scaled "
            "to the test image resolution."
        )

    else:

        K_current = K


    # --------------------------------------------------------
    # Undistort image
    # --------------------------------------------------------

    undistorted = cv2.undistort(
        image,
        K_current,
        dist
    )


    # --------------------------------------------------------
    # Add labels
    # --------------------------------------------------------

    original_labeled = image.copy()
    undistorted_labeled = undistorted.copy()


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
        undistorted_labeled,
        "Undistorted",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Create comparison
    # --------------------------------------------------------

    comparison = np.hstack(
        (
            original_labeled,
            undistorted_labeled
        )
    )


    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    output_path = os.path.join(
        BEFORE_AFTER_FOLDER,
        f"comparison_{i + 1}.jpg"
    )


    cv2.imwrite(
        output_path,
        comparison
    )


    print(
        f"Saved: {output_path}"
    )


    processed_images += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("DAY 2 QUALITY ANALYSIS COMPLETE")
print("========================================")


print(
    f"\nProcessed images: "
    f"{processed_images}"
)


print(
    "\nCamera parameters loaded from:"
)

print(PARAMS_FILE)


print(
    "\nComparison images saved to:"
)

print(BEFORE_AFTER_FOLDER)


print(
    "\nOpen the comparison images and check:"
)

print(
    "Original  ->  Undistorted"
)

print(
    "\nLook at straight lines such as:"
)

print("- door frames")
print("- windows")
print("- shelves")
print("- table edges")
print("- walls")
print("- building lines")


print(
    "\nThe goal is to check whether "
    "straight lines look straighter "
    "after undistortion."
)