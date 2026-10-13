from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Image dimensions
WIDTH = 640
HEIGHT = 480

# Camera intrinsic parameters
FX = 800.0
FY = 800.0
CX = WIDTH / 2
CY = HEIGHT / 2

# Camera intrinsic matrix
K = np.array(
    [
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1],
    ],
    dtype=np.float32,
)

# Camera rotation and translation vectors
RVEC = np.array([0.5, 0.5, 0.0], dtype=np.float32)
TVEC = np.array([0.0, 0.0, 4.0], dtype=np.float32)

# Output directory and file
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "result.png"


# ============================================================
# 3D SCENE
# ============================================================

# Unit cube vertices in 3D space
POINTS_3D = np.array(
    [
        [0, 0, 0],  # Origin
        [1, 0, 0],  # X-axis
        [0, 1, 0],  # Y-axis
        [0, 0, 1],  # Z-axis
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)

# Cube edges defined by vertex indices
CUBE_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """Project the 3D cube and render the result."""

    # Project 3D points into 2D image coordinates
    points_2d, _ = cv2.projectPoints(
        POINTS_3D,
        RVEC,
        TVEC,
        K,
        None,
    )

    points_2d = points_2d.reshape(-1, 2)

    # Validate the projection output
    if points_2d.shape != (8, 2):
        raise RuntimeError(
            f"Unexpected projected points shape: {points_2d.shape}"
        )

    if not np.all(np.isfinite(points_2d)):
        raise RuntimeError("Projection produced invalid coordinates.")

    # Create a white background
    img = np.ones(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    ) * 255

    # Draw cube edges in gray
    for i, j in CUBE_EDGES:
        pt1 = tuple(points_2d[i].astype(int))
        pt2 = tuple(points_2d[j].astype(int))

        cv2.line(
            img,
            pt1,
            pt2,
            (180, 180, 180),
            2,
        )

    # Draw projected vertices as black points
    for pt in points_2d:
        x, y = pt.astype(int)

        cv2.circle(
            img,
            (int(x), int(y)),
            6,
            (0, 0, 0),
            -1,
        )

    # Draw coordinate axes from the origin
    origin = tuple(points_2d[0].astype(int))

    # X-axis: red
    cv2.line(
        img,
        origin,
        tuple(points_2d[1].astype(int)),
        (0, 0, 255),
        3,
    )

    # Y-axis: green
    cv2.line(
        img,
        origin,
        tuple(points_2d[2].astype(int)),
        (0, 255, 0),
        3,
    )

    # Z-axis: blue
    cv2.line(
        img,
        origin,
        tuple(points_2d[3].astype(int)),
        (255, 0, 0),
        3,
    )

    # Create the output directory if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the rendered image
    success = cv2.imwrite(str(OUTPUT_PATH), img)

    if not success:
        raise RuntimeError(f"Failed to save image to: {OUTPUT_PATH}")

    print("Projection completed successfully!")
    print(f"Projected vertices: {len(points_2d)}")
    print(f"Cube edges: {len(CUBE_EDGES)}")
    print(f"Image resolution: {WIDTH}x{HEIGHT}")
    print(f"Result saved to: {OUTPUT_PATH}")

    # Display the result
    cv2.imshow("Pinhole Projection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()