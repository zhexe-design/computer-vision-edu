import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


# ============================================================
# IMAGE AND CAMERA SETTINGS
# ============================================================

WIDTH = 640
HEIGHT = 480

FX = 800.0
FY = 800.0
CX = WIDTH / 2
CY = HEIGHT / 2

K = np.array(
    [
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1],
    ],
    dtype=np.float32,
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_GIF = OUTPUT_DIR / "orbit.gif"

# Animation settings
ANGLE_STEP = 2
FRAME_DELAY_MS = 30
GIF_FRAME_DURATION = 0.03


# ============================================================
# 3D SCENE
# ============================================================

# Unit cube vertices
POINTS_3D = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)

# Cube edges defined by pairs of vertex indices
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
# GROUND GRID
# ============================================================

GRID_SIZE = 4.0
GRID_STEP = 0.5

GRID_COORDINATES = np.arange(
    -GRID_SIZE,
    GRID_SIZE + GRID_STEP / 2,
    GRID_STEP,
    dtype=np.float32,
)

GRID_POINTS = np.array(
    [[x, 0.0, z] for x in GRID_COORDINATES for z in GRID_COORDINATES],
    dtype=np.float32,
)

GRID_LINE_COUNT = len(GRID_COORDINATES)


# ============================================================
# CAMERA ORBIT PARAMETERS
# ============================================================

# The camera always looks at the center of the cube.
TARGET = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Distance from the camera to the target remains constant.
CAMERA_DISTANCE = 4.0

# The camera's height remains constant during the orbit.
CAMERA_HEIGHT = 1.5

HEIGHT_DIFFERENCE = CAMERA_HEIGHT - float(TARGET[1])

if abs(HEIGHT_DIFFERENCE) >= CAMERA_DISTANCE:
    raise ValueError(
        "The vertical distance from the target must be smaller "
        "than CAMERA_DISTANCE."
    )

# Orbit radius in the XZ plane keeps the camera-target distance constant.
HORIZONTAL_RADIUS = math.sqrt(
    CAMERA_DISTANCE**2 - HEIGHT_DIFFERENCE**2
)


# ============================================================
# PROJECTION
# ============================================================

def project_points(
    points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Project an array of 3D points into 2D image coordinates."""
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    points_2d, _ = cv2.projectPoints(points, rvec, tvec, K, None)
    return points_2d.reshape(-1, 2)


# ============================================================
# CAMERA LOOK-AT
# ============================================================

def camera_look_at(
    camera_position: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate OpenCV rotation and translation vectors for a camera
    positioned at camera_position and looking toward target.
    """
    camera_position = np.asarray(camera_position, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)

    # Camera viewing direction (camera Z axis).
    forward = target - camera_position
    norm = np.linalg.norm(forward)

    if norm < 1e-8:
        raise ValueError("camera_position and target are too close")

    forward = forward / norm

    # World vertical direction.
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Camera X axis.
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-8:
        raise ValueError(
            "Camera viewing direction is parallel to world_up; "
            "a different up vector is required."
        )

    right = right / right_norm

    # Camera Y axis points downward in OpenCV's coordinate convention.
    down = np.cross(forward, right)
    down_norm = np.linalg.norm(down)

    if down_norm < 1e-8:
        raise ValueError("Could not construct a valid camera coordinate system")

    down = down / down_norm

    # Camera-to-world rotation matrix; columns are camera axes in world space.
    rotation_camera_to_world = np.column_stack(
        [right, down, forward]
    ).astype(np.float32)

    # OpenCV requires world-to-camera rotation.
    rotation_world_to_camera = rotation_camera_to_world.T

    rvec, _ = cv2.Rodrigues(rotation_world_to_camera)
    tvec = -rotation_world_to_camera @ camera_position.reshape(3, 1)

    return rvec.astype(np.float32), tvec.astype(np.float32)


# ============================================================
# DRAWING HELPERS
# ============================================================

def _pixel(point: np.ndarray) -> tuple[int, int]:
    """Convert a 2D point to integer pixel coordinates."""
    return tuple(np.round(point).astype(int))


def draw_grid(img: np.ndarray, grid_2d: np.ndarray) -> None:
    """Draw the projected ground grid."""
    count = GRID_LINE_COUNT

    # Lines running along the Z direction.
    for row in range(count):
        start = row * count
        end = start + count - 1

        cv2.line(
            img,
            _pixel(grid_2d[start]),
            _pixel(grid_2d[end]),
            (210, 210, 210),
            1,
        )

    # Lines running along the X direction.
    for col in range(count):
        start = col
        end = (count - 1) * count + col

        cv2.line(
            img,
            _pixel(grid_2d[start]),
            _pixel(grid_2d[end]),
            (210, 210, 210),
            1,
        )


def draw_cube(img: np.ndarray, points_2d: np.ndarray) -> None:
    """Draw all cube edges and visible projected vertices."""
    for start, end in CUBE_EDGES:
        cv2.line(
            img,
            _pixel(points_2d[start]),
            _pixel(points_2d[end]),
            (80, 80, 80),
            2,
        )

    for point in points_2d:
        x, y = _pixel(point)

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)


def draw_coordinate_axes(img: np.ndarray, points_2d: np.ndarray) -> None:
    """Draw color-coded X, Y, and Z axes from the cube origin."""
    origin = _pixel(points_2d[0])

    # X axis: red in BGR.
    cv2.line(img, origin, _pixel(points_2d[1]), (0, 0, 255), 2)

    # Y axis: green in BGR.
    cv2.line(img, origin, _pixel(points_2d[2]), (0, 255, 0), 2)

    # Z axis: blue in BGR.
    cv2.line(img, origin, _pixel(points_2d[3]), (255, 0, 0), 2)


def draw_axis_labels(img: np.ndarray, points_2d: np.ndarray) -> None:
    """Draw X and Y labels near their projected axis endpoints."""
    x_end = points_2d[1]
    y_end = points_2d[2]

    x_pos = (int(round(x_end[0])) + 10, int(round(x_end[1])) + 5)
    y_pos = (int(round(y_end[0])) + 10, int(round(y_end[1])) + 5)

    cv2.putText(
        img,
        "X",
        x_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        img,
        "Y",
        y_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 0),
        2,
    )


# ============================================================
# MAIN ANIMATION
# ============================================================

def main() -> None:
    """Render the camera orbit and save it as a looping GIF."""
    print("=" * 38)
    print("CAMERA ORBIT AROUND THE CUBE")
    print("=" * 38)
    print()
    print(f"Distance to cube center: {CAMERA_DISTANCE:.2f}")
    print(f"Camera height:           {CAMERA_HEIGHT:.2f}")
    print(f"Horizontal orbit radius: {HORIZONTAL_RADIUS:.2f}")
    print()
    print("The camera moves in a horizontal circle")
    print("and continuously looks at the cube center.")
    print()
    print("Press ESC to stop.")
    print()

    frames = []

    # Avoid duplicating the first frame at the end of the looping animation.
    try:
        for angle in range(0, 360, ANGLE_STEP):
            theta = math.radians(angle)

            # Camera position on a horizontal circle around the target.
            camera_x = TARGET[0] + HORIZONTAL_RADIUS * math.cos(theta)
            camera_z = TARGET[2] + HORIZONTAL_RADIUS * math.sin(theta)
            camera_position = np.array(
                [camera_x, CAMERA_HEIGHT, camera_z],
                dtype=np.float32,
            )

            # Point the camera toward the center of the cube.
            rvec, tvec = camera_look_at(camera_position, TARGET)

            # Project the cube and grid using the same camera pose.
            points_2d = project_points(POINTS_3D, rvec, tvec)
            grid_2d = project_points(GRID_POINTS, rvec, tvec)

            # Render the current frame.
            img = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

            draw_grid(img, grid_2d)
            draw_cube(img, points_2d)
            draw_coordinate_axes(img, points_2d)
            draw_axis_labels(img, points_2d)

            cv2.putText(
                img,
                f"Camera angle: {angle} deg",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                img,
                "Horizontal orbit around cube",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                img,
                f"Distance: {CAMERA_DISTANCE:.1f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            cv2.imshow("Camera Orbit", img)

            # imageio expects RGB frames, while OpenCV uses BGR.
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(FRAME_DELAY_MS)
            if key == 27:
                break
    finally:
        cv2.destroyAllWindows()

    if frames:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        imageio.mimsave(
            str(OUTPUT_GIF),
            frames,
            duration=GIF_FRAME_DURATION,
            loop=0,
        )

        print()
        print("Orbit completed successfully!")
        print(f"Frames saved: {len(frames)}")
        print(f"Saved: {OUTPUT_GIF}")
    else:
        print("No frames were generated; GIF was not saved.")


if __name__ == "__main__":
    main()
