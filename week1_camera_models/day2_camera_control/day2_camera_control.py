 import numpy as np
import cv2
import math
import imageio.v2 as imageio


# ============================================================
# IMAGE AND CAMERA SETTINGS
# ============================================================

WIDTH = 640
HEIGHT = 480

FX = 800.0
FY = 800.0
CX = WIDTH / 2
CY = HEIGHT / 2

K = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float32)


# ============================================================
# CUBE (unit, from 0 to 1 on each axis)
# ============================================================

points_3d = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
], dtype=np.float32)

cube_edges = [
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
    (6, 7)
]


# ============================================================
# GRID UNDER THE CUBE
#
# The cube rests on its base at Y = 0, so the grid is located
# in the same horizontal plane Y = 0 (XZ plane).
# ============================================================

GRID_SIZE = 4.0
GRID_STEP = 0.5

grid_points = []

for x in np.arange(-GRID_SIZE, GRID_SIZE + GRID_STEP, GRID_STEP):
    for z in np.arange(-GRID_SIZE, GRID_SIZE + GRID_STEP, GRID_STEP):
        grid_points.append([x, 0.0, z])

grid_points = np.array(grid_points, dtype=np.float32)

GRID_LINE_COUNT = int(round((2 * GRID_SIZE) / GRID_STEP)) + 1


# ============================================================
# PROJECTION 3D -> 2D
# ============================================================

def project_points(points, rvec, tvec):

    points_2d, _ = cv2.projectPoints(
        points,
        rvec,
        tvec,
        K,
        None
    )

    return points_2d.reshape(-1, 2)


# ============================================================
# CAMERA "LOOKS AT A POINT" (look-at)
# ============================================================

def camera_look_at(camera_position, target):

    camera_position = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    # Camera viewing direction (camera Z axis)
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    # World vertical
    world_up = np.array([0, 1, 0], dtype=np.float32)

    # Camera "right" axis (camera X axis)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Camera Y axis.
    #
    # IMPORTANT: use "down = forward x right", NOT
    # "up = right x forward". This gives a right-handed
    # (det = +1) coordinate system, as expected by
    # cv2.Rodrigues, and matches the standard OpenCV convention,
    # where the camera Y axis points downward.
    #
    # If "up" is used instead, the basis becomes
    # left-handed (det = -1) — this is a mirror reflection,
    # not a rotation, and Rodrigues/projectPoints start behaving
    # differently depending on the viewing angle. This is exactly
    # why the camera was visually "moving diagonally" instead of
    # making a pure horizontal orbit.
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Camera -> world matrix: columns are the camera axes (X, Y, Z),
    # expressed in world coordinates. Now det(R_c2w) = +1.
    R_c2w = np.column_stack([right, down, forward]).astype(np.float32)

    # cv2.projectPoints requires a world -> camera rotation, i.e.
    # the inverse matrix. For an orthogonal matrix, this is the transpose.
    R = R_c2w.T

    rvec, _ = cv2.Rodrigues(R)

    # World -> Camera translation
    tvec = -R @ camera_position
    tvec = tvec.reshape(3, 1)

    return rvec.astype(np.float32), tvec.astype(np.float32)


# ============================================================
# DRAWING THE GRID
# ============================================================

def draw_grid(img, grid_2d):

    count = GRID_LINE_COUNT

    for row in range(count):
        start = row * count
        end = start + count - 1

        p1 = grid_2d[start]
        p2 = grid_2d[end]

        cv2.line(
            img,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (210, 210, 210),
            1
        )

    for col in range(count):
        start = col
        end = (count - 1) * count + col

        p1 = grid_2d[start]
        p2 = grid_2d[end]

        cv2.line(
            img,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (210, 210, 210),
            1
        )


# ============================================================
# DRAWING THE CUBE
# ============================================================

def draw_cube(img, points_2d):

    for start, end in cube_edges:
        p1 = points_2d[start]
        p2 = points_2d[end]

        cv2.line(
            img,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (80, 80, 80),
            2
        )

    for point in points_2d:
        x = int(round(point[0]))
        y = int(round(point[1]))

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)


# ============================================================
# X AND Y AXIS LABELS
#
# points_3d[0] = (0,0,0) — origin of the cube
# points_3d[1] = (1,0,0) — end of the X axis
# points_3d[2] = (0,1,0) — end of the Y axis
# ============================================================

def draw_axis_labels(img, points_2d):

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
        (0, 0, 255),   # red, like the X axis
        2
    )

    cv2.putText(
        img,
        "Y",
        y_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 0),   # green, like the Y axis
        2
    )


# ============================================================
# CAMERA ORBIT PARAMETERS
# ============================================================

# Center of the cube — the camera always looks exactly here
TARGET = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Distance from the camera to the center of the cube (constant)
CAMERA_DISTANCE = 4.0

# Camera height above the grid plane (constant -> movement
# occurs STRICTLY in one horizontal plane)
CAMERA_HEIGHT = 1.5

# Height difference between the camera and the center of the cube
height_difference = CAMERA_HEIGHT - TARGET[1]

# Radius of the orbit circle in the XZ plane, at which the distance
# camera-target remains equal to CAMERA_DISTANCE at any angle
HORIZONTAL_RADIUS = math.sqrt(CAMERA_DISTANCE ** 2 - height_difference ** 2)


print("=" * 38)
print("CAMERA ORBIT AROUND THE CUBE")
print("=" * 38)
print()
print(f"Distance to cube center: {CAMERA_DISTANCE:.2f}")
print(f"Camera height:             {CAMERA_HEIGHT:.2f}")
print(f"Horizontal orbit radius: {HORIZONTAL_RADIUS:.2f}")
print()
print("The camera moves strictly along a horizontal")
print("plane, making a full circle around the cube.")
print()
print("ESC — stop")
print()


# ============================================================
# ANIMATION
# ============================================================

frames = []

# 0..358 degrees with a step of 2 -> exactly one full circle without
# duplicating the last frame with the first (for a smooth looping GIF)
for angle in range(0, 360, 2):

    theta = math.radians(angle)

    # ------------------------------------------------------------
    # Camera position on the circle.
    # camera_y is CONSTANT -> movement only in the horizontal
    # plane, without any changes in height.
    # ------------------------------------------------------------

    camera_x = TARGET[0] + HORIZONTAL_RADIUS * math.cos(theta)
    camera_z = TARGET[2] + HORIZONTAL_RADIUS * math.sin(theta)
    camera_y = CAMERA_HEIGHT

    camera_position = np.array(
        [camera_x, camera_y, camera_z],
        dtype=np.float32
    )

    # Camera always looks at the center of the cube
    rvec, tvec = camera_look_at(camera_position, TARGET)

    # Projection of the cube and grid using the same camera
    points_2d = project_points(points_3d, rvec, tvec)
    grid_2d = project_points(grid_points, rvec, tvec)

    # ------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------

    img = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    draw_grid(img, grid_2d)
    draw_cube(img, points_2d)
    draw_axis_labels(img, points_2d)

    cv2.putText(
        img,
        f"Camera angle: {angle} deg",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.putText(
        img,
        "Horizontal orbit around cube",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.putText(
        img,
        f"Distance: {CAMERA_DISTANCE:.1f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.imshow("Camera Orbit", img)

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

    key = cv2.waitKey(30)
    if key == 27:
        break


cv2.destroyAllWindows()


# ============================================================
# SAVE GIF
# ============================================================

if len(frames) > 0:
    imageio.mimsave(
        "orbit.gif",
        frames,
        duration=0.03,
        loop=0
    )

    print()
    print("Done!")
    print("File: orbit.gif")

print()
print("ORBIT COMPLETED!")

import numpy as np
import cv2
import math
import imageio.v2 as imageio


# ============================================================
# IMAGE AND CAMERA SETTINGS
# ============================================================

WIDTH = 640
HEIGHT = 480

FX = 800.0
FY = 800.0
CX = WIDTH / 2
CY = HEIGHT / 2

K = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float32)


# ============================================================
# CUBE (unit, from 0 to 1 on each axis)
# ============================================================

points_3d = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
], dtype=np.float32)

cube_edges = [
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
    (6, 7)
]


# ============================================================
# GRID UNDER THE CUBE
#
# The cube rests on its base at Y = 0, so the grid is located
# in the same horizontal plane Y = 0 (XZ plane).
# ============================================================

GRID_SIZE = 4.0
GRID_STEP = 0.5

grid_points = []

for x in np.arange(-GRID_SIZE, GRID_SIZE + GRID_STEP, GRID_STEP):
    for z in np.arange(-GRID_SIZE, GRID_SIZE + GRID_STEP, GRID_STEP):
        grid_points.append([x, 0.0, z])

grid_points = np.array(grid_points, dtype=np.float32)

GRID_LINE_COUNT = int(round((2 * GRID_SIZE) / GRID_STEP)) + 1


# ============================================================
# PROJECTION 3D -> 2D
# ============================================================

def project_points(points, rvec, tvec):

    points_2d, _ = cv2.projectPoints(
        points,
        rvec,
        tvec,
        K,
        None
    )

    return points_2d.reshape(-1, 2)


# ============================================================
# CAMERA "LOOKS AT A POINT" (look-at)
# ============================================================

def camera_look_at(camera_position, target):

    camera_position = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    # Camera viewing direction (camera Z axis)
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    # World vertical
    world_up = np.array([0, 1, 0], dtype=np.float32)

    # Camera "right" axis (camera X axis)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Camera Y axis.
    #
    # IMPORTANT: use "down = forward x right", NOT
    # "up = right x forward". This gives a right-handed
    # (det = +1) coordinate system, as expected by
    # cv2.Rodrigues, and matches the standard OpenCV convention,
    # where the camera Y axis points downward.
    #
    # If "up" is used instead, the basis becomes
    # left-handed (det = -1) — this is a mirror reflection,
    # not a rotation, and Rodrigues/projectPoints start behaving
    # differently depending on the viewing angle. This is exactly
    # why the camera was visually "moving diagonally" instead of
    # making a pure horizontal orbit.
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Camera -> world matrix: columns are the camera axes (X, Y, Z),
    # expressed in world coordinates. Now det(R_c2w) = +1.
    R_c2w = np.column_stack([right, down, forward]).astype(np.float32)

    # cv2.projectPoints requires a world -> camera rotation, i.e.
    # the inverse matrix. For an orthogonal matrix, this is the transpose.
    R = R_c2w.T

    rvec, _ = cv2.Rodrigues(R)

    # World -> Camera translation
    tvec = -R @ camera_position
    tvec = tvec.reshape(3, 1)

    return rvec.astype(np.float32), tvec.astype(np.float32)


# ============================================================
# DRAWING THE GRID
# ============================================================

def draw_grid(img, grid_2d):

    count = GRID_LINE_COUNT

    for row in range(count):
        start = row * count
        end = start + count - 1

        p1 = grid_2d[start]
        p2 = grid_2d[end]

        cv2.line(
            img,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (210, 210, 210),
            1
        )

    for col in range(count):
        start = col
        end = (count - 1) * count + col

        p1 = grid_2d[start]
        p2 = grid_2d[end]

        cv2.line(
            img,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (210, 210, 210),
            1
        )


# ============================================================
# DRAWING THE CUBE
# ============================================================

def draw_cube(img, points_2d):

    for start, end in cube_edges:
        p1 = points_2d[start]
        p2 = points_2d[end]

        cv2.line(
            img,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (80, 80, 80),
            2
        )

    for point in points_2d:
        x = int(round(point[0]))
        y = int(round(point[1]))

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)


# ============================================================
# X AND Y AXIS LABELS
#
# points_3d[0] = (0,0,0) — origin of the cube
# points_3d[1] = (1,0,0) — end of the X axis
# points_3d[2] = (0,1,0) — end of the Y axis
# ============================================================

def draw_axis_labels(img, points_2d):

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
        (0, 0, 255),   # red, like the X axis
        2
    )

    cv2.putText(
        img,
        "Y",
        y_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 0),   # green, like the Y axis
        2
    )


# ============================================================
# CAMERA ORBIT PARAMETERS
# ============================================================

# Center of the cube — the camera always looks exactly here
TARGET = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Distance from the camera to the center of the cube (constant)
CAMERA_DISTANCE = 4.0

# Camera height above the grid plane (constant -> movement
# occurs STRICTLY in one horizontal plane)
CAMERA_HEIGHT = 1.5

# Height difference between the camera and the center of the cube
height_difference = CAMERA_HEIGHT - TARGET[1]

# Radius of the orbit circle in the XZ plane, at which the distance
# camera-target remains equal to CAMERA_DISTANCE at any angle
HORIZONTAL_RADIUS = math.sqrt(CAMERA_DISTANCE ** 2 - height_difference ** 2)


print("=" * 38)
print("CAMERA ORBIT AROUND THE CUBE")
print("=" * 38)
print()
print(f"Distance to cube center: {CAMERA_DISTANCE:.2f}")
print(f"Camera height:             {CAMERA_HEIGHT:.2f}")
print(f"Horizontal orbit radius: {HORIZONTAL_RADIUS:.2f}")
print()
print("The camera moves strictly along a horizontal")
print("plane, making a full circle around the cube.")
print()
print("ESC — stop")
print()


# ============================================================
# ANIMATION
# ============================================================

frames = []

# 0..358 degrees with a step of 2 -> exactly one full circle without
# duplicating the last frame with the first (for a smooth looping GIF)
for angle in range(0, 360, 2):

    theta = math.radians(angle)

    # ------------------------------------------------------------
    # Camera position on the circle.
    # camera_y is CONSTANT -> movement only in the horizontal
    # plane, without any changes in height.
    # ------------------------------------------------------------

    camera_x = TARGET[0] + HORIZONTAL_RADIUS * math.cos(theta)
    camera_z = TARGET[2] + HORIZONTAL_RADIUS * math.sin(theta)
    camera_y = CAMERA_HEIGHT

    camera_position = np.array(
        [camera_x, camera_y, camera_z],
        dtype=np.float32
    )

    # Camera always looks at the center of the cube
    rvec, tvec = camera_look_at(camera_position, TARGET)

    # Projection of the cube and grid using the same camera
    points_2d = project_points(points_3d, rvec, tvec)
    grid_2d = project_points(grid_points, rvec, tvec)

    # ------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------

    img = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    draw_grid(img, grid_2d)
    draw_cube(img, points_2d)
    draw_axis_labels(img, points_2d)

    cv2.putText(
        img,
        f"Camera angle: {angle} deg",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.putText(
        img,
        "Horizontal orbit around cube",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.putText(
        img,
        f"Distance: {CAMERA_DISTANCE:.1f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.imshow("Camera Orbit", img)

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

    key = cv2.waitKey(30)
    if key == 27:
        break


cv2.destroyAllWindows()


# ============================================================
# SAVE GIF
# ============================================================

if len(frames) > 0:
    imageio.mimsave(
        "orbit.gif",
        frames,
        duration=0.03,
        loop=0
    )

    print()
    print("Done!")
    print("File: orbit.gif")

print()
print("ORBIT COMPLETED!")