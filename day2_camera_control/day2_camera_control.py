import numpy as np
import cv2
import math
import imageio.v2 as imageio


# ============================================================
# НАСТРОЙКИ ИЗОБРАЖЕНИЯ И КАМЕРЫ
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
# КУБ (единичный, от 0 до 1 по каждой оси)
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
# СЕТКА ПОД КУБОМ
#
# Куб лежит основанием на Y = 0, поэтому сетка находится
# в той же горизонтальной плоскости Y = 0 (плоскость XZ).
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
# ПРОЕКЦИЯ 3D -> 2D
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
# КАМЕРА "СМОТРИТ НА ТОЧКУ" (look-at)
# ============================================================

def camera_look_at(camera_position, target):

    camera_position = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    # Направление взгляда камеры (ось Z камеры)
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    # Мировая вертикаль
    world_up = np.array([0, 1, 0], dtype=np.float32)

    # Ось "вправо" камеры (ось X камеры)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Ось Y камеры.
    #
    # ВАЖНО: берём именно "down = forward x right", а НЕ
    # "up = right x forward". Это даёт правостороннюю
    # (det = +1) систему координат, как того ожидает
    # cv2.Rodrigues, и совпадает со стандартной конвенцией
    # OpenCV, где ось Y камеры направлена вниз.
    #
    # Если вместо этого использовать "up", базис получается
    # левосторонним (det = -1) — это зеркальное отражение,
    # а не поворот, и Rodrigues/projectPoints начинают врать
    # по-разному в зависимости от угла обзора. Именно из-за
    # этого камера визуально "уезжала по диагонали" вместо
    # чистого горизонтального облёта.
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Матрица camera -> world: столбцы — оси камеры (X, Y, Z),
    # выраженные в мировых координатах. Теперь det(R_c2w) = +1.
    R_c2w = np.column_stack([right, down, forward]).astype(np.float32)

    # cv2.projectPoints требует поворот world -> camera, то есть
    # обратную матрицу. Для ортогональной матрицы это транспонирование.
    R = R_c2w.T

    rvec, _ = cv2.Rodrigues(R)

    # World -> Camera смещение
    tvec = -R @ camera_position
    tvec = tvec.reshape(3, 1)

    return rvec.astype(np.float32), tvec.astype(np.float32)


# ============================================================
# ОТРИСОВКА СЕТКИ
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
# ОТРИСОВКА КУБА
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
# ПОДПИСИ ОСЕЙ X И Y
#
# points_3d[0] = (0,0,0) — начало координат куба
# points_3d[1] = (1,0,0) — конец оси X
# points_3d[2] = (0,1,0) — конец оси Y
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
        (0, 0, 255),   # красный, как ось X
        2
    )

    cv2.putText(
        img,
        "Y",
        y_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 0),   # зелёный, как ось Y
        2
    )


# ============================================================
# ПАРАМЕТРЫ ОРБИТЫ КАМЕРЫ
# ============================================================

# Центр куба — камера всегда смотрит именно сюда
TARGET = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Расстояние от камеры до центра куба (постоянное)
CAMERA_DISTANCE = 4.0

# Высота камеры над плоскостью сетки (постоянная -> движение
# происходит СТРОГО в одной горизонтальной плоскости)
CAMERA_HEIGHT = 1.5

# Разница высот между камерой и центром куба
height_difference = CAMERA_HEIGHT - TARGET[1]

# Радиус окружности облёта в плоскости XZ, при котором расстояние
# камера-цель остаётся равным CAMERA_DISTANCE на любом угле
HORIZONTAL_RADIUS = math.sqrt(CAMERA_DISTANCE ** 2 - height_difference ** 2)


print("=" * 38)
print("ОБЛЁТ КАМЕРЫ ВОКРУГ КУБА")
print("=" * 38)
print()
print(f"Расстояние до центра куба: {CAMERA_DISTANCE:.2f}")
print(f"Высота камеры:             {CAMERA_HEIGHT:.2f}")
print(f"Радиус горизонтальной окружности: {HORIZONTAL_RADIUS:.2f}")
print()
print("Камера движется строго по горизонтальной")
print("плоскости, полным кругом вокруг куба.")
print()
print("ESC — остановить")
print()


# ============================================================
# АНИМАЦИЯ
# ============================================================

frames = []

# 0..358 градусов с шагом 2 -> ровно полный круг без дублирования
# последнего кадра с первым (для гладкой зацикленной GIF)
for angle in range(0, 360, 2):

    theta = math.radians(angle)

    # ------------------------------------------------------------
    # Положение камеры на окружности.
    # camera_y ПОСТОЯННА -> движение только в горизонтальной
    # плоскости, без каких-либо изменений по высоте.
    # ------------------------------------------------------------

    camera_x = TARGET[0] + HORIZONTAL_RADIUS * math.cos(theta)
    camera_z = TARGET[2] + HORIZONTAL_RADIUS * math.sin(theta)
    camera_y = CAMERA_HEIGHT

    camera_position = np.array(
        [camera_x, camera_y, camera_z],
        dtype=np.float32
    )

    # Камера всегда смотрит на центр куба
    rvec, tvec = camera_look_at(camera_position, TARGET)

    # Проекция куба и сетки одной и той же камерой
    points_2d = project_points(points_3d, rvec, tvec)
    grid_2d = project_points(grid_points, rvec, tvec)

    # ------------------------------------------------------------
    # Кадр
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
# СОХРАНЕНИЕ GIF
# ============================================================

if len(frames) > 0:
    imageio.mimsave(
        "orbit.gif",
        frames,
        duration=0.03,
        loop=0
    )

    print()
    print("Готово!")
    print("Файл: orbit.gif")

print()
print("ОБЛЁТ ЗАВЕРШЁН!")
