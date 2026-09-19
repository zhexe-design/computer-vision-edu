import cv2
import numpy as np
import os
import glob


# ==========================================
# ПАПКА С ФОТОГРАФИЯМИ
# ==========================================

input_folder = r"Y:\1\week1_camera_models\day3_distortion\images"

# Папка, куда будут сохраняться результаты
output_folder = os.path.join(
    r"Y:\1\week1_camera_models\day3_distortion",
    "results"
)

os.makedirs(output_folder, exist_ok=True)


# ==========================================
# НАХОДИМ ВСЕ ФОТОГРАФИИ
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
    print("❌ Фотографии не найдены!")
    print(f"Проверь папку:")
    print(input_folder)
    exit()


print(f"Найдено фотографий: {len(image_paths)}")
print()


# ==========================================
# ОБРАБАТЫВАЕМ КАЖДОЕ ФОТО
# ==========================================

for image_path in image_paths:

    # --------------------------------------
    # Загружаем изображение
    # --------------------------------------

    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Не удалось открыть: {image_path}")
        continue

    h, w = img.shape[:2]

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]

    print(f"Обрабатывается: {filename}")
    print(f"Размер: {w} x {h}")


    # ======================================
    # СОЗДАЁМ МАТРИЦУ КАМЕРЫ
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
    # КОЭФФИЦИЕНТЫ ДИСТОРСИИ
    # ======================================

    # k1, k2, p1, p2, k3

    dist_coeffs = np.array(
        [-0.3, 0.1, 0.0, 0.0, 0.0],
        dtype=np.float32
    )


    # ======================================
    # СОЗДАЁМ ИСКУССТВЕННУЮ ДИСТОРСИЮ
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
    # ИСПРАВЛЯЕМ ДИСТОРСИЮ
    # ======================================

    undistorted = cv2.undistort(
        distorted,
        K,
        dist_coeffs
    )


    # ======================================
    # СОЗДАЁМ ИЗОБРАЖЕНИЕ ДЛЯ СРАВНЕНИЯ
    # ======================================

    original_labeled = img.copy()
    distorted_labeled = distorted.copy()
    undistorted_labeled = undistorted.copy()


    # Подписи

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


    # Объединяем три изображения

    combined = np.hstack([
        original_labeled,
        distorted_labeled,
        undistorted_labeled
    ])


    # ======================================
    # СОХРАНЯЕМ РЕЗУЛЬТАТЫ
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
    # ПОКАЗЫВАЕМ РЕЗУЛЬТАТ
    # ======================================

    cv2.imshow(
        f"Original | Distorted | Undistorted - {filename}",
        combined
    )

    # Нажми любую клавишу, чтобы перейти
    # к следующей фотографии

    cv2.waitKey(0)

    cv2.destroyAllWindows()


print()
print("===================================")
print("✅ Все фотографии обработаны!")
print("===================================")
print()
print(f"Результаты находятся здесь:")
print(output_folder)