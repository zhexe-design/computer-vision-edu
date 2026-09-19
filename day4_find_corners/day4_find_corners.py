import cv2
import numpy as np
import glob
import os

# ==========================================
# НАСТРОЙКИ
# ==========================================

# Количество ВНУТРЕННИХ углов шахматной доски
# Например: 9 углов по горизонтали и 6 по вертикали
CHESSBOARD_SIZE = (9, 6)

# Размер одной клетки в метрах
# Пока это значение не очень важно
SQUARE_SIZE = 0.025

# Критерии уточнения положения углов
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================================
# ПОДГОТОВКА 3D-ТОЧЕК
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

# Здесь будут храниться точки
objpoints = []  # 3D точки
imgpoints = []  # 2D точки

print("Программа запущена.")
print(f"Размер доски: {CHESSBOARD_SIZE}")
print(f"Размер клетки: {SQUARE_SIZE} м")

# ==========================================
# ПОИСК ФОТОГРАФИЙ
# ==========================================

images = (
    glob.glob("calibration_images/*.jpg")
    + glob.glob("calibration_images/*.png")
)

# ==========================================
# ЕСЛИ ФОТОГРАФИЙ ПОКА НЕТ
# ==========================================

if len(images) == 0:
    print()
    print("Фотографии не найдены.")
    print("Папка calibration_images/ пока пустая.")
    print()
    print("Подготовка 3D-точек прошла успешно.")
    print(f"Количество точек на одной доске: {len(objp)}")

else:
    # ==========================================
    # ОБРАБОТКА ФОТОГРАФИЙ
    # ==========================================

    print(f"Найдено фотографий: {len(images)}")
    print()

    for fname in images:

        print(f"Обрабатывается: {fname}")

        img = cv2.imread(fname)

        if img is None:
            print("  Ошибка загрузки изображения")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Поиск углов шахматной доски
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SIZE,
            None
        )

        if ret:
            print("  Углы найдены!")

            # Уточняем положение углов
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            # Сохраняем точки
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Рисуем найденные углы
            cv2.drawChessboardCorners(
                img,
                CHESSBOARD_SIZE,
                corners2,
                ret
            )

            # Показываем изображение
            cv2.imshow("Corners", img)
            cv2.waitKey(500)

        else:
            print("  Углы НЕ найдены.")

    cv2.destroyAllWindows()

    # ==========================================
    # РЕЗУЛЬТАТ
    # ==========================================

    print()
    print("================================")
    print("Результат")
    print("================================")

    print(f"Всего фотографий: {len(images)}")
    print(f"Успешно найдено досок: {len(objpoints)}")

    if len(objpoints) > 0:
        print("Можно переходить к калибровке камеры.")
    else:
        print("Ни на одной фотографии доска не найдена.")