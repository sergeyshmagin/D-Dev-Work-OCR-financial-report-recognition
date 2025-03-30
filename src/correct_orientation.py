import cv2
import numpy as np
import os


# def remove_scan_artifacts(image: np.ndarray, max_strip_ratio: float = 0.25) -> np.ndarray:
#     """
#     Удаляет левый край (спираль, тени, артефакты сканера), не затрагивая содержимое страницы.
#     Параметр max_strip_ratio — максимальная доля ширины, которую можно обрезать (0.25 = 25%).
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape
#     max_strip = int(w * max_strip_ratio)

#     # Усиливаем вертикальные структуры
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobelx = np.absolute(sobelx).astype(np.uint8)
#     projection = np.sum(sobelx, axis=0)

#     # Нормализуем и сглаживаем проекцию
#     projection = cv2.GaussianBlur(projection, (15, 1), 0)

#     # Ищем левую границу с минимальным количеством текста
#     threshold = np.max(projection[:max_strip]) * 0.4
#     cut_index = 0
#     for i in range(0, max_strip):
#         if projection[i] > threshold:
#             cut_index = i
#             break

#     if cut_index > 5:
#         print(f"✂️ Обрезан левый край на {cut_index} пикселей")
#         image = image[:, cut_index:]

#     return image


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """
    Универсальная функция:
    1. Авто-поворот "лежачих" страниц (альбомных, вставленных как портрет).
    2. Точное выравнивание контента относительно горизонта (deskew).
    """
    original = image.copy()
    h, w = image.shape[:2]
    is_portrait = h > w

    # --- Шаг 1: предобработка
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # --- Шаг 2: поиск линий
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None:
        return original  # Нет линий – возвращаем как есть

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # --- Шаг 3: поворот страницы, если она "лежит"
    vertical_like = [a for a in angles if abs(abs(a) - 90) < 15]
    horizontal_like = [a for a in angles if abs(a) < 15]

    rotated = False
    if is_portrait and len(vertical_like) > 10 and len(vertical_like) > len(horizontal_like) * 1.5:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated = True
        print(f"🔁 Поворот: вертикальных {len(vertical_like)}, горизонтальных {len(horizontal_like)}")

    # --- Шаг 4: deskew (точное выравнивание контента)
    # Повторно ищем углы после возможного поворота
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return image  # Без выравнивания

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:  # анализируем только почти горизонтальные
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image  # наклон слишком мал

    print(f"📐 Deskew: поворот на {median_angle:.2f}°")

    # Поворот без изменения размеров
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
    new_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)

    # Центрирование
    M[0, 2] += (new_w - image.shape[1]) / 2
    M[1, 2] += (new_h - image.shape[0]) / 2

    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated

def process_images(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Обработка: {filename}")
        image = cv2.imread(input_path)

        if image is None:
            print(f"⚠ Не удалось загрузить: {filename}")
            continue

        rotated = correct_orientation(image)
        cv2.imwrite(output_path, rotated)
        print(f"✅ Сохранено: {output_path}")

if __name__ == "__main__":
    input_dir = "data/reports/intermediate_images"
    output_dir = "data/reports/corrected_images"
    process_images(input_dir, output_dir)