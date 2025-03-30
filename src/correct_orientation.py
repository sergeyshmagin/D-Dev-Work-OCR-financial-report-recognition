import cv2
import numpy as np
import os

def correct_orientation(image: np.ndarray) -> np.ndarray:
    """
    Универсально определяет, нужно ли повернуть портретную страницу с альбомной таблицей.
    Нормальные страницы не трогаются.
    """
    h, w = image.shape[:2]
    is_portrait = h > w

    # Предобработка изображения
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Поиск линий
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None:
        return image  # Линий не найдено — не трогаем

    # Считаем углы наклона линий
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Фильтрация "почти вертикальных" линий
    vertical_like_lines = [a for a in angles if abs(abs(a) - 90) < 15]

    # Фильтрация "почти горизонтальных" линий
    horizontal_like_lines = [a for a in angles if abs(a) < 15]

    # Условия поворота:
    # - изображение портретное (высокое)
    # - вертикальных линий больше, чем горизонтальных
    # - и достаточно много (например, >10)
    if is_portrait and len(vertical_like_lines) > 10 and len(vertical_like_lines) > len(horizontal_like_lines) * 1.5:
        print(f"🔁 Поворот: вертикальных линий {len(vertical_like_lines)}, горизонтальных {len(horizontal_like_lines)}")
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image

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