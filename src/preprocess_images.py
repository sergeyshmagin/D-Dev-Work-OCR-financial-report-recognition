import cv2
import numpy as np
import os

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Улучшает локальный контраст с помощью CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def enhance_text_sharpness(image: np.ndarray) -> np.ndarray:
    """Уплотняет, затемняет слабый текст и делает его более жирным"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Адаптивная бинаризация текста
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=15, C=10)

    # Увеличим плотность символов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Восстановим изображение: текст черный, остальное белое
    mask = cv2.bitwise_not(dilated)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask == 0] = [0, 0, 0]
    return result


def enhance_table_lines(image: np.ndarray) -> np.ndarray:
    """Усиливает таблицы, делая линии более контрастными"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)

    thresh = cv2.adaptiveThreshold(inv, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    # Горизонтальные линии
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.erode(thresh, h_kernel)
    horizontal = cv2.dilate(horizontal, h_kernel)

    # Вертикальные линии
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.erode(thresh, v_kernel)
    vertical = cv2.dilate(vertical, v_kernel)

    # Объединяем и накладываем
    lines_mask = cv2.bitwise_or(horizontal, vertical)
    result = image.copy()
    result[lines_mask > 0] = [0, 0, 0]
    return result


def reconstruct_table_grid(image: np.ndarray) -> np.ndarray:
    """
    Восстанавливает сетку таблицы на изображении:
    - усиливает существующие линии;
    - дорисовывает недостающие фрагменты;
    - подходит для черно-белых и серых таблиц.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    
    # Подчеркнём структуры
    thresh = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    # Горизонтальные линии
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    # Вертикальные линии
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    # Комбинируем сетку
    grid = cv2.bitwise_or(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((2, 2), np.uint8), iterations=1)  # Утолщение

    # Накладываем сетку на изображение
    result = image.copy()
    result[grid > 0] = [0, 0, 0]  # чёрные линии
    return result


# def remove_noise_and_speckles(image: np.ndarray) -> np.ndarray:
#     """
#     Улучшенный алгоритм удаления мелкого шума, пятен и грязи:
#     - работает на адаптивной бинаризации (лучше при засветке),
#     - сохраняет мелкий текст,
#     - удаляет только нерелевантные артефакты.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Сглаживание для подавления локального шума
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Адаптивная бинаризация (инвертированная: текст = белый, фон = черный)
#     bin_img = cv2.adaptiveThreshold(blur, 255,
#                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                     cv2.THRESH_BINARY_INV,
#                                     blockSize=21, C=15)

#     # Морфологическая очистка
#     cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,
#                                 cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
#                                 iterations=1)

#     # Контуры → фильтруем мусор
#     contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     mask = np.zeros_like(gray)

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         area = cv2.contourArea(cnt)

#         if area > 20 and w > 3 and h > 3:
#             cv2.drawContours(mask, [cnt], -1, 255, -1)

#     # Сохраняем только значимые участки (оставшееся — делаем белым)
#     result = image.copy()
#     result[mask == 0] = [255, 255, 255]
#     return result


def remove_stamp_candidates(image: np.ndarray) -> np.ndarray:
    """Удаляет круглые элементы (печати)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=30, param1=50, param2=30,
                               minRadius=10, maxRadius=100)
    
    output = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r + 5, (255, 255, 255), -1)

    return output

# def remove_handwritten_notes(image: np.ndarray) -> np.ndarray:
#     """Удаляет рукописные пометки и подписи"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     inv = 255 - gray
#     _, binarized = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     morph = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel, iterations=1)

#     contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cleaned = image.copy()
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         if 5 < w < 80 and 5 < h < 40:
#             aspect = w / h
#             if aspect < 2.5:
#                 cv2.rectangle(cleaned, (x, y), (x + w, y + h), (255, 255, 255), -1)
#     return cleaned

def fade_blue_stamp(image: np.ndarray) -> np.ndarray:
    """Ослабляет синие печати без удаления текста под ними"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    if np.count_nonzero(mask) == 0:
        return image  # нет синего — ничего не делаем

    faded = image.copy()
    blue_pixels = faded[mask > 0]

    # Ослабляем только те пиксели, которые попали в маску
    lightened = cv2.addWeighted(blue_pixels.astype(np.uint8), 0.2,
                                np.full_like(blue_pixels, 255), 0.8, 0)

    faded[mask > 0] = lightened
    return faded




def remove_irregular_contours(image: np.ndarray) -> np.ndarray:
    """Удаляет подписи и печати на основе неправильной формы контуров"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = image.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        rect_area = w * h
        if rect_area == 0:
            continue

        extent = area / rect_area
        if extent < 0.25 and area > 300:
            cv2.drawContours(cleaned, [cnt], -1, (255, 255, 255), -1)

    return cleaned

def crop_content_region(image: np.ndarray, padding: int = 20) -> np.ndarray:
    """Обрезает поля вокруг текста."""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + 
                            cv2.THRESH_OTSU)[1]

    coords = cv2.findNonZero(bin_img)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1], x + w + padding)
    h = min(image.shape[0], y + h + padding)
    cropped = image[y:h, x:w]
    return cropped

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Комбинированная предобработка изображения для OCR"""
    step0 = enhance_contrast(image)
    step1 = enhance_table_lines(step0)
    # step1 = remove_stamp_candidates(step0)
    # step1 = remove_noise_and_speckles(step0)
    step2 = crop_content_region(step1)
    # step3 = reconstruct_table_grid(step2)
    # step3 = fade_blue_stamp(step2)
    #step4 = remove_irregular_contours(step3)
    # step1 = remove_noise_and_speckles(step0)
    # step2 = remove_stamp_candidates(step1)
    # step3 = remove_handwritten_notes(step2)
    # step4 = crop_content_region(step3)
    return step2


def process_images(input_dir: str, output_dir: str):
    """Обрабатывает все изображения в указанной директории."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            continue
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Обработка: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠ Ошибка загрузки: {input_path}")
            continue
        
        # Предварительная обработка
        processed_image = preprocess_image(image)
        
        cv2.imwrite(output_path, processed_image)
        print(f"✅ Сохранено: {output_path}")

if __name__ == "__main__":
    input_directory = "data/reports/corrected_images"
    output_directory = "data/reports/preprocessed_images"
    process_images(input_directory, output_directory)
