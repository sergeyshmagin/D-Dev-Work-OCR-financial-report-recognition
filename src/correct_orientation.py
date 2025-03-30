import cv2
import numpy as np
import os

def correct_orientation(image: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """Корректирует ориентацию страницы и выравнивает текст без кадрирования"""
    # Шаг 1: Определяем исходную ориентацию
    is_portrait = image.shape[0] > image.shape[1]
    
    # Шаг 2: Предварительная обработка для детекции текста
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # Шаг 3: Находим линии текста
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=100, maxLineGap=10
    )
    
    if lines is None:
        return image
    
    # Шаг 4: Анализируем углы наклона текста
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    # Фильтруем только значительные углы
    significant_angles = [a for a in angles if abs(a) > threshold]
    
    if not significant_angles:
        return image  # Наклон текста недостаточен для коррекции
    
    median_angle = np.median(significant_angles)
    
    # Шаг 5: Первичный поворот для перевода в альбомную ориентацию
    if is_portrait and abs(median_angle) > threshold:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        print(f"Перевод в альбомную ориентацию (исходный наклон {median_angle:.2f}°)")
        # Обновляем параметры после поворота
        is_portrait = False
        # Пересчитываем углы для нового положения
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        median_angle = np.median(angles)
    
    # Шаг 6: Поворот с сохранением исходного размера страницы
    rotation_angle = median_angle
    print(f"Точное выравнивание: поворот на {rotation_angle:.2f}°")
    
    # Получаем размеры изображения
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Вычисляем матрицу поворота
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Вычисляем новые размеры изображения после поворота
    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Корректируем матрицу поворота для центрирования
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Применяем поворот с белыми границами
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[255, 255, 255]  # Белый цвет для новых областей
    )
    
    return rotated

def process_images(input_dir: str, output_dir: str):
    """Обрабатывает все изображения в папке"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            continue
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\nОбработка: {input_path}")
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"⚠ Ошибка загрузки: {input_path}")
            continue
        
        corrected = correct_orientation(image)
        cv2.imwrite(output_path, corrected)
        print(f"Сохранено: {output_path}")

if __name__ == "__main__":
    input_dir = "data/reports/intermediate_images"
    output_dir = "data/reports/corrected_images"
    process_images(input_dir, output_dir)