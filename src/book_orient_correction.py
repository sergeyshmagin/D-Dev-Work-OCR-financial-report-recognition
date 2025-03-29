import cv2
import numpy as np
import os
import math

class UniversalAligner:
    def __init__(self):
        self.input_dir = os.path.join("data", "reports", "intermediate_images")
        self.output_dir = os.path.join("data", "reports", "aligned")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.params = {
            'min_table_area': 30000,
            'line_threshold': 100,
            'max_skew_angle': 15,
            'min_line_length': 0.3
        }

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Не удалось загрузить: {path}")
        return img

    def should_rotate(self, img):
        """Определяет, нужно ли поворачивать изображение"""
        h, w = img.shape[:2]
        aspect_ratio = h / w
        
        # Если изображение явно вертикальное (высота > ширины в 1.5 раза)
        # И содержит горизонтальный текст (определяем по линиям)
        if aspect_ratio > 1.5:
            # Анализируем ориентацию текста
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                  minLineLength=w//2, 
                                  maxLineGap=20)
            
            if lines is not None:
                horizontal = 0
                vertical = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x1 - x2) < 10:  # Вертикальная линия
                        vertical += 1
                    elif abs(y1 - y2) < 10:  # Горизонтальная линия
                        horizontal += 1
                
                # Если преобладают вертикальные линии - вероятно нужно повернуть
                if vertical > horizontal * 1.5:
                    return True
        
        return False

    def detect_table_lines(self, img):
        """Обнаружение линий таблицы"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        
        min_line_length = int(img.shape[1] * self.params['min_line_length'])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.params['line_threshold'],
                              minLineLength=min_line_length,
                              maxLineGap=20)
        return lines if lines is not None else []

    def calculate_skew_angle(self, lines, img_width):
        """Вычисление угла наклона"""
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            if abs(angle) < self.params['max_skew_angle']:
                angles.append(angle)
        
        return np.median(angles) if angles else 0.0

    def rotate_image(self, img, angle):
        """Поворот изображения"""
        if angle == 0:
            return img
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(img, M, (new_w, new_h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    def process_image(self, filename):
        """Обработка изображения"""
        try:
            input_path = os.path.join(self.input_dir, filename)
            output_path = os.path.join(self.output_dir, f"aligned_{filename}")
            
            img = self.load_image(input_path)
            
            # 1. Проверка необходимости поворота
            if self.should_rotate(img):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            # 2. Детекция линий и выравнивание
            lines = self.detect_table_lines(img)
            skew_angle = self.calculate_skew_angle(lines, img.shape[1])
            
            if abs(skew_angle) > 0.5:
                final_img = self.rotate_image(img, skew_angle)
            else:
                final_img = img
            
            cv2.imwrite(output_path, final_img)
            print(f"Обработан: {filename} | Коррекция наклона: {skew_angle:.2f}°")
            return True
            
        except Exception as e:
            print(f"Ошибка обработки {filename}: {str(e)}")
            return False

    def process_all(self):
        """Обработка всех изображений"""
        if not os.path.exists(self.input_dir):
            print(f"Директория не найдена: {self.input_dir}")
            return

        success = 0
        for filename in sorted(os.listdir(self.input_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if self.process_image(filename):
                    success += 1
        
        print(f"\nГотово! Успешно обработано: {success} файлов")

if __name__ == "__main__":
    print("=== Универсальный Document Aligner ===")
    print("Автоматическое выравнивание таблиц в документах\n")
    
    aligner = UniversalAligner()
    aligner.process_all()