import PyPDF2
import os
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение переменных окружения
reports_dir = os.getenv("REPORTS_DIR")
poppler_path = os.getenv("POPLER_PATH")  # Получаем путь до Poppler

# Установка переменной окружения для Poppler
if poppler_path:
    os.environ["PATH"] += os.pathsep + poppler_path

# Промежуточный каталог для сохранения изображений
intermediate_dir = os.path.join(reports_dir, 'intermediate_images')
os.makedirs(intermediate_dir, exist_ok=True)  # Создаем каталог, если он не существует


def convert_pdf_to_images(pdf_path, output_dir):
    """
    Конвертирует весь PDF-файл в изображения и сохраняет их в указанной директории.

    :param pdf_path: Путь к PDF-файлу.
    :param output_dir: Директория для сохранения изображений.
    """
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f'page_{i + 1}.png')
        image.save(image_path, 'PNG')
        print(f"Сохранено изображение: {image_path}")

if __name__ == "__main__":
    # Получаем список всех PDF-файлов в директории
    pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("Нет PDF-файлов в директории.")
    else:
        # Выбираем первый PDF-файл
        input_pdf = os.path.join(reports_dir, pdf_files[0])

        # Конвертация всего документа в изображения
        convert_pdf_to_images(input_pdf, intermediate_dir)
