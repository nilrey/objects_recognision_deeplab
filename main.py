import os
import time
import torch
import cv2
import numpy as np
import torch.hub
from torchvision import transforms
from PIL import Image

from torchvision.models.segmentation import deeplabv3_resnet101

def load_model(weights_path='./weights/deeplabv3_resnet101_coco-586e9e4e.pth', backbone_path='./backbone/resnet101-63fe2227.pth'):
    """
    Загружает модель DeepLabV3 из локальных файлов
    """
    print("Загрузка модели...")
    
    # Переопределяем функцию загрузки для использования локального backbone
    original_load_state_dict_from_url = torch.hub.load_state_dict_from_url
    
    def local_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
        if 'resnet101-63fe2227.pth' in url:
            print(f"Используем локальный backbone: {backbone_path}")
            return torch.load(backbone_path, map_location=map_location)
        return original_load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
    
    torch.hub.load_state_dict_from_url = local_load_state_dict_from_url
    
    # Создаем модель
    model = deeplabv3_resnet101(weights=None, aux_loss=True)
    
    # Восстанавливаем оригинальную функцию
    torch.hub.load_state_dict_from_url = original_load_state_dict_from_url
    
    # Загружаем веса DeepLabV3
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Веса DeepLabV3 загружены из {weights_path}")
    else:
        print(f"Файл весов не найден: {weights_path}")
        exit(1)
    
    model.eval()
    return model

# ========== 2. КОНСТАНТЫ ==========
# COCO классы
PERSON_ID = 15
CAR_IDS = {2, 3, 4, 5, 6, 7, 8}  # машина, автобус, грузовик, мотоцикл, велосипед и т.д.

# ========== 3. ОБРАБОТКА ВИДЕО С ПРОГРЕССОМ ==========
def process_video(input_video, output_video, model):
    """
    Обрабатывает видео с отображением прогресса в процентах
    """
    # Открываем видео
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {input_video}")
        return
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nПараметры видео:")
    print(f"   Размер: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Всего кадров: {total_frames}")
    print(f"   Примерное время: ~{total_frames // 10} секунд (при 10 FPS)")
    print("\nНачинаем обработку...\n")
    
    # Создаем видеописатель
    out = cv2.VideoWriter(output_video, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
    
    # Подготовка трансформаций для модели
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Переменные для прогресса
    frame_count = 0
    start_time = time.time()
    last_percent = -1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ========== ВЫЧИСЛЕНИЕ ПРОГРЕССА ==========
        percent = int((frame_count / total_frames) * 100)
        
        # Выводим прогресс только при изменении процента (каждый 1%)
        if percent != last_percent:
            # Расчет ETA (оставшегося времени)
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            remaining_frames = total_frames - frame_count
            eta = remaining_frames / fps_processing if fps_processing > 0 else 0
            
            # Форматируем время
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            
            # Создаем прогресс-бар
            bar_length = 30
            filled = int(bar_length * frame_count / total_frames)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r[{bar}] {percent}% | Кадры: {frame_count}/{total_frames} | "
                  f"Скорость: {fps_processing:.1f} FPS | "
                  f"Осталось: {eta_min}:{eta_sec:02d}", end='', flush=True)
            
            last_percent = percent
        
        # ========== СЕГМЕНТАЦИЯ КАДРА ==========
        # Конвертируем BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Инференс модели
        input_tensor = preprocess(frame_rgb).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        # Получаем маску классов
        mask = torch.argmax(output, dim=0).byte().cpu().numpy()
        
        people_mask = (mask == PERSON_ID)
        frame[people_mask] = [255, 0, 0]
        
        cars_mask = np.isin(mask, list(CAR_IDS))
        frame[cars_mask] = [0, 0, 255]
        
        # Записываем кадр
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Финальная статистика
    elapsed_total = time.time() - start_time
    print(f"\n\nОбработка завершена!")
    print(f"   Всего кадров: {frame_count}")
    print(f"   Время: {elapsed_total:.1f} секунд")
    print(f"   Средняя скорость: {frame_count/elapsed_total:.1f} FPS")
    print(f"   Результат сохранен: {output_video}")


if __name__ == "__main__":
    # Настройки
    FILE_NAME = "spb-cam1-short-001"
    INPUT_VIDEO = f"data/input/{FILE_NAME}.mp4"
    OUTPUT_VIDEO = f"data/output/out-{FILE_NAME}_{time.time()}.mp4"   
    WEIGHTS_PATH = "./weights/deeplabv3_resnet101_coco-586e9e4e.pth"  # 
    
    # Проверка существования входного файла
    if not os.path.exists(INPUT_VIDEO):
        print(f"Видео не найдено: {INPUT_VIDEO}")
        print(f"Положите видео файл '{INPUT_VIDEO}' в папку проекта")
        exit(1)
    
    # Загрузка модели
    model = load_model(WEIGHTS_PATH)
    
    # Обработка видео
    process_video(INPUT_VIDEO, OUTPUT_VIDEO, model)