"""
тестовый код для получение предварительной статистики по пересекающимся объектам в рамках общего bounding box для нескольких объектов - результат ошибки маскирования
написан для анализа где алгоритм ошибается - получим размеры объектов, характер перекрытий
"""
import os
import time
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import json
from datetime import datetime


_object_counter = 0

# ========== 1. ЗАГРУЗКА МОДЕЛИ ==========
def load_model(weights_path='./weights/deeplabv3_resnet101_coco-586e9e4e.pth', 
               backbone_path='./backbone/resnet101-63fe2227.pth'):
    """Загружает модель DeepLabV3 из локальных файлов"""
    print("Загрузка модели...")
    
    original_load_state_dict_from_url = torch.hub.load_state_dict_from_url
    
    def local_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
        if 'resnet101-63fe2227.pth' in url:
            print(f"Используем локальный backbone: {backbone_path}")
            return torch.load(backbone_path, map_location=map_location)
        return original_load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
    
    torch.hub.load_state_dict_from_url = local_load_state_dict_from_url
    model = deeplabv3_resnet101(weights=None, aux_loss=True)
    torch.hub.load_state_dict_from_url = original_load_state_dict_from_url
    
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
PERSON_ID = 15
CAR_IDS = {2, 3, 4, 5, 6, 7, 8}
CLASS_NAMES = {
    15: 'person',
    2: 'car',
    3: 'bus',
    5: 'truck',
    6: 'motorcycle',
    7: 'airplane',
    8: 'bicycle'
}

# ========== 3. ЛОГГЕР ==========
class FrameLogger:
    """Сбор и сохранение данных о кадрах"""
    def __init__(self, log_dir='./debug_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.frames_data = []
        self.frame_count = 0
        
    def log_frame(self, frame_num, detections, masks_info, frame_shape):
        """Логирует информацию о кадре"""
        log_entry = {
            'frame': frame_num,
            'timestamp': datetime.now().isoformat(),
            'shape': {'height': frame_shape[0], 'width': frame_shape[1]},
            'num_detections': len(detections),
            'detections': [],
            'masks': []
        }
        
        # Логируем детекции
        for det in detections:
            log_entry['detections'].append({
                'bbox': [int(x) for x in det[:4]],
                'class_id': int(det[4]),
                'class_name': CLASS_NAMES.get(int(det[4]), 'unknown'),
                'confidence': float(det[5]) if len(det) > 5 else 1.0,
                'width': int(det[2] - det[0]),
                'height': int(det[3] - det[1]),
                'aspect_ratio': float((det[2] - det[0]) / (det[3] - det[1] + 1e-6)),
                'area': int((det[2] - det[0]) * (det[3] - det[1]))
            })
        
        # Логируем информацию о масках
        for class_id, info in masks_info.items():
            log_entry['masks'].append({
                'class_id': int(class_id),
                'class_name': CLASS_NAMES.get(int(class_id), 'unknown'),
                'num_contours': info['num_contours'],
                'total_area': int(info['total_area']),
                'max_contour_area': int(info['max_contour_area']),
                'bbox': [int(x) for x in info['bbox']]
            })
        
        self.frames_data.append(log_entry)
        self.frame_count += 1
    
    def save_log(self):
        """Сохраняет лог в JSON"""
        filename = f"{self.log_dir}/debug_log_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(self.frames_data, f, indent=2)
        print(f"\nЛог сохранен: {filename}")
        return filename
    
    def print_statistics(self):
        """Выводит статистику по собранным данным"""
        if not self.frames_data:
            print("Нет данных")
            return
        
        total_detections = sum(f['num_detections'] for f in self.frames_data)
        avg_detections = total_detections / len(self.frames_data)
        
        # Собираем статистику по размерам
        all_areas = []
        all_aspects = []
        class_counts = {}
        
        for frame in self.frames_data:
            for det in frame['detections']:
                all_areas.append(det['area'])
                all_aspects.append(det['aspect_ratio'])
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\n" + "="*60)
        print("СТАТИСТИКА ПО КАДРАМ")
        print("="*60)
        print(f"Всего кадров: {len(self.frames_data)}")
        print(f"Всего детекций: {total_detections}")
        print(f"Среднее объектов на кадр: {avg_detections:.2f}")
        print(f"\nРаспределение по классам:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {class_name}: {count} ({count/total_detections*100:.1f}%)")
        
        if all_areas:
            print(f"\nРазмеры объектов (area):")
            print(f"  Мин: {min(all_areas)}")
            print(f"  Макс: {max(all_areas)}")
            print(f"  Средний: {sum(all_areas)/len(all_areas):.0f}")
            print(f"  Медиана: {sorted(all_areas)[len(all_areas)//2]}")
        
        if all_aspects:
            print(f"\nСоотношение сторон (w/h):")
            print(f"  Мин: {min(all_aspects):.2f}")
            print(f"  Макс: {max(all_aspects):.2f}")
            print(f"  Средний: {sum(all_aspects)/len(all_aspects):.2f}")
            print(f"  Медиана: {sorted(all_aspects)[len(all_aspects)//2]:.2f}")
        
        # Статистика для людей
        person_aspects = [d['aspect_ratio'] for f in self.frames_data 
                         for d in f['detections'] if d['class_name'] == 'person']
        if person_aspects:
            print(f"\nСоотношение сторон для ЛЮДЕЙ:")
            print(f"  Диапазон: {min(person_aspects):.2f} - {max(person_aspects):.2f}")
            print(f"  Средний: {sum(person_aspects)/len(person_aspects):.2f}")
        
        # Статистика для машин
        car_aspects = [d['aspect_ratio'] for f in self.frames_data 
                      for d in f['detections'] if d['class_name'] == 'car']
        if car_aspects:
            print(f"\nСоотношение сторон для МАШИН:")
            print(f"  Диапазон: {min(car_aspects):.2f} - {max(car_aspects):.2f}")
            print(f"  Средний: {sum(car_aspects)/len(car_aspects):.2f}")
        print("="*60)

# ========== 4. ИЗВЛЕЧЕНИЕ ОБЪЕКТОВ С ЛОГИРОВАНИЕМ ==========
def extract_objects_with_logging(mask, frame_shape, frame_num, logger):
    """
    Извлекает объекты с логированием информации о масках
    """
    global _object_counter

    height, width = frame_shape[:2]
    detections = []
    masks_info = {}
    
    # Получаем уникальные классы
    unique_classes = np.unique(mask)
    
    for class_id in unique_classes:
        if class_id == 0:
            continue
        
        # Бинарная маска для класса
        class_mask = (mask == class_id).astype(np.uint8)
        
        # Находим контуры
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Собираем информацию о маске
        total_area = sum(cv2.contourArea(c) for c in contours)
        max_area = max(cv2.contourArea(c) for c in contours)
        
        # Общий bbox для класса
        all_points = np.vstack([c.reshape(-1, 2) for c in contours])
        class_bbox = [all_points[:, 0].min(), all_points[:, 1].min(),
                     all_points[:, 0].max(), all_points[:, 1].max()]
        
        masks_info[class_id] = {
            'num_contours': len(contours),
            'total_area': total_area,
            'max_contour_area': max_area,
            'bbox': class_bbox
        }
        
        # Обрабатываем каждый контур
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Динамический порог
            if area < 200:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 10 or h < 10:
                continue
            
            aspect_ratio = w / h
            
            _object_counter += 1
            object_id = _object_counter

            # Логируем каждый контур отдельно
            detections.append([x, y, x + w, y + h, class_id, 1.0, object_id])
            
            # Вывод в консоль для отладки
            print(f"  [Frame {frame_num}] {object_id} {CLASS_NAMES.get(class_id, 'unknown')}: "
                  f"area={area:.0f}, bbox=({w}x{h}), ratio={aspect_ratio:.2f}")
    
    # Логируем информацию о маске
    logger.log_frame(frame_num, detections, masks_info, frame_shape)
    
    return detections

# ========== 5. ТЕСТОВАЯ ОБРАБОТКА ==========
def test_processing(input_video, model, num_frames=50, save_debug_images=True):
    """
    Тестовая обработка первых N кадров с логированием
    """
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {input_video}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nПАРАМЕТРЫ ВИДЕО")
    print(f"   Размер: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Всего кадров: {total_frames}")
    print(f"   Тестовых кадров: {min(num_frames, total_frames)}")
    print("\n" + "="*60)
    
    # Подготовка
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    logger = FrameLogger()
    debug_dir = './debug_frames'
    os.makedirs(debug_dir, exist_ok=True)
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Сегментация
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(frame_rgb).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        mask = torch.argmax(output, dim=0).byte().cpu().numpy()
        
        # Извлекаем объекты с логированием
        print(f"\nКадр {frame_count}/{num_frames}")
        detections = extract_objects_with_logging(mask, frame.shape, frame_count, logger)
        
        print(f"   Найдено объектов: {len(detections)}")
        
        # Сохраняем кадр с визуализацией
        if save_debug_images and detections:
            debug_frame = frame.copy()
            
            # Рисуем bbox
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det[:4]]
                class_id = int(det[4])
                object_id = int(det[6]) if len(det) > 6 else 0  
                color = (0, 255, 0) if class_id == PERSON_ID else (0, 0, 255)
                
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 1)
                
                label = f"#{object_id} {CLASS_NAMES.get(class_id, 'unknown')}"
                cv2.putText(debug_frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Информация о кадре
            cv2.putText(debug_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Сохраняем
            debug_path = f"{debug_dir}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(debug_path, debug_frame)
            print(f" Сохранен: {debug_path}")
        
        # Прогресс
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Время: {elapsed:.1f}с, скорость: {frame_count/elapsed:.1f} FPS")
    
    cap.release()
    
    # Сохраняем лог
    log_file = logger.save_log()
    
    # Выводим статистику
    logger.print_statistics()
    
    print(f"\nТест завершен!")
    print(f"   Обработано кадров: {frame_count}")
    print(f"   Лог сохранен: {log_file}")
    print(f"   Изображения сохранены в: {debug_dir}/")
    
    return logger, log_file

# ========== 6. ЗАПУСК ==========
if __name__ == "__main__":
    # Настройки
    FILE_NAME = "spb-cam1-short-001"
    INPUT_VIDEO = f"data/input/{FILE_NAME}.mp4"
    WEIGHTS_PATH = "./weights/deeplabv3_resnet101_coco-586e9e4e.pth"
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"Видео не найдено: {INPUT_VIDEO}")
        exit(1)
    
    # Загрузка модели
    model = load_model(WEIGHTS_PATH)
    
    # Тестовая обработка (первые 100 кадров)
    print("\nЗАПУСК ТЕСТОВОГО АНАЛИЗА")
    print("="*60)
    
    logger, log_file = test_processing(
        input_video=INPUT_VIDEO,
        model=model,
        num_frames=5,  # анализируем первые 100 кадров
        save_debug_images=True
    )
    
    print("\n" + "="*60)
    print("ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
    print("1. Откройте сохраненные изображения в папке debug_frames/")
    print("2. Проверьте, правильно ли выделены объекты")
    print("3. Посмотрите на статистику в логе")
    print("4. Сообщите о проблемных случаях для настройки")
    print("="*60)