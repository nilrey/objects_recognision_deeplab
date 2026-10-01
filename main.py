import os
import time
import torch
import cv2
import numpy as np
import torch.hub
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
from collections import deque

from torchvision.models.segmentation import deeplabv3_resnet101

# ========== 1. ТРЕКЕР (без внешних зависимостей) ==========
class ObjectTracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        self.frame_count += 1
        
        # Если нет детекций, обновляем время жизни
        if len(detections) == 0:
            for track in self.tracks:
                track['age'] += 1
            self._remove_old_tracks()
            return self._get_active_tracks()
        
        # Если нет трекеров, создаем новые
        if len(self.tracks) == 0:
            for det in detections:
                self._create_track(det)
            return self._get_active_tracks()
        
        # Вычисляем матрицу IoU
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track['bbox'], det[:4])
        
        # Используем венгерский алгоритм для оптимального сопоставления
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        matched_tracks = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Обрабатываем совпадения
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] > self.iou_threshold:
                matched_tracks.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        # Обновляем совпавшие трекеры
        for row, col in matched_tracks:
            self.tracks[row]['bbox'] = detections[col][:4]
            self.tracks[row]['class_id'] = detections[col][4]
            self.tracks[row]['confidence'] = detections[col][5] if len(detections[col]) > 5 else 1.0
            self.tracks[row]['age'] = 0
            self.tracks[row]['hits'] += 1
            self.tracks[row]['color'] = self._get_color(self.tracks[row]['class_id'])
            self.tracks[row]['history'].append(self._get_center(detections[col][:4]))
        
        # Создаем новые трекеры для несовпавших детекций
        for col in unmatched_detections:
            self._create_track(detections[col])
        
        # Обновляем возраст несовпавших трекеров
        for row in unmatched_tracks:
            self.tracks[row]['age'] += 1
        
        # Удаляем старые трекеры
        self._remove_old_tracks()
        
        return self._get_active_tracks()
    
    def _create_track(self, det):
        """Создаем новый трек"""
        track = {
            'id': self.next_id,
            'bbox': det[:4],
            'class_id': int(det[4]),
            'confidence': det[5] if len(det) > 5 else 1.0,
            'age': 0,
            'hits': 1,
            'color': self._get_color(int(det[4])),
            'history': deque(maxlen=30)
        }
        track['history'].append(self._get_center(det[:4]))
        self.tracks.append(track)
        self.next_id += 1
    
    def _get_active_tracks(self):
        """Получаем активные треки"""
        return [t for t in self.tracks if t['hits'] >= self.min_hits and t['age'] <= self.max_age]
    
    def _remove_old_tracks(self):
        """Удаляет старые треки"""
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
    
    def _compute_iou(self, bbox1, bbox2):
        """Вычисляет IoU между двумя bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _get_center(self, bbox):
        """Возвращает центр bbox"""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def _get_color(self, class_id):
        """Возвращает цвет для класса"""
        colors = {
            15: (0, 255, 0),     # человек - зеленый
            2: (0, 0, 255),      # машина - красный
            3: (255, 0, 0),      # автобус - синий
            5: (255, 255, 0),    # грузовик - голубой
            6: (255, 0, 255),    # мотоцикл - фиолетовый
            7: (0, 255, 255),    # самолет - желтый
            8: (128, 0, 128),    # велосипед - пурпурный
        }
        return colors.get(class_id, (128, 128, 128))

# ========== 2. ЗАГРУЗКА МОДЕЛИ ==========
def load_model(weights_path='./weights/deeplabv3_resnet101_coco-586e9e4e.pth', backbone_path='./backbone/resnet101-63fe2227.pth'):
    """
    Загружает модель DeepLabV3 из локальных файлов
    """
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

# ========== 3. КОНСТАНТЫ ==========
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

# ========== 4. ИЗВЛЕЧЕНИЕ ОБЪЕКТОВ ИЗ МАСКИ ==========
def extract_objects_from_mask(mask, frame_shape):
    """
    Извлекает bounding boxes из маски сегментации
    Возвращает список [x1, y1, x2, y2, class_id, confidence]
    """
    height, width = frame_shape[:2]
    detections = []
    
    # Получаем уникальные классы в маске
    unique_classes = np.unique(mask)
    
    for class_id in unique_classes:
        # Пропускаем фон (0)
        if class_id == 0:
            continue
            
        # Создаем бинарную маску для класса
        class_mask = (mask == class_id).astype(np.uint8)
        
        # Находим контуры
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Игнорируем слишком маленькие объекты
            if area < 500:  # Увеличил порог для уменьшения шума
                continue
            
            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Проверяем, что bbox не слишком мал
            if w < 15 or h < 15:
                continue
            
            # Проверяем соотношение сторон (для фильтрации артефактов)
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            # Добавляем детекцию
            detections.append([x, y, x + w, y + h, class_id, 1.0])
    
    return detections

# ========== 5. ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ==========
def process_video_with_tracking(input_video, output_video, model):
    """
    Обрабатывает видео с детекцией bounding boxes и трекингом
    """
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {input_video}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nПараметры видео:")
    print(f"   Размер: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Всего кадров: {total_frames}")
    
    out = cv2.VideoWriter(output_video, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    tracker = ObjectTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    
    frame_count = 0
    start_time = time.time()
    last_percent = -1
    total_objects = set()  # Для подсчета уникальных объектов
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Прогресс
        percent = int((frame_count / total_frames) * 100)
        if percent != last_percent:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            remaining_frames = total_frames - frame_count
            eta = remaining_frames / fps_processing if fps_processing > 0 else 0
            
            bar_length = 30
            filled = int(bar_length * frame_count / total_frames)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r[{bar}] {percent}% | Кадры: {frame_count}/{total_frames} | "
                  f"Скорость: {fps_processing:.1f} FPS | "
                  f"Осталось: {int(eta//60)}:{int(eta%60):02d}", end='', flush=True)
            
            last_percent = percent
        
        # ===== СЕГМЕНТАЦИЯ =====
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(frame_rgb).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        mask = torch.argmax(output, dim=0).byte().cpu().numpy()
        
        # ===== ИЗВЛЕЧЕНИЕ ОБЪЕКТОВ =====
        detections = extract_objects_from_mask(mask, frame.shape)
        
        # ===== ТРЕКИНГ =====
        tracked_objects = tracker.update(detections)
        
        # Обновляем счетчик уникальных объектов
        for obj in tracked_objects:
            total_objects.add(obj['id'])
        
        # ===== ВИЗУАЛИЗАЦИЯ =====
        # Полупрозрачные маски
        people_mask = (mask == PERSON_ID)
        frame[people_mask] = frame[people_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        
        cars_mask = np.isin(mask, list(CAR_IDS))
        frame[cars_mask] = frame[cars_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
        
        # Рисуем bounding boxes
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
            class_id = obj['class_id']
            track_id = obj['id']
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
            color = obj['color']
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Фон для текста
            label = f'#{track_id} {class_name}'
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, 
                         (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), 
                         color, -1)
            
            # Текст
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Траектория движения
            if len(obj['history']) > 2:
                points = list(obj['history'])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
                    # Рисуем маленькие точки на траектории
                    cv2.circle(frame, points[i], 2, color, -1)
        
        # ===== ИНФОРМАЦИОННАЯ ПАНЕЛЬ =====
        info_lines = [
            f"Objects: {len(tracked_objects)}",
            f"Total unique: {len(total_objects)}",
            f"Frame: {frame_count}/{total_frames}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ===== СОХРАНЕНИЕ =====
        out.write(frame)
    
    cap.release()
    out.release()
    
    elapsed_total = time.time() - start_time
    print(f"\n\nОбработка завершена!")
    print(f"   Всего кадров: {frame_count}")
    print(f"   Уникальных объектов: {len(total_objects)}")
    print(f"   Время: {elapsed_total:.1f} секунд")
    print(f"   Средняя скорость: {frame_count/elapsed_total:.1f} FPS")
    print(f"   Результат сохранен: {output_video}")

# ========== 6. ЗАПУСК ==========
if __name__ == "__main__":
    FILE_NAME = "spb-cam1-short-001"
    INPUT_VIDEO = f"data/input/{FILE_NAME}.mp4"
    OUTPUT_VIDEO = f"data/output/out-{FILE_NAME}_{time.time()}.mp4"   
    WEIGHTS_PATH = "./weights/deeplabv3_resnet101_coco-586e9e4e.pth"
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"Видео не найдено: {INPUT_VIDEO}")
        exit(1)
    
    model = load_model(WEIGHTS_PATH)
    process_video_with_tracking(INPUT_VIDEO, OUTPUT_VIDEO, model)