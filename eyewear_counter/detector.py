import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .utils import load_pt_from_url


class YoloDetector:
    """
    Обёртка над моделью YOLO для обнаружения лиц.
    """
    def __init__(self, model_path=None, device=None, threshold=0.7, max_faces=4, detect_size=640):
        """
        Args:
            model_path (str): Путь к YOLO модели.
            device (torch.device): 'cuda' или 'cpu'
            threshold (float): Порог уверенности детекции.
            max_faces (int): Максимальное число лиц на изображение.
        """
        from ultralytics import YOLO
        import logging

        logging.getLogger('ultralytics').setLevel(logging.WARNING)


        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        if model_path is None:
            model_path = Path(__file__).parent.parent / "weights" / "yolov11n-face.pt"
            if not model_path.is_file():
                model_url = "https://github.com/qksolov/eyewear-counter/raw/main/weights/yolov11n-face.pt"
                model_path = load_pt_from_url(model_url)
        else:
            model_path = Path(model_path)
        
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель YOLO из {model_path}: {e}")
        
        self.threshold = threshold
        self.max_faces = max_faces
        self.input_type = "torch"
        self.detect_size = detect_size


    def update_parameters(self, threshold=0.7, max_faces=4):
        self.threshold = threshold
        self.max_faces = max_faces


    def detect(self, images_tensor):
        """
        Args:
            images_tensor (torch.Tensor): Батч изображений shape=(B, 3, H, W),
                                         dtype=float32, диапазон [0, 1], RGB
        
        Returns:
            torch.Tensor: Тензор обнаруженных лиц shape=(N, 5), 
                         где каждая строка: (batch_idx, x1, y1, x2, y2)
                         batch_idx - индекс изображения в батче
        """
        if images_tensor.ndim != 4 or images_tensor.shape[1] != 3:
            raise ValueError(
                f"Неверная форма входного тензора. Ожидается (B, 3, H, W), "
                f"получено {tuple(images_tensor.shape)}. "
            )

        with torch.no_grad():
            B, _, H, W = images_tensor.shape
            
            if max(H, W) > self.detect_size:
                resized_tensor = F.interpolate(images_tensor, size=(self.detect_size, self.detect_size), mode="bilinear", align_corners=False)
                scale_x = W / self.detect_size
                scale_y = H / self.detect_size
            else:
                resized_tensor = images_tensor
                scale_x = 1
                scale_y = 1

            all_faces = self.model.predict(
                resized_tensor,
                conf=self.threshold,
                max_det=self.max_faces,
                verbose=False, save=False, save_txt=False, save_conf=False, save_crop=False
                )

            rois = torch.zeros((B * self.max_faces, 5), 
                               dtype=torch.float32, device=self.device)

            index = 0
            for i, faces in enumerate(all_faces):
                if faces.boxes.shape[0] > 0:
                    boxes = faces.boxes.xyxy.clone()
                    boxes[:, [0, 2]] *= scale_x  # x1, x2
                    boxes[:, [1, 3]] *= scale_y  # y1, y2

                    n = boxes.shape[0]
                    rois[index:index + n, 0] = i
                    rois[index:index + n, 1:] = boxes
                    index += n

        return rois[:index]


class RetinaFaceDetector:
    """
    Обертка над моделью RetinaFace для обнаружения лиц.
    """
    def __init__(self, device=None, threshold=0.8, max_faces=4):
        """
        Args:
            device (torch.device): 'cuda' или 'cpu'
            threshold (float): Порог уверенности детекции.
            max_faces (int): Максимальное число лиц на изображение.
        """
        try:
            from batch_face import RetinaFace
        except ImportError as e:
            raise ImportError(
                "Для использования RetinaFace необходимо установить batch-face:\n"
                "pip install batch-face\n"
            ) from e
        

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.input_type = "np"
        self.model = RetinaFace(gpu_id=0 if self.device.type == 'cuda' else -1)        
        self.threshold = threshold
        self.max_faces = max_faces


    def update_parameters(self, threshold=0.8, max_faces=4):
        self.threshold = threshold
        self.max_faces = max_faces


    def detect(self, images_np):
        """
        Args:
            images_np (np.ndarray): Батч изображений shape=(B, H, W, 3), dtype=uint8, RGB.
        
        Returns:
            torch.Tensor: Тензор обнаруженных лиц shape=(N, 5), 
                         где каждая строка: (batch_idx, x1, y1, x2, y2)
                         batch_idx - индекс изображения в батче
        """
        if images_np.ndim != 4 or images_np.shape[3] != 3:
            raise ValueError(
                f"Неверная форма входного тензора. Ожидается (B, 3, H, W), "
                f"получено {tuple(images_np.shape)}. "
            )
        
        with torch.no_grad():
            all_faces = self.model(images_np, threshold=self.threshold, resize=1, max_size=-1)
            num_images = images_np.shape[0]
            
            rois = np.empty((num_images * self.max_faces, 5), dtype=np.float32, )

            index = 0
            for i, faces in enumerate(all_faces):
                for face in faces[:self.max_faces]:
                    rois[index, 1:] = face[0]
                    rois[index, 0] = i
                    index += 1
            rois_tensor = torch.tensor(rois[:index], device=self.device)
        return rois_tensor