from .detector import YoloDetector
from .classifier import EyewearClassifier
from .async_image_processor import AsyncImageProcessor

import torch
from torchvision.ops import roi_align
import numpy as np
from tqdm import tqdm


class EyewearCounter:
    """
    Класс для поиска лиц на изображениях и классификации наличия очков/солнцезащитных очков.
    
    Атрибуты:
        detector (BaseFaceDetector): детектор лиц, реализующий метод detect.
        classifier (BaseClassifier): классификатор лиц, реализующий метод predict.
        device (torch.device): устройство, на котором выполняются все тензорные вычисления.
    
    Attributes:
        detector (BaseFaceDetector): Детектор лиц, реализующий метод detect.
        classifier (BaseClassifier): Классификатор очков, реализующий метод predict.
        device (torch.device): 'cuda' или 'cpu'.
        results (torch.Tensor): Накопленные результаты классификации.
        save_samples (bool): Флаг сохранения примеров лиц.
        samples (dict): Примеры лиц для каждого класса.

    """
    def __init__(self, detector=None, classifier=None, device=None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if detector is None:
            detector = YoloDetector(device=device)
        self.detector = detector

        if classifier is None:
            classifier = EyewearClassifier(model_type="resnet18", device=device)
        self.classifier = classifier

        self.results = []

        self.save_samples = False
        self.samples = {}
        

    def process_batch(self, images_np, batch_indices):
        """
        Обрабатывает батч изображений: детекция лиц и классификация очков.

        Args:
            images_np (np.ndarray): Батч изображений shape=(B, H, W, 3), dtype=uint8, RGB.
            batch_indices (Sequence[int]): Индексы изображений в исходном списке.

        """
        with torch.no_grad():
            images_tensor = torch.tensor(images_np, dtype=torch.float32, device=self.device)
            images_tensor = images_tensor.permute(0, 3, 1, 2) / 255.0       # (B, 3, H, W), float32
            
            if self.detector.input_type == 'torch':
                rois = self.detector.detect(images_tensor)
            else: #"np"
                rois = self.detector.detect(images_np)
                
            batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.int32, device=self.device)
            expanded_batch_indices = batch_indices_tensor[rois[:, 0].int()]

            if rois.shape[0] > 0:
                faces_tensor = roi_align(images_tensor, rois, (224, 224))   # (num_faces, 3, 224, 224), float32
                probs = self.classifier.predict(faces_tensor)
                predicted_classes = torch.argmax(probs, dim=1)
                one_hot_classes = torch.nn.functional.one_hot(predicted_classes, 3).to(torch.int32)
                self.results.index_add_(0, expanded_batch_indices, one_hot_classes)

                # Сохранение примеров, если включено
                if self.save_samples:
                    for k in (0,1,2):
                        if k not in self.samples:
                            mask = (predicted_classes == k)
                            if mask.any():
                                idx = mask.nonzero()[0].item()
                                tensor = faces_tensor[idx]                      # (3, H, W) H = W = 224
                                image = tensor.permute(1, 2, 0).cpu().numpy()   # (H, W, 3)
                                image = (image * 255).astype(np.uint8)
                                self.samples[k] = image


    def run(self, sources,
             image_size=800, image_fit=True,batch_size=32, max_workers=3,
             max_faces=4, threshold=0.7,
             save_samples=False,
             progress_bar=tqdm):
        """
        Основной метод для обработки списка изображений. Вызывает AsyncImageProcessor.

        Args:
            sources (Sequence[str]): Список URL или путей к изображениям.
            image_size (int, optional): Размер для ресайза изображений. Default: 800.
            image_fit (bool, optional): Сохранять пропорции с padding. Default: True.
            batch_size (int, optional): Размер батча. Default: 32.
            max_workers (int, optional): Количество worker'ов. Default: 3.
            max_faces (int, optional): Макс. лиц на изображении. Default: 4.
            threshold (float, optional): Порог уверенности детектора. Default: 0.7.
            save_samples (bool, optional): Сохранять примеры лиц. Default: False.
            progress_bar (optional): Объект прогресс-бара с интерфейсом tqdm. Default: tqdm.

        Returns:
            tuple: (results, errors_cnt)
            - results (torch.Tensor): Тензор shape=(N, 3) с количеством обнаруженных лиц 
                                    каждого класса на изображение.
            - errors_cnt (int): Количество ошибок при загрузке изображения.
        """
                
        self.results = torch.zeros((len(sources), 3), dtype=torch.int32, device=self.device)
        self.save_samples = save_samples
        self.samples = {}
        self.detector.update_parameters(threshold=threshold, max_faces=max_faces)

        processor = AsyncImageProcessor(
            process_fn=self.process_batch,
            image_size=image_size, image_fit=image_fit,
            batch_size=batch_size, max_workers=max_workers
            )
        processor.run(sources, pbar=progress_bar)
        return self.results, processor.errors_cnt

