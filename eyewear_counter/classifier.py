import torch
import torch.nn.functional as F
from torchvision import models
from pathlib import Path


def build_resnet18_classifier(num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenet_classifier(num_classes=3):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


# Конфигурация моделей
MODEL_CONFIGS = {
    'resnet18': {
        'builder': build_resnet18_classifier,
        'weights': 'resnet18_glasses.pt'
    },
    'mobilenet_v3_small': {
        'builder': build_mobilenet_classifier,
        'weights': 'mobilenet_v3_large_glasses.pt'
    }
}


class EyewearClassifier:
    """
    Классификатор очков на лицах. Поддерживает несколько архитектур моделей.
    
    Attributes:
        model (torch.nn.Module): Загруженная модель классификации
        device (torch.device): Устройство для вычислений (CPU/GPU)
        mean (torch.Tensor): Средние значения для нормализации (ImageNet)
        std (torch.Tensor): Стандартные отклонения для нормализации (ImageNet)
    """
    def __init__(self, 
                 model_type="resnet18", 
                 weights_path=None, 
                 device=None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"Неподдерживаемый тип модели: {model_type}. "
                f"Доступные: {list(MODEL_CONFIGS.keys())}"
            )
        
        config = MODEL_CONFIGS[model_type]
        
        if weights_path is None:
            weights_path = Path(__file__).parent.parent / "weights" / config['weights']
        else:
            weights_path = Path(weights_path)

        if not weights_path.is_file():
            raise FileNotFoundError(f"Файл весов модели не найден: {weights_path}")

        try:
            self.model = config['builder'](num_classes=3)
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель {model_type}: {e}")
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


    def predict(self, faces_tensor):
        """
        Args:
            faces_tensor (torch.Tensor): Батч изображений лиц shape=(N, 3, 224, 224), 
                                        dtype=float32, диапазон [0, 1], RGB
        Returns: 
            torch.Tensor: Вероятности классов shape=(N, 3),
                        где классы: 0-обычные, 1-нет очков, 2-солнцезащитные
        """
        if faces_tensor.ndim != 4 or faces_tensor.shape[1:] != (3, 224, 224):
            raise ValueError(
                f"Неверная форма входного тензора. Ожидается (N, 3, 224, 224), "
                f"получено {tuple(faces_tensor.shape)}. "
            )

        faces_tensor = (faces_tensor - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(faces_tensor.to(self.device))
            probs = F.softmax(logits, dim=1)
        return probs

