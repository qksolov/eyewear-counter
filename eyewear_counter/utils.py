from tqdm import tqdm
import tempfile
import requests


class DummyProgressBar:
    """
        Заглушка для прогресс-бара с интерфейсом, совместимым с tqdm.
        Используется по умолчанию, когда не требуется отображение прогресса.

        Все использования в AsyncImageProcessor:
        >>> pbar = DummyProgressBar(total=len(image_urls), desc="Обработка",
                                    ncols=100, dynamic_ncols=True, position=0)
        >>> pbar.update(10)
        >>> pbar.set_description("Обработка завершена")
        >>> pbar.write("Error")
        >>> pbar.close()
    """
    def __init__(self, total=0, desc="", ncols=100, dynamic_ncols=True, position=0):
        pass

    def update(self, n):
        """Обновление прогресса."""
        pass

    def set_description(self, desc):
        """Установку описания."""
        pass
    
    def write(text):
        """Выводит текст в консоль."""
        print(text)
    
    def close(self):
        """Завершение работы прогресс-бара."""
        pass


def load_pt_from_url(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    print(f'Downloading: "{url}"')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        with tqdm(
            total=total_size, 
            unit='B', 
            unit_scale=True            
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk: 
                    tmp.write(chunk)
                    pbar.update(len(chunk))
        
        tmp_path = tmp.name

    return tmp_path