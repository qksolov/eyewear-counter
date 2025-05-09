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


