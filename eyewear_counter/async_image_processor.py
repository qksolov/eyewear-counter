import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from .utils import DummyProgressBar
import nest_asyncio


class AsyncImageProcessor:
    """
    Конвейер для загрузки и обработки изображений:
    - Асинхронная загрузка (через aiohttp или с диска).
    - Параллельная обработка батчей (через ThreadPoolExecutor).
    """
    def __init__(
            self, 
            process_fn, 
            image_size=640,
            image_fit=True,
            batch_size=32, 
            max_workers=3
            ):
        self.image_size = image_size
        self.image_fit = image_fit
        self.process_fn = process_fn  # Функция для обработки батчей изображений, вызывается в consumer
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.queue = asyncio.Queue(maxsize=self.max_workers * self.batch_size)
        self.pbar = None
        self.from_disk = False # Флаг загрузки изображений с диска (True) или из интернета (False)
        self.errors_cnt = 0 # Счетчик ошибок при загрузке изображений.


    def preprocess_image(self, image):
        if self.image_fit:
            h, w, _ = image.shape # (H, W, 3) BGR
            if h != w:
                pad1 = abs(h - w) // 2
                pad2 = abs(h - w) - pad1
                if h < w:
                    image = cv2.copyMakeBorder(image, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                else:
                    image = cv2.copyMakeBorder(image, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        padding = int(0.05 * max(image.shape[:2]))
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        image = cv2.resize(cv2.UMat(image), (self.image_size, self.image_size)).get()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image    # (image_size, image_size, 3) RGB


    async def load_image(self, idx, image_url, session, semaphore):
        try:
            async with semaphore:
                async with session.get(image_url) as response:
                    img_bytes = await response.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # (H, W, 3) BGR
                    image = self.preprocess_image(image)           # (H, W, 3) RGB H = W = image_size
                    await self.queue.put((idx, image))
        except Exception as e:
            self.errors_cnt += 1
            self.pbar.write(f"Не удалось загрузить изображение {idx} ({image_url}): {e}")
            self.pbar.update(1)

        
    async def load_image_from_disk(self, idx, path, semaphore):
        try:
            async with semaphore:
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, cv2.imread, path)
                image = self.preprocess_image(image)
                await self.queue.put((idx, image))
        except Exception as e:
            self.errors_cnt += 1
            self.pbar.write(f"Не удалось загрузить изображение {idx} ({path}): {e}")
            self.pbar.update(1)
    

    async def producer(self, sources):
        semaphore = asyncio.Semaphore(self.max_workers * self.batch_size)
        if self.from_disk:
            tasks = [
                asyncio.create_task(self.load_image_from_disk(idx, path, semaphore))
                for idx, path in enumerate(sources)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                tasks = [
                    asyncio.create_task(self.load_image(idx, url, session, semaphore))
                    for idx, url in enumerate(sources)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

        for _ in range(self.max_workers):
            await self.queue.put((None, None))
        

    async def consumer(self, executor):
        batch = []
        batch_indices = []
        while True:
            idx, image = await self.queue.get()
            self.queue.task_done()

            if idx is None:
                if batch:
                    batch_np = np.stack(batch, axis=0)      # (B, H, W, 3), np.uint8, RGB
                    await asyncio.get_event_loop().run_in_executor(
                        executor, self.process_fn, batch_np, batch_indices
                    )
                    self.pbar.update(len(batch_indices))
                break 
            
            batch.append(image)
            batch_indices.append(idx)

            if len(batch) == self.batch_size:
                batch_np = np.stack(batch, axis=0)
                await asyncio.get_event_loop().run_in_executor(
                    executor, self.process_fn, batch_np, batch_indices
                )
                self.pbar.update(self.batch_size)
                batch = []
                batch_indices = []


    async def main(self, image_urls):
        executor = ThreadPoolExecutor(max_workers=self.max_workers)        
        consumers = [
            asyncio.create_task(self.consumer(executor))
            for _ in range(self.max_workers)
        ]
        
        await asyncio.gather(
            asyncio.create_task(self.producer(image_urls)),
            *consumers
            )

    
    def run(self, image_urls, pbar=None):
        nest_asyncio.apply()
        
        self.errors_cnt = 0
        if image_urls[0].startswith("http"):
            self.from_disk=False
        else:
            self.from_disk=True
        
        if pbar is None:
            pbar = DummyProgressBar
        self.pbar = pbar(total=len(image_urls), desc="Обработка",
                         ncols=100, dynamic_ncols=True, position=0)

        asyncio.run(self.main(image_urls))
        
        self.pbar.set_description("Обработка завершена")
        self.pbar.close()





