"""
Трансформации для изображений (Задание 3)
Все классы работают с PIL.Image (вход и выход, кроме ToTensor)
"""

import random
import math
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image
import numpy as np


class BaseTransform(ABC):
    """Базовый класс для всех трансформаций"""
    
    def __init__(self, p: float):
        if not 0 <= p <= 1:
            raise ValueError(f"p должно быть между 0 и 1, получено {p}")
        self.p = p
    
    @abstractmethod
    def apply_transform(self, img: Image.Image) -> Image.Image:
        """Применяет трансформацию к изображению"""
        pass
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Применяет трансформацию с вероятностью p"""
        if random.random() < self.p:
            return self.apply_transform(img)
        return img


class RandomCrop(BaseTransform):
    """Случайный кроп изображения"""
    
    def __init__(self, p: float, size: tuple[int, int] = (24, 24), padding: int = 0):
        super().__init__(p)
        self.size = size
        self.padding = padding
    
    def apply_transform(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        
        # Добавляем padding если нужно
        if self.padding > 0:
            img = img.crop((-self.padding, -self.padding, 
                          width + self.padding, height + self.padding))
            width, height = img.size
        
        # Вычисляем возможные координаты для кропа
        crop_width, crop_height = self.size
        if crop_width > width or crop_height > height:
            # Если размер кропа больше изображения, возвращаем исходное
            return img
        
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        
        return img.crop((left, top, left + crop_width, top + crop_height))


class RandomRotate(BaseTransform):
    """Случайный поворот изображения"""
    
    def __init__(self, p: float, degrees: float = 15.0):
        super().__init__(p)
        self.degrees = degrees
    
    def apply_transform(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, expand=False, fillcolor=(0, 0, 0) if img.mode == 'RGB' else 0)


class RandomZoom(BaseTransform):
    """Случайное масштабирование (зум) изображения"""
    
    def __init__(self, p: float, zoom_range: tuple[float, float] = (0.8, 1.2)):
        super().__init__(p)
        self.zoom_range = zoom_range
    
    def apply_transform(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
        
        # Новый размер
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Масштабируем
        img_resized = img.resize((new_width, new_height), Image.BILINEAR)
        
        # Если увеличили, делаем кроп до исходного размера
        if zoom_factor > 1.0:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img_resized = img_resized.crop((left, top, left + width, top + height))
        # Если уменьшили, делаем padding
        elif zoom_factor < 1.0:
            new_img = Image.new(img.mode, (width, height), 
                               color=(0, 0, 0) if img.mode == 'RGB' else 0)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_img.paste(img_resized, (left, top))
            img_resized = new_img
        
        return img_resized


class ToTensor:
    """Преобразует PIL.Image в torch.Tensor"""
    
    def __call__(self, img: Image.Image):
        """
        Преобразует PIL.Image в torch.Tensor
        Формат: (C, H, W), значения в [0, 1]
        """
        # Конвертируем в numpy array
        if img.mode == 'RGB':
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # HWC -> CHW
        elif img.mode == 'L' or img.mode == 'P':
            arr = np.array(img, dtype=np.float32)
            arr = arr[np.newaxis, :, :]  # Добавляем канал
        else:
            # Для других режимов конвертируем в RGB
            img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)
        
        # Нормализуем в [0, 1]
        arr = arr / 255.0
        
        return torch.from_numpy(arr)


class Compose:
    """Композиция трансформаций"""
    
    def __init__(self, transforms: List[BaseTransform | ToTensor]):
        self.transforms = transforms
    
    def __call__(self, img: Image.Image):
        """
        Применяет трансформации последовательно
        Возвращает PIL.Image или torch.Tensor (если есть ToTensor)
        """
        for transform in self.transforms:
            img = transform(img)
        return img

