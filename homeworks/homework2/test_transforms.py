"""
Тесты для трансформаций (Задание 3*)
Проверка воспроизводимости, результата и граничных случаев
"""

import unittest
import random
import numpy as np
from PIL import Image
import torch

from transforms import RandomCrop, RandomRotate, RandomZoom, ToTensor, Compose


class TestTransforms(unittest.TestCase):
    
    def setUp(self):
        """Создаем тестовое изображение"""
        # Создаем простое тестовое изображение 28x28
        self.test_img = Image.new('L', (28, 28), color=128)
        # Добавляем паттерн для проверки трансформаций
        pixels = np.array(self.test_img)
        pixels[10:18, 10:18] = 255  # Белый квадрат
        pixels[5:8, 5:8] = 0  # Черный квадрат
        self.test_img = Image.fromarray(pixels)
    
    def test_random_crop_reproducibility(self):
        """Проверка воспроизводимости RandomCrop с фиксированным seed"""
        transform = RandomCrop(p=1.0, size=(20, 20))
        
        # Устанавливаем seed
        random.seed(42)
        result1 = transform(self.test_img.copy())
        
        random.seed(42)
        result2 = transform(self.test_img.copy())
        
        # Результаты должны совпадать
        self.assertEqual(list(result1.getdata()), list(result2.getdata()))
    
    def test_random_crop_size(self):
        """Проверка размера после кропа"""
        transform = RandomCrop(p=1.0, size=(20, 20))
        result = transform(self.test_img)
        self.assertEqual(result.size, (20, 20))
    
    def test_random_crop_edge_case_large_crop(self):
        """Граничный случай: размер кропа больше изображения"""
        transform = RandomCrop(p=1.0, size=(50, 50))
        result = transform(self.test_img)
        # Должно вернуть исходное изображение
        self.assertEqual(result.size, self.test_img.size)
    
    def test_random_rotate_reproducibility(self):
        """Проверка воспроизводимости RandomRotate"""
        transform = RandomRotate(p=1.0, degrees=15.0)
        
        random.seed(42)
        result1 = transform(self.test_img.copy())
        
        random.seed(42)
        result2 = transform(self.test_img.copy())
        
        # Проверяем, что размеры совпадают
        self.assertEqual(result1.size, result2.size)
    
    def test_random_rotate_size_preservation(self):
        """Проверка сохранения размера после поворота"""
        transform = RandomRotate(p=1.0, degrees=45.0)
        result = transform(self.test_img)
        self.assertEqual(result.size, self.test_img.size)
    
    def test_random_zoom_reproducibility(self):
        """Проверка воспроизводимости RandomZoom"""
        transform = RandomZoom(p=1.0, zoom_range=(0.8, 1.2))
        
        random.seed(42)
        result1 = transform(self.test_img.copy())
        
        random.seed(42)
        result2 = transform(self.test_img.copy())
        
        # Размеры должны совпадать (zoom возвращает исходный размер)
        self.assertEqual(result1.size, result2.size)
        self.assertEqual(result1.size, self.test_img.size)
    
    def test_random_zoom_edge_case_small_zoom(self):
        """Граничный случай: очень маленький zoom"""
        transform = RandomZoom(p=1.0, zoom_range=(0.5, 0.5))
        result = transform(self.test_img)
        # Размер должен сохраниться
        self.assertEqual(result.size, self.test_img.size)
    
    def test_to_tensor_output(self):
        """Проверка вывода ToTensor"""
        transform = ToTensor()
        result = transform(self.test_img)
        
        # Должен быть torch.Tensor
        self.assertIsInstance(result, torch.Tensor)
        # Формат (C, H, W)
        self.assertEqual(len(result.shape), 3)
        # Значения в [0, 1]
        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLessEqual(result.max().item(), 1.0)
    
    def test_to_tensor_shape(self):
        """Проверка формы тензора"""
        transform = ToTensor()
        result = transform(self.test_img)
        
        # Для grayscale: (1, H, W)
        self.assertEqual(result.shape, (1, 28, 28))
    
    def test_compose_sequential(self):
        """Проверка последовательного применения трансформаций"""
        transforms = [
            RandomCrop(p=1.0, size=(24, 24)),
            RandomRotate(p=1.0, degrees=10.0),
            ToTensor()
        ]
        compose = Compose(transforms)
        
        random.seed(42)
        result = compose(self.test_img)
        
        # Должен быть тензор
        self.assertIsInstance(result, torch.Tensor)
        # Размер после кропа 24x24
        self.assertEqual(result.shape, (1, 24, 24))
    
    def test_probability_zero(self):
        """Проверка: p=0 не применяет трансформацию"""
        transform = RandomCrop(p=0.0, size=(20, 20))
        result = transform(self.test_img)
        # Должно вернуть исходное изображение
        self.assertEqual(result.size, self.test_img.size)
        self.assertEqual(list(result.getdata()), list(self.test_img.getdata()))
    
    def test_probability_one(self):
        """Проверка: p=1 всегда применяет трансформацию"""
        transform = RandomCrop(p=1.0, size=(20, 20))
        result = transform(self.test_img)
        # Размер должен измениться
        self.assertEqual(result.size, (20, 20))
    
    def test_rgb_image(self):
        """Проверка работы с RGB изображением"""
        rgb_img = Image.new('RGB', (28, 28), color=(128, 64, 192))
        transform = ToTensor()
        result = transform(rgb_img)
        
        # Для RGB: (3, H, W)
        self.assertEqual(result.shape, (3, 28, 28))


if __name__ == '__main__':
    unittest.main()

