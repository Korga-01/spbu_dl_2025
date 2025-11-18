"""
Медианный фильтр для изображений (Задание 2*)
Используется только чистый torch
"""

import torch


def median_filter(image, kernel_size: int):
    """
    Применяет медианный фильтр к изображению.
    
    Args:
        image: тензор изображения (C, H, W) или (H, W)
        kernel_size: размер ядра фильтра (должен быть нечетным)
    
    Returns:
        отфильтрованное изображение того же размера
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size должен быть нечетным")
    
    # Добавляем канал если его нет
    if image.dim() == 2:
        image = image.unsqueeze(0)
    elif image.dim() == 3:
        pass
    else:
        raise ValueError(f"Ожидается 2D или 3D тензор, получен {image.dim()}D")
    
    C, H, W = image.shape
    pad = kernel_size // 2
    output = torch.zeros_like(image)
    
    # Добавляем padding (reflect mode вручную)
    padded = torch.zeros(C, H + 2*pad, W + 2*pad, dtype=image.dtype)
    padded[:, pad:H+pad, pad:W+pad] = image
    # Reflect padding
    padded[:, :pad, pad:W+pad] = image[:, pad:0:-1, :]  # top
    padded[:, H+pad:, pad:W+pad] = image[:, -2:-pad-2:-1, :]  # bottom
    padded[:, pad:H+pad, :pad] = image[:, :, pad:0:-1]  # left
    padded[:, pad:H+pad, W+pad:] = image[:, :, -2:-pad-2:-1]  # right
    # Углы
    padded[:, :pad, :pad] = image[:, pad:0:-1, pad:0:-1]  # top-left
    padded[:, :pad, W+pad:] = image[:, pad:0:-1, -2:-pad-2:-1]  # top-right
    padded[:, H+pad:, :pad] = image[:, -2:-pad-2:-1, pad:0:-1]  # bottom-left
    padded[:, H+pad:, W+pad:] = image[:, -2:-pad-2:-1, -2:-pad-2:-1]  # bottom-right
    
    for c in range(C):
        for i in range(H):
            for j in range(W):
                # Извлекаем окно
                window = padded[c, i:i+kernel_size, j:j+kernel_size]
                # Сортируем и берем медиану
                sorted_vals = torch.sort(window.flatten())[0]
                median_idx = len(sorted_vals) // 2
                output[c, i, j] = sorted_vals[median_idx]
    
    # Убираем канал если его не было
    if output.shape[0] == 1:
        output = output.squeeze(0)
    
    return output


if __name__ == "__main__":
    # Демонстрация на примере
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    # Создаем тестовое изображение с шумом
    test_img = torch.randn(1, 28, 28) * 0.3 + 0.5
    test_img = torch.clamp(test_img, 0, 1)
    
    # Применяем фильтры разных размеров
    kernels = [3, 5, 10]
    
    fig, axes = plt.subplots(1, len(kernels) + 1, figsize=(15, 4))
    axes[0].imshow(test_img.squeeze(), cmap='gray')
    axes[0].set_title('Оригинал')
    axes[0].axis('off')
    
    for idx, k in enumerate(kernels, 1):
        filtered = median_filter(test_img, k)
        axes[idx].imshow(filtered.squeeze(), cmap='gray')
        axes[idx].set_title(f'Медианный фильтр {k}x{k}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('spbu_dl_2025/homeworks/homework2/median_filter_demo.png', dpi=150)
    print("Демонстрация медианного фильтра сохранена в median_filter_demo.png")

