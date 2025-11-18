# Домашнее задание 2: Свертки и базовые слои

Реализация базовых слоев без `torch.nn`, трансформаций для изображений и обучение CNN на FashionMNIST.

## Структура проекта

- `layers.py` - Реализация базовых слоев (BatchNorm, Linear, Dropout, ReLU, Sigmoid, Softmax)
- `median_filter.py` - Медианный фильтр для изображений (Задание 2*)
- `transforms.py` - Трансформации для изображений (BaseTransform, RandomCrop, RandomRotate, RandomZoom, ToTensor, Compose)
- `test_transforms.py` - Тесты для трансформаций (Задание 3*)
- `train_fashionmnist.py` - Обучение CNN на FashionMNIST с трансформациями (Задание 4)
- `train_fashionmnist_wandb.py` - Версия с логированием в Weights & Biases (Задание 5*)

## Установка зависимостей

```bash
pip install torch torchvision pillow matplotlib numpy
# Для wandb (опционально):
pip install wandb
```

## Запуск

### Задание 1-4: Базовые слои

Слои реализованы в `layers.py` и могут использоваться напрямую:

```python
from layers import BatchNorm, Linear, Dropout, ReLU

# Пример использования
linear = Linear(784, 128)
bn = BatchNorm(128)
dropout = Dropout(0.5)
relu = ReLU()

x = torch.randn(32, 784)
x = linear(x)
x = bn(x)
x = relu(x)
x = dropout(x)
```

### Задание 2*: Медианный фильтр

```bash
python spbu_dl_2025/homeworks/homework2/median_filter.py
```

Создаст демонстрацию медианного фильтра с ядрами 3x3, 5x5, 10x10.

### Задание 3: Трансформации

#### Тестирование трансформаций

```bash
python spbu_dl_2025/homeworks/homework2/test_transforms.py
```

#### Использование трансформаций

```python
from transforms import RandomCrop, RandomRotate, RandomZoom, ToTensor, Compose
from PIL import Image

# Создаем композицию
transform = Compose([
    RandomCrop(p=0.5, size=(24, 24)),
    RandomRotate(p=0.5, degrees=15.0),
    RandomZoom(p=0.5, zoom_range=(0.8, 1.2)),
    ToTensor()
])

# Применяем к изображению
img = Image.open('image.png')
tensor = transform(img)
```

### Задание 4: Обучение на FashionMNIST

```bash
python spbu_dl_2025/homeworks/homework2/train_fashionmnist.py --epochs 10 --batch_size 64 --lr 0.001
```

Скрипт обучит модель с различными конфигурациями трансформаций и создаст графики:
- Без трансформаций
- С RandomCrop (p=0.5)
- С RandomRotate (p=0.5)
- С RandomZoom (p=0.5)
- Со всеми трансформациями (p=0.3)
- Со всеми трансформациями (p=0.7)

Графики сохраняются в `fashionmnist_results.png`.

### Задание 5*: Weights & Biases

```bash
# Сначала залогиньтесь в wandb
wandb login

# Запустите обучение с логированием
python spbu_dl_2025/homeworks/homework2/train_fashionmnist_wandb.py --epochs 10 --batch_size 64 --lr 0.001
```

## Реализованные компоненты

### Базовые слои (Задания 1-4)

- ✅ **BatchNorm** - Batch Normalization с поддержкой training/eval режимов
- ✅ **Linear** - Полносвязный слой с инициализацией Xavier
- ✅ **Dropout** - Dropout с поддержкой training/eval режимов
- ✅ **ReLU** - ReLU активация
- ✅ **Sigmoid** - Sigmoid активация
- ✅ **Softmax** - Softmax активация (с численной стабильностью)

### Трансформации (Задание 3)

- ✅ **BaseTransform** - Базовый класс для всех трансформаций
- ✅ **RandomCrop** - Случайный кроп с вероятностью p
- ✅ **RandomRotate** - Случайный поворот с вероятностью p
- ✅ **RandomZoom** - Случайное масштабирование с вероятностью p
- ✅ **ToTensor** - Преобразование PIL.Image в torch.Tensor
- ✅ **Compose** - Композиция трансформаций

### Тесты (Задание 3*)

Тесты проверяют:
- Воспроизводимость (с фиксированным seed)
- Корректность размеров
- Граничные случаи
- Работу с различными типами изображений

## Результаты обучения

После запуска `train_fashionmnist.py` вы увидите:
- Сравнение различных конфигураций трансформаций
- Графики loss и accuracy на train и test
- Таблицу финальных результатов

Трансформации обычно улучшают обобщающую способность модели, особенно при высоких вероятностях применения.

