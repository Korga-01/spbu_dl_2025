# LSTM Языковая Модель с BPE Токенизатором

Простая LSTM модель для языкового моделирования, использующая ваш BPE токенизатор.

## Установка зависимостей

```bash
pip install torch>=2.0.0
```

Или установите все зависимости:
```bash
pip install -r 1/requirements.txt
```

## Структура

- `language_model.py` - Класс LSTM модели
- `train_language_model.py` - Скрипт для обучения модели
- `generate_text.py` - Скрипт для генерации текста

## Обучение модели

```bash
python train_language_model.py --train 1/train_corpus.txt --tokenizer 1/results/tokenizer.json --output language_model.pt --epochs 10 --batch-size 32 --seq-length 128 --embedding-dim 256 --hidden-dim 512 --num-layers 2 --learning-rate 0.001
```

### Параметры обучения:

- `--train`: Путь к обучающему корпусу
- `--tokenizer`: Путь к обученному BPE токенизатору
- `--output`: Путь для сохранения модели (по умолчанию: `language_model.pt`)
- `--epochs`: Количество эпох обучения (по умолчанию: 10)
- `--batch-size`: Размер батча (по умолчанию: 32)
- `--seq-length`: Длина последовательности (по умолчанию: 128)
- `--embedding-dim`: Размерность эмбеддингов (по умолчанию: 256)
- `--hidden-dim`: Размерность скрытого состояния LSTM (по умолчанию: 512)
- `--num-layers`: Количество слоев LSTM (по умолчанию: 2)
- `--learning-rate`: Скорость обучения (по умолчанию: 0.001)
- `--device`: Устройство (cpu/cuda/auto, по умолчанию: auto)

## Генерация текста

После обучения модели можно генерировать текст:

```bash
python generate_text.py --model language_model.pt --tokenizer 1/results/tokenizer.json --prompt "Машинное обучение" --max-length 100 --temperature 1.0 --top-k 50
```

### Параметры генерации:

- `--model`: Путь к обученной модели
- `--tokenizer`: Путь к токенизатору
- `--prompt`: Начальный текст для генерации (по умолчанию: пустая строка)
- `--max-length`: Максимальная длина генерируемого текста (по умолчанию: 100)
- `--temperature`: Температура сэмплирования (выше = более случайно, по умолчанию: 1.0)
- `--top-k`: Top-k сэмплирование (по умолчанию: 50, 0 = отключить)
- `--device`: Устройство (cpu/cuda/auto, по умолчанию: auto)

## Архитектура модели

Модель состоит из:
1. **Embedding слой**: Преобразует токены в векторы
2. **LSTM слои**: Обрабатывают последовательности
3. **Output слой**: Предсказывает следующий токен

### Параметры модели:

- **Embedding размер**: 256 (настраивается)
- **LSTM hidden размер**: 512 (настраивается)
- **Количество слоев**: 2 (настраивается)
- **Dropout**: 0.2

## Пример использования

```python
from language_model import LSTMLanguageModel
from bpe_tokenizer import BPETokenizer

# Загрузка токенизатора
tokenizer = BPETokenizer()
tokenizer.load('1/results/tokenizer.json')

# Создание модели
model = LSTMLanguageModel(
    vocab_size=len(tokenizer.vocab),
    embedding_dim=256,
    hidden_dim=512,
    num_layers=2
)

# Генерация текста
generated = model.generate(
    tokenizer=tokenizer,
    prompt="Привет, мир!",
    max_length=50,
    temperature=1.0
)
print(generated)
```

## Рекомендации

1. **Для обучения**: Используйте достаточно большой корпус (минимум 10-20 МБ текста)
2. **Размер батча**: Зависит от доступной памяти. Для GPU можно использовать 64-128
3. **Длина последовательности**: 128-256 токенов обычно достаточно
4. **Количество эпох**: Начните с 10, следите за loss, чтобы избежать переобучения
5. **Температура генерации**: 
   - 0.5-0.8 для более детерминированного текста
   - 1.0 для баланса
   - 1.2-1.5 для более креативного текста

## Производительность

Модель автоматически использует GPU, если доступен CUDA. Для проверки:

```python
import torch
print(torch.cuda.is_available())  # True если GPU доступен
```

## Сохранение и загрузка

Модель сохраняется в формате PyTorch checkpoint и включает:
- Веса модели
- Параметры архитектуры
- Номер эпохи и loss

Загрузка происходит автоматически в `generate_text.py`.

