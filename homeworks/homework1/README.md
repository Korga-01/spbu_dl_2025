# Домашнее задание: Задания 1–3 (без autograd/nn)

Ниже — краткое описание реализованных файлов и то, что они демонстрируют. Задание 4 уже выполнено отдельно.

- `task1_neuron.py`: один нейрон с сигмоидой, ручные градиенты, NLL/BCE. Демонстрирует GD, SGD и Mini-batch; печатает сравнительную таблицу (LR, эпохи, финальный NLL, веса, смещение).
- `task2_autograd.py`: мини‑autograd для скаляров на классе `Node` (операции: +, *, ReLU). Печатает таблицу значений и градиентов узлов `a, b, c, d, e` и отдельную строку для выражения `x*y + x`.
- `task3_optimizer.py`: кастомный Adam; сравнение Adam vs обычный GD на том же нейроне; печатает сводную таблицу.

## Как запустить (установка и команды)

Предварительно установите зависимости (нужен только `torch`):

```bash
pip install torch
```

Далее запустите любой файл напрямую (в каждом есть минимальное демо и табличный вывод):

```bash
# Задание 1: нейрон + GD/SGD/Mini-batch
python spbu_dl_2025/homeworks/task1_neuron.py

# Задание 2: autograd Node (операции + ReLU) и примеры
python spbu_dl_2025/homeworks/task2_autograd.py

# Задание 3: кастомный Adam + обучение нейрона
python spbu_dl_2025/homeworks/task3_optimizer.py
```

Для PowerShell (Windows) — те же команды в формате PowerShell:

```powershell
python .\spbu_dl_2025\homeworks\task1_neuron.py
python .\spbu_dl_2025\homeworks\task2_autograd.py
python .\spbu_dl_2025\homeworks\task3_optimizer.py
```

### Быстрый старт: запустить все демо подряд

Из корня репозитория (bash):

```bash
python spbu_dl_2025/homeworks/task1_neuron.py && \
python spbu_dl_2025/homeworks/task2_autograd.py && \
python spbu_dl_2025/homeworks/task3_optimizer.py
```

Или перейти в директорию с домашкой (bash):

```bash
cd spbu_dl_2025/homeworks
python task1_neuron.py && python task2_autograd.py && python task3_optimizer.py
```

Для PowerShell из корня:

```powershell
python .\spbu_dl_2025\homeworks\task1_neuron.py; python .\spbu_dl_2025\homeworks\task2_autograd.py; python .\spbu_dl_2025\homeworks\task3_optimizer.py
```

Или после перехода в директорию:

```powershell
Set-Location .\spbu_dl_2025\homeworks
python .\task1_neuron.py; python .\task2_autograd.py; python .\task3_optimizer.py
```

## Задание 1 — Один нейрон с сигмоидой и ручными градиентами

Файл: `task1_neuron.py`

- Функция `train_neuron` обучает один логистический нейрон с сигмоидной активацией по функции потерь NLL (эквивалент BCE) с ручным градиентом (без `autograd`, `nn`).
- Поддерживаемые режимы обучения (выводятся в таблице):
  - `gd` — полный батч (классический градиентный спуск),
  - `sgd` — стохастический (батч=1),
  - `minibatch` — мини‑батчи (укажите `batch_size`).
- Возвращает: обновлённые `weights`, `bias` и историю среднеэпохальных NLL (округлено до 4 знаков).
- Градиенты считаются аналитически: для BCE с сигмоидой справедливо `dL/dz = y_hat - y`, далее `dL/dw = X^T (y_hat - y) / B`, `dL/db = mean(y_hat - y)`.
- В `__main__` есть игрушечный пример и таблица с итогами по всем режимам.

Аргументы `train_neuron`:
- `features: List[List[float]]`, `labels: List[int]`,
- `initial_weights: List[float]`, `initial_bias: float`,
- `learning_rate: float`, `epochs: int`,
- `method: str = "gd"`, `batch_size: Optional[int] = None`, `shuffle: bool = True`, `seed: Optional[int] = None`.

Пример использования API:

```python
w, b, history = train_neuron(X, y, w0, b0, learning_rate=0.1, epochs=50, method="minibatch", batch_size=8)
```

## Задание 2 — Мини‑autograd (Node)

Файл: `task2_autograd.py`

- Класс `Node` хранит `data`, `grad`, замыкание `_backward`, множество родителей `_prev` и имя операции `_op`.
- Реализованы операции:
  - сложение `__add__`/`__radd__` (правило: `d/dx (x+y) = 1`),
  - умножение `__mul__`/`__rmul__` (правило: `d/dx (x*y) = y`, `d/dy (x*y) = x`),
  - `relu()` (градиент 1 на положительной части, 0 иначе).
- `backward()` строит топологический порядок (DFS по графу вычислений) и затем выполняет обратный проход, инициализируя `self.grad = 1.0` у целевого узла.
- В `__main__` печатается таблица значений/градиентов `a, b, c, d, e` и строка для `x*y + x`.

Ограничения/заметки:
- Поддерживаются скаляры (float). Векторные/матричные операции не реализованы, т.к. по условию требуется базовый скалярный autograd.

## Задание 3 — Кастомный оптимизатор Adam

Файл: `task3_optimizer.py`

- Класс `Adam` реализует минимальный Adam для списка параметров (`torch.Tensor`) с векторами моментов `m`, `v` и биас‑коррекцией.
- Функция `train_neuron_with_adam` использует ту же модель из Задания 1 (один нейрон, ручные градиенты `y_hat - y`), но обновляет параметры через `Adam.step(grads)`.
- Возвращает обновлённые веса/сдвиг и историю NLL по эпохам (округление до 4 знаков).
- В `__main__` — таблица сравнения Adam vs GD.

Аргументы `train_neuron_with_adam`:
- `features, labels, initial_weights, initial_bias, learning_rate, epochs, seed`.

## Подсказки и проверка

- Каждый файл запускается отдельно; в консоли появится таблица с ключевыми метриками.
- Если наблюдается переобучение/нестабильность на игрушечных данных — уменьшите `learning_rate` или увеличьте `epochs`.
- Для сравнения режимов в Задании 1 меняйте `method` и `batch_size`; ориентируйтесь на финальный NLL.

## Версии и окружение

- Python 3.9+ (тестировалось на 3.10).
- PyTorch 2.x (достаточно установить `torch`).




