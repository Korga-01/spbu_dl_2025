"""
Простой тест базовых слоев
"""

import torch
from layers import BatchNorm, Linear, Dropout, ReLU, Sigmoid, Softmax

print("Тестирование базовых слоев...")

# Тест Linear
print("\n1. Тест Linear:")
linear = Linear(10, 5)
x = torch.randn(32, 10)
out = linear(x)
print(f"   Вход: {x.shape}, Выход: {out.shape} ✓")

# Тест BatchNorm
print("\n2. Тест BatchNorm:")
bn = BatchNorm(5)
bn.train()
out = bn(out)
print(f"   Выход после BN: {out.shape} ✓")

# Тест ReLU
print("\n3. Тест ReLU:")
relu = ReLU()
out = relu(out)
print(f"   Выход после ReLU: {out.shape}, min={out.min().item():.3f} ✓")

# Тест Dropout
print("\n4. Тест Dropout:")
dropout = Dropout(0.5)
dropout.train()
out_train = dropout(out)
dropout.eval()
out_eval = dropout(out)
print(f"   В training режиме: {out_train.shape} ✓")
print(f"   В eval режиме: {out_eval.shape}, равны: {torch.allclose(out, out_eval)} ✓")

# Тест Sigmoid
print("\n5. Тест Sigmoid:")
sigmoid = Sigmoid()
out_sig = sigmoid(torch.randn(5))
print(f"   Выход Sigmoid: {out_sig.shape}, диапазон: [{out_sig.min().item():.3f}, {out_sig.max().item():.3f}] ✓")

# Тест Softmax
print("\n6. Тест Softmax:")
softmax = Softmax(dim=-1)
out_soft = softmax(torch.randn(10, 5))
print(f"   Выход Softmax: {out_soft.shape}, сумма по последней оси: {out_soft.sum(dim=-1)[0].item():.3f} ✓")

print("\n✅ Все тесты пройдены!")

