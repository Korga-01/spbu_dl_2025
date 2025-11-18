"""
Реализация базовых слоев без использования torch.nn
Задания 1-4: BatchNorm, Linear, Dropout, ReLU/Sigmoid/Softmax
"""

import torch
import math


class BatchNorm:
    """Batch Normalization слой (nn.BatchNorm1d для 2D тензоров)"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Параметры для обучения
        self.weight = torch.ones(num_features, requires_grad=True)
        self.bias = torch.zeros(num_features, requires_grad=True)
        
        # Статистика для инференса
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        
        self.training = True
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        x: (batch_size, num_features) или (batch_size, num_features, ...)
        """
        if self.training:
            # Вычисляем статистику по батчу
            if x.dim() == 2:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
            else:
                # Для многомерных тензоров усредняем по batch и пространственным измерениям
                dims = [0] + list(range(2, x.dim()))
                mean = x.mean(dim=dims, keepdim=True)
                var = x.var(dim=dims, keepdim=True, unbiased=False)
            
            # Обновляем running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.flatten()[:self.num_features]
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.flatten()[:self.num_features]
        else:
            # Используем running statistics
            if x.dim() == 2:
                mean = self.running_mean
                var = self.running_var
            else:
                mean = self.running_mean.view(1, -1, *([1] * (x.dim() - 2)))
                var = self.running_var.view(1, -1, *([1] * (x.dim() - 2)))
        
        # Нормализация
        if x.dim() == 2:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            # Применяем scale и shift
            out = self.weight * x_norm + self.bias
        else:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            weight = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
            bias = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
            out = weight * x_norm + bias
        
        return out
    
    def eval(self):
        self.training = False
    
    def train(self, mode: bool = True):
        self.training = mode


class Linear:
    """Линейный слой (nn.Linear)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        # Инициализация весов (Xavier/Glorot)
        k = 1.0 / math.sqrt(in_features)
        self.weight = torch.empty(out_features, in_features).uniform_(-k, k)
        self.weight.requires_grad = True
        
        if bias:
            self.bias = torch.zeros(out_features, requires_grad=True)
        else:
            self.bias = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        x: (batch_size, in_features) или (batch_size, ..., in_features)
        """
        # x @ weight^T + bias
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output


class Dropout:
    """Dropout слой (nn.Dropout)"""
    
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.training = True
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Генерируем маску
        mask = (torch.rand_like(x) > self.p).float()
        # Масштабируем для сохранения математического ожидания
        return x * mask / (1 - self.p)
    
    def eval(self):
        self.training = False
    
    def train(self, mode: bool = True):
        self.training = mode


class ReLU:
    """ReLU активация (nn.ReLU)"""
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return torch.clamp(x, min=0.0)


class Sigmoid:
    """Sigmoid активация (nn.Sigmoid)"""
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-x))


class Softmax:
    """Softmax активация (nn.Softmax)"""
    
    def __init__(self, dim: int = -1):
        self.dim = dim
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # Стабильная версия: вычитаем максимум для численной стабильности
        x_shifted = x - x.max(dim=self.dim, keepdim=True)[0]
        exp_x = torch.exp(x_shifted)
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)

