"""
Обучение CNN на FashionMNIST с трансформациями (Задание 4)
Применяет трансформации из задания 3, обучает модель, показывает графики
"""

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

from transforms import RandomCrop, RandomRotate, RandomZoom, ToTensor, Compose


class FashionMNISTWithTransforms(Dataset):
    """Обертка над FashionMNIST с кастомными трансформациями"""
    
    def __init__(self, root: str, train: bool = True, transform=None):
        # Загружаем без трансформаций torchvision
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=True,
            transform=None  # Без стандартных трансформаций
        )
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Применяем наши трансформации
        if self.transform:
            img = self.transform(img)
        
        return img, label


class SimpleCNN(nn.Module):
    """Простая CNN для FashionMNIST"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 7, 7)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 3, 3)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Оценка на тестовом наборе"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_model(config_name: str, train_transform, test_transform, epochs: int = 10, 
                batch_size: int = 64, lr: float = 0.001):
    """Обучает модель с заданными трансформациями"""
    
    # Создаем датасеты
    train_dataset = FashionMNISTWithTransforms(
        root='data',
        train=True,
        transform=train_transform
    )
    test_dataset = FashionMNISTWithTransforms(
        root='data',
        train=False,
        transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # История
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print(f"\n{'='*60}")
    print(f"Конфигурация: {config_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return history


def plot_results(all_histories: Dict[str, Dict], save_path: str = 'results.png'):
    """Строит графики для всех конфигураций"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss на train
    ax = axes[0, 0]
    for name, hist in all_histories.items():
        ax.plot(hist['train_loss'], label=name, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train Loss')
    ax.legend()
    ax.grid(True)
    
    # Accuracy на train
    ax = axes[0, 1]
    for name, hist in all_histories.items():
        ax.plot(hist['train_acc'], label=name, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Train Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Loss на test
    ax = axes[1, 0]
    for name, hist in all_histories.items():
        ax.plot(hist['test_loss'], label=name, marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss')
    ax.legend()
    ax.grid(True)
    
    # Accuracy на test
    ax = axes[1, 1]
    for name, hist in all_histories.items():
        ax.plot(hist['test_acc'], label=name, marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nГрафики сохранены в {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Фиксируем seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Базовый ToTensor для всех
    to_tensor = ToTensor()
    
    # Различные конфигурации трансформаций
    configs = {
        'Без трансформаций': (
            to_tensor,
            to_tensor
        ),
        'RandomCrop (p=0.5)': (
            Compose([RandomCrop(p=0.5, size=(24, 24)), to_tensor]),
            to_tensor
        ),
        'RandomRotate (p=0.5)': (
            Compose([RandomRotate(p=0.5, degrees=15.0), to_tensor]),
            to_tensor
        ),
        'RandomZoom (p=0.5)': (
            Compose([RandomZoom(p=0.5, zoom_range=(0.8, 1.2)), to_tensor]),
            to_tensor
        ),
        'Все трансформации (p=0.3)': (
            Compose([
                RandomCrop(p=0.3, size=(24, 24)),
                RandomRotate(p=0.3, degrees=15.0),
                RandomZoom(p=0.3, zoom_range=(0.8, 1.2)),
                to_tensor
            ]),
            to_tensor
        ),
        'Все трансформации (p=0.7)': (
            Compose([
                RandomCrop(p=0.7, size=(24, 24)),
                RandomRotate(p=0.7, degrees=15.0),
                RandomZoom(p=0.7, zoom_range=(0.8, 1.2)),
                to_tensor
            ]),
            to_tensor
        ),
    }
    
    all_histories = {}
    
    for config_name, (train_transform, test_transform) in configs.items():
        history = train_model(
            config_name,
            train_transform,
            test_transform,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        all_histories[config_name] = history
    
    # Строим графики
    plot_results(all_histories, save_path='spbu_dl_2025/homeworks/homework2/fashionmnist_results.png')
    
    # Печатаем финальные результаты
    print(f"\n{'='*60}")
    print("Финальные результаты:")
    print(f"{'='*60}")
    print(f"{'Конфигурация':<30} {'Train Acc':<12} {'Test Acc':<12}")
    print(f"{'-'*60}")
    for name, hist in all_histories.items():
        train_acc = hist['train_acc'][-1]
        test_acc = hist['test_acc'][-1]
        print(f"{name:<30} {train_acc:>10.2f}% {test_acc:>10.2f}%")


if __name__ == '__main__':
    main()

