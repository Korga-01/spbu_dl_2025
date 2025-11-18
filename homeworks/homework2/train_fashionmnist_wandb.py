"""
Обучение CNN на FashionMNIST с логированием в Weights & Biases (Задание 5*)
"""

import argparse
import random
import numpy as np
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("wandb не установлен. Установите: pip install wandb")
    WANDB_AVAILABLE = False

from transforms import ToTensor, Compose, RandomCrop, RandomRotate, RandomZoom
from train_fashionmnist import (
    FashionMNISTWithTransforms,
    SimpleCNN,
    train_epoch,
    evaluate
)


def train_with_wandb(config_name: str, train_transform, test_transform, 
                     epochs: int = 10, batch_size: int = 64, lr: float = 0.001):
    """Обучает модель с логированием в wandb"""
    
    if not WANDB_AVAILABLE:
        print("wandb недоступен, используйте train_fashionmnist.py")
        return
    
    # Инициализация wandb
    wandb.init(
        project="fashionmnist-transforms",
        name=config_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "transform_config": config_name
        }
    )
    
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
    
    print(f"\n{'='*60}")
    print(f"Конфигурация: {config_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Логируем в wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='all', 
                       choices=['all', 'none', 'crop', 'rotate', 'zoom', 'all_low', 'all_high'])
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        return
    
    # Фиксируем seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    to_tensor = ToTensor()
    
    configs = {
        'none': ('Без трансформаций', to_tensor, to_tensor),
        'crop': ('RandomCrop (p=0.5)', 
                Compose([RandomCrop(p=0.5, size=(24, 24)), to_tensor]),
                to_tensor),
        'rotate': ('RandomRotate (p=0.5)',
                  Compose([RandomRotate(p=0.5, degrees=15.0), to_tensor]),
                  to_tensor),
        'zoom': ('RandomZoom (p=0.5)',
                Compose([RandomZoom(p=0.5, zoom_range=(0.8, 1.2)), to_tensor]),
                to_tensor),
        'all_low': ('Все трансформации (p=0.3)',
                   Compose([
                       RandomCrop(p=0.3, size=(24, 24)),
                       RandomRotate(p=0.3, degrees=15.0),
                       RandomZoom(p=0.3, zoom_range=(0.8, 1.2)),
                       to_tensor
                   ]),
                   to_tensor),
        'all_high': ('Все трансформации (p=0.7)',
                    Compose([
                        RandomCrop(p=0.7, size=(24, 24)),
                        RandomRotate(p=0.7, degrees=15.0),
                        RandomZoom(p=0.7, zoom_range=(0.8, 1.2)),
                        to_tensor
                    ]),
                    to_tensor),
    }
    
    if args.config == 'all':
        # Запускаем все конфигурации
        for config_key, (name, train_tf, test_tf) in configs.items():
            train_with_wandb(
                name,
                train_tf,
                test_tf,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
    else:
        # Запускаем одну конфигурацию
        name, train_tf, test_tf = configs[args.config]
        train_with_wandb(
            name,
            train_tf,
            test_tf,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )


if __name__ == '__main__':
    main()

