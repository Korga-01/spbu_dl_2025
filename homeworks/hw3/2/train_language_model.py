"""
Скрипт для обучения LSTM языковой модели с BPE токенизатором
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Добавляем путь к токенизатору
sys.path.insert(0, str(Path(__file__).parent / '1'))
from bpe_tokenizer import BPETokenizer
from language_model import LSTMLanguageModel


class TextDataset(Dataset):
    """Датасет для языкового моделирования"""
    
    def __init__(self, texts: list, tokenizer: BPETokenizer, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.vocab.get(tokenizer.config.pad_token, 1)
        
        # Токенизируем все тексты
        print("Токенизация текстов...")
        all_token_ids = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            all_token_ids.extend(token_ids)
        
        self.token_ids = all_token_ids
        print(f"Всего токенов: {len(self.token_ids):,}")
    
    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_length)
    
    def __getitem__(self, idx):
        # Берем последовательность длины seq_length + 1 (для input и target)
        sequence = self.token_ids[idx:idx + self.seq_length + 1]
        
        # Input: все кроме последнего токена
        input_ids = sequence[:-1]
        # Target: все кроме первого токена (сдвиг на 1)
        target_ids = sequence[1:]
        
        # Паддинг если нужно
        if len(input_ids) < self.seq_length:
            padding = [self.pad_token_id] * (self.seq_length - len(input_ids))
            input_ids = input_ids + padding
            target_ids = target_ids + padding
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Reshape для loss
        logits = logits.reshape(-1, logits.size(-1))
        target_ids = target_ids.reshape(-1)
        
        # Вычисляем loss (игнорируем padding токены)
        loss = criterion(logits, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, dataloader, criterion, device):
    """Оценка модели"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, _ = model(input_ids)
            logits = logits.reshape(-1, logits.size(-1))
            target_ids = target_ids.reshape(-1)
            
            loss = criterion(logits, target_ids)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def load_texts(filepath: str) -> list:
    """Загружает тексты из файла"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        paragraphs = content.split('\n\n')
        texts.extend([p.strip() for p in paragraphs if p.strip()])
    return texts


def main():
    parser = argparse.ArgumentParser(description='Обучение LSTM языковой модели')
    parser.add_argument('--train', type=str, required=True, help='Путь к обучающему корпусу')
    parser.add_argument('--tokenizer', type=str, required=True, help='Путь к обученному токенизатору')
    parser.add_argument('--output', type=str, default='language_model.pt', help='Путь для сохранения модели')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=32, help='Размер батча')
    parser.add_argument('--seq-length', type=int, default=128, help='Длина последовательности')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Размерность эмбеддингов')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Размерность скрытого состояния')
    parser.add_argument('--num-layers', type=int, default=2, help='Количество слоев LSTM')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Скорость обучения')
    parser.add_argument('--device', type=str, default='auto', help='Устройство (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Определяем устройство
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Используемое устройство: {device}")
    
    # Загрузка токенизатора
    print(f"Загрузка токенизатора из {args.tokenizer}...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    vocab_size = len(tokenizer.vocab)
    print(f"Размер словаря: {vocab_size}")
    
    # Загрузка данных
    print(f"Загрузка данных из {args.train}...")
    texts = load_texts(args.train)
    print(f"Загружено {len(texts)} текстовых фрагментов")
    
    # Создание датасета
    dataset = TextDataset(texts, tokenizer, seq_length=args.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Для Windows
    )
    
    print(f"Размер датасета: {len(dataset)} последовательностей")
    print(f"Количество батчей: {len(dataloader)}")
    
    # Создание модели
    print("\nСоздание модели...")
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_token_id=tokenizer.vocab.get(tokenizer.config.pad_token, 1)
    )
    model.to(device)
    
    # Подсчет параметров
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Количество параметров: {num_params:,}")
    
    # Оптимизатор и loss
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.get(tokenizer.config.pad_token, 1))
    
    # Обучение
    print(f"\nНачало обучения на {args.epochs} эпох...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Эпоха {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        start_time = time.time()
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        epoch_time = time.time() - start_time
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Время эпохи: {epoch_time:.2f} секунд")
        
        # Обновляем learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(train_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate изменен: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Сохраняем лучшую модель
        if train_loss < best_loss:
            best_loss = train_loss
            print(f"Новая лучшая модель! Сохранение в {args.output}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'epoch': epoch + 1,
                'loss': train_loss,
            }, args.output)
    
    print(f"\n{'='*60}")
    print("Обучение завершено!")
    print(f"Лучший loss: {best_loss:.4f}")
    print(f"Модель сохранена в {args.output}")


if __name__ == '__main__':
    main()

