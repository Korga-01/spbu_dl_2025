"""
Скрипт для генерации текста обученной языковой моделью
"""

import argparse
import sys
from pathlib import Path

import torch

# Добавляем путь к токенизатору
sys.path.insert(0, str(Path(__file__).parent / '1'))
from bpe_tokenizer import BPETokenizer
from language_model import LSTMLanguageModel


def load_model(checkpoint_path: str, tokenizer_path: str, device: str = 'cpu'):
    """Загружает модель и токенизатор"""
    # Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # Загрузка чекпоинта
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Создание модели
    model = LSTMLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        pad_token_id=tokenizer.vocab.get(tokenizer.config.pad_token, 1)
    )
    
    # Загрузка весов
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Генерация текста языковой моделью')
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели')
    parser.add_argument('--tokenizer', type=str, required=True, help='Путь к токенизатору')
    parser.add_argument('--prompt', type=str, default='', help='Начальный промпт')
    parser.add_argument('--max-length', type=int, default=100, help='Максимальная длина генерации')
    parser.add_argument('--temperature', type=float, default=1.0, help='Температура сэмплирования')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k сэмплирование')
    parser.add_argument('--device', type=str, default='auto', help='Устройство (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Определяем устройство
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Загрузка модели из {args.model}...")
    model, tokenizer = load_model(args.model, args.tokenizer, device)
    print("Модель загружена!")
    
    # Генерация
    prompt = args.prompt if args.prompt else "Машинное обучение"
    print(f"\nПромпт: {prompt}")
    print(f"Генерация текста (max_length={args.max_length}, temperature={args.temperature})...")
    print("-" * 60)
    
    generated = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(generated)
    print("-" * 60)


if __name__ == '__main__':
    main()

