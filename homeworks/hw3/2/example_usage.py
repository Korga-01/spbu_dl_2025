"""
Пример использования языковой модели
"""

import sys
from pathlib import Path

# Добавляем путь к токенизатору
sys.path.insert(0, str(Path(__file__).parent / '1'))

from bpe_tokenizer import BPETokenizer
from language_model import LSTMLanguageModel
import torch

def main():
    print("Пример использования языковой модели\n")
    
    # Загрузка токенизатора
    print("1. Загрузка токенизатора...")
    tokenizer = BPETokenizer()
    tokenizer_path = Path('1/results/tokenizer.json')
    
    if not tokenizer_path.exists():
        print(f"ОШИБКА: Токенизатор не найден по пути {tokenizer_path}")
        print("Сначала обучите токенизатор!")
        return
    
    tokenizer.load(str(tokenizer_path))
    print(f"   Размер словаря: {len(tokenizer.vocab)}")
    
    # Создание модели
    print("\n2. Создание модели...")
    model = LSTMLanguageModel(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Количество параметров: {num_params:,}")
    
    # Проверка генерации (без обучения, только структура)
    print("\n3. Проверка структуры модели...")
    device = 'cpu'
    model.to(device)
    
    # Тестовый промпт
    test_prompt = "Машинное обучение"
    print(f"   Тестовый промпт: '{test_prompt}'")
    
    # Токенизация
    token_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
    print(f"   Токенов в промпте: {len(token_ids)}")
    
    print("\n✓ Модель создана успешно!")
    print("\nДля обучения модели выполните:")
    print("  python train_language_model.py --train 1/train_corpus.txt --tokenizer 1/results/tokenizer.json")
    print("\nДля генерации текста выполните:")
    print("  python generate_text.py --model language_model.pt --tokenizer 1/results/tokenizer.json --prompt 'Ваш текст'")

if __name__ == '__main__':
    main()

