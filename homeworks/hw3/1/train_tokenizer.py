"""
Скрипт для обучения BPE токенизатора на корпусе текстов
"""

import os
import argparse
from bpe_tokenizer import BPETokenizer, TokenizerConfig


def load_texts(filepath: str) -> list:
    """Загружает тексты из файла"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        # Разбиваем на предложения или абзацы
        # Для простоты разбиваем по двойным переносам строк
        paragraphs = content.split('\n\n')
        texts.extend([p.strip() for p in paragraphs if p.strip()])
    return texts


def main():
    parser = argparse.ArgumentParser(description='Обучение BPE токенизатора')
    parser.add_argument('--input', type=str, required=True,
                       help='Путь к файлу с обучающим корпусом')
    parser.add_argument('--output', type=str, default='tokenizer.json',
                       help='Путь для сохранения обученного токенизатора')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Размер словаря (по умолчанию: 10000)')
    
    args = parser.parse_args()
    
    print(f"Загрузка корпуса из {args.input}...")
    texts = load_texts(args.input)
    print(f"Загружено {len(texts)} текстовых фрагментов")
    
    # Создаем конфигурацию
    config = TokenizerConfig(vocab_size=args.vocab_size)
    
    # Создаем и обучаем токенизатор
    tokenizer = BPETokenizer(config)
    tokenizer.train(texts, vocab_size=args.vocab_size)
    
    # Сохраняем токенизатор
    print(f"Сохранение токенизатора в {args.output}...")
    tokenizer.save(args.output)
    print("Готово!")


if __name__ == '__main__':
    main()

