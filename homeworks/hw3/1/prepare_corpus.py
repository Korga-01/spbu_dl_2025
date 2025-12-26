"""
Скрипт для подготовки корпусов из различных источников
"""

import os
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Очищает текст от лишних символов"""
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    # Удаляем множественные переносы строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def prepare_from_files(input_dir: str, output_file: str, min_length: int = 50):
    """
    Подготавливает корпус из нескольких файлов
    
    Args:
        input_dir: Директория с исходными файлами
        output_file: Путь к выходному файлу
        min_length: Минимальная длина текстового фрагмента
    """
    texts = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Директория {input_dir} не существует!")
        return
    
    # Обрабатываем все .txt файлы в директории
    for file_path in input_path.glob('*.txt'):
        print(f"Обработка {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Разбиваем на абзацы
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                para = clean_text(para)
                if len(para) >= min_length:
                    texts.append(para)
        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")
    
    # Сохраняем объединенный корпус
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))
    
    print(f"\nКорпус сохранен в {output_file}")
    print(f"Всего текстовых фрагментов: {len(texts)}")
    print(f"Общий размер: {os.path.getsize(output_file) / 1024 / 1024:.2f} МБ")


def split_corpus(input_file: str, train_file: str, test_file: str, train_ratio: float = 0.8):
    """
    Разделяет корпус на обучающий и тестовый
    
    Args:
        input_file: Исходный корпус
        train_file: Файл для обучающего корпуса
        test_file: Файл для тестового корпуса
        train_ratio: Доля обучающего корпуса (0.8 = 80%)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    split_idx = int(len(paragraphs) * train_ratio)
    train_texts = paragraphs[:split_idx]
    test_texts = paragraphs[split_idx:]
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_texts))
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_texts))
    
    print(f"Обучающий корпус: {len(train_texts)} фрагментов")
    print(f"Тестовый корпус: {len(test_texts)} фрагментов")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Подготовка корпусов для обучения токенизатора')
    parser.add_argument('--input-dir', type=str, help='Директория с исходными файлами')
    parser.add_argument('--output', type=str, help='Выходной файл')
    parser.add_argument('--split', action='store_true', help='Разделить на train/test')
    parser.add_argument('--train-file', type=str, default='train_corpus.txt', help='Файл для обучающего корпуса')
    parser.add_argument('--test-file', type=str, default='test_corpus.txt', help='Файл для тестового корпуса')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Доля обучающего корпуса')
    parser.add_argument('--min-length', type=int, default=50, help='Минимальная длина фрагмента')
    
    args = parser.parse_args()
    
    if args.input_dir and args.output:
        prepare_from_files(args.input_dir, args.output, args.min_length)
        
        if args.split:
            split_corpus(args.output, args.train_file, args.test_file, args.train_ratio)
    else:
        print("Использование:")
        print("  python prepare_corpus.py --input-dir data/raw --output corpus.txt")
        print("  python prepare_corpus.py --input-dir data/raw --output corpus.txt --split")


if __name__ == '__main__':
    main()

