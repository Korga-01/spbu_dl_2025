"""
Скрипт для определения неиспользованных токенов при токенизации корпуса Пушкина
"""

import argparse
import sys
from collections import Counter
from bpe_tokenizer import BPETokenizer


def safe_print(text):
    """Безопасный вывод текста с обработкой ошибок кодировки"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Если не получается вывести, пробуем заменить нечитаемые символы
        try:
            print(text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
        except:
            print(text.encode('ascii', errors='replace').decode('ascii', errors='replace'))


def load_texts(filepath: str) -> list:
    """Загружает тексты из файла"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        paragraphs = content.split('\n\n')
        texts.extend([p.strip() for p in paragraphs if p.strip()])
    return texts


def main():
    parser = argparse.ArgumentParser(description='Анализ неиспользованных токенов')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Путь к файлу с обученным токенизатором')
    parser.add_argument('--texts', type=str, required=True,
                       help='Путь к файлу с корпусом Пушкина')
    parser.add_argument('--output', type=str, default='unused_tokens.txt',
                       help='Путь для сохранения списка неиспользованных токенов')
    
    args = parser.parse_args()
    
    # Загрузка токенизатора
    safe_print(f"Загрузка токенизатора из {args.tokenizer}...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    safe_print(f"Токенизатор загружен. Размер словаря: {len(tokenizer.vocab)}")
    
    # Загрузка текстов
    safe_print(f"Загрузка текстов из {args.texts}...")
    texts = load_texts(args.texts)
    safe_print(f"Загружено {len(texts)} текстовых фрагментов")
    
    # Токенизация и подсчет использованных токенов
    safe_print("Токенизация корпуса...")
    used_tokens = set()
    token_counts = Counter()
    
    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            safe_print(f"  Обработано {i + 1}/{len(texts)} текстов...")
        
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        used_tokens.update(token_ids)
        token_counts.update(token_ids)
    
    # Определение неиспользованных токенов
    all_token_ids = set(tokenizer.vocab.values())
    unused_token_ids = all_token_ids - used_tokens
    
    # Преобразование в токены
    unused_tokens = []
    for token_id in unused_token_ids:
        if token_id in tokenizer.id_to_token:
            token = tokenizer.id_to_token[token_id]
            unused_tokens.append((token_id, token))
    
    unused_tokens.sort(key=lambda x: x[0])
    
    # Статистика
    total_tokens = len(tokenizer.vocab)
    used_count = len(used_tokens)
    unused_count = len(unused_token_ids)
    unused_percentage = (unused_count / total_tokens) * 100 if total_tokens > 0 else 0
    
    safe_print(f"\n{'='*60}")
    safe_print("Результаты анализа неиспользованных токенов:")
    safe_print(f"{'='*60}")
    safe_print(f"Всего токенов в словаре: {total_tokens:,}")
    safe_print(f"Использованных токенов: {used_count:,}")
    safe_print(f"Неиспользованных токенов: {unused_count:,}")
    safe_print(f"Процент неиспользованных: {unused_percentage:.2f}%")
    
    # Сохранение списка неиспользованных токенов
    safe_print(f"\nСохранение списка неиспользованных токенов в {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"Неиспользованные токены при токенизации корпуса Пушкина\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Всего токенов в словаре: {total_tokens:,}\n")
        f.write(f"Использованных: {used_count:,}\n")
        f.write(f"Неиспользованных: {unused_count:,} ({unused_percentage:.2f}%)\n\n")
        f.write(f"Список неиспользованных токенов:\n")
        f.write(f"{'-'*60}\n")
        
        for token_id, token in unused_tokens:
            # Пропускаем специальные токены, которые могут не использоваться
            if token not in tokenizer.config.special_tokens:
                f.write(f"ID: {token_id:6d} | Токен: {repr(token)}\n")
    
    # Топ использованных токенов
    safe_print(f"\nТоп-20 наиболее часто используемых токенов:")
    safe_print(f"{'-'*60}")
    for token_id, count in token_counts.most_common(20):
        if token_id in tokenizer.id_to_token:
            token = tokenizer.id_to_token[token_id]
            # Безопасное представление токена для вывода
            try:
                token_repr = repr(token)
            except:
                token_repr = token.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            safe_print(f"ID: {token_id:6d} | Токен: {token_repr:30s} | Частота: {count:,}")
    
    safe_print(f"\nАнализ завершен!")


if __name__ == '__main__':
    main()


