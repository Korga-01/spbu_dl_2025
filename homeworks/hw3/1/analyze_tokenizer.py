"""
Скрипт для анализа эффективности токенизатора
"""

import argparse
import re
from collections import Counter
from bpe_tokenizer import BPETokenizer


def load_texts(filepath: str) -> list:
    """Загружает тексты из файла"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        paragraphs = content.split('\n\n')
        texts.extend([p.strip() for p in paragraphs if p.strip()])
    return texts


def count_words(text: str) -> int:
    """Подсчитывает количество слов в тексте"""
    # Для русского языка используем простой подсчет
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    return len(words)


def get_word_frequencies(texts: list) -> Counter:
    """Получает частоты слов в корпусе"""
    word_freq = Counter()
    for text in texts:
        words = re.findall(r'\b\w+\b', text, re.UNICODE)
        word_freq.update(words)
    return word_freq


def analyze_tokenizer(tokenizer: BPETokenizer, texts: list, domain_name: str = ""):
    """Анализирует эффективность токенизатора"""
    print(f"\n{'='*60}")
    print(f"Анализ токенизатора{' - ' + domain_name if domain_name else ''}")
    print(f"{'='*60}")
    
    total_tokens = 0
    total_bytes = 0
    total_chars = 0
    total_words = 0
    tokens_per_word_list = []
    word_token_counts = {}
    
    for text in texts:
        # Токенизация
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(token_ids)
        total_tokens += num_tokens
        
        # Байты и символы
        text_bytes = len(text.encode('utf-8'))
        text_chars = len(text)
        total_bytes += text_bytes
        total_chars += text_chars
        
        # Слова
        words = re.findall(r'\b\w+\b', text, re.UNICODE)
        num_words = len(words)
        total_words += num_words
        
        if num_words > 0:
            tokens_per_word = num_tokens / num_words
            tokens_per_word_list.append(tokens_per_word)
            
            # Подсчет токенов для каждого слова
            for word in words:
                word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                if word not in word_token_counts:
                    word_token_counts[word] = []
                word_token_counts[word].append(word_tokens)
    
    # Коэффициент сжатия
    compression_ratio_bytes = total_tokens / total_bytes if total_bytes > 0 else 0
    compression_ratio_chars = total_tokens / total_chars if total_chars > 0 else 0
    
    # Среднее количество токенов на слово
    avg_tokens_per_word = total_tokens / total_words if total_words > 0 else 0
    mean_tokens_per_word = sum(tokens_per_word_list) / len(tokens_per_word_list) if tokens_per_word_list else 0
    
    # Среднее количество токенов для топ 10% частотных слов
    word_freq = get_word_frequencies(texts)
    top_10_percent_count = max(1, len(word_freq) // 10)
    top_words = [word for word, _ in word_freq.most_common(top_10_percent_count)]
    
    top_10_tokens = []
    for word in top_words:
        if word in word_token_counts:
            top_10_tokens.extend(word_token_counts[word])
    
    avg_tokens_top_10 = sum(top_10_tokens) / len(top_10_tokens) if top_10_tokens else 0
    
    # Вывод результатов
    print(f"Общая статистика:")
    print(f"  Текстов обработано: {len(texts)}")
    print(f"  Всего токенов: {total_tokens:,}")
    print(f"  Всего байт: {total_bytes:,}")
    print(f"  Всего символов: {total_chars:,}")
    print(f"  Всего слов: {total_words:,}")
    print(f"\nКоэффициент сжатия:")
    print(f"  Токены / Байты: {compression_ratio_bytes:.4f}")
    print(f"  Токены / Символы: {compression_ratio_chars:.4f}")
    print(f"\nТокены на слово:")
    print(f"  Среднее (общее): {avg_tokens_per_word:.4f}")
    print(f"  Среднее (по текстам): {mean_tokens_per_word:.4f}")
    print(f"  Среднее для топ 10% частотных слов: {avg_tokens_top_10:.4f}")
    
    return {
        'compression_ratio_bytes': compression_ratio_bytes,
        'compression_ratio_chars': compression_ratio_chars,
        'avg_tokens_per_word': avg_tokens_per_word,
        'avg_tokens_top_10': avg_tokens_top_10,
        'total_tokens': total_tokens,
        'total_bytes': total_bytes,
        'total_chars': total_chars,
        'total_words': total_words
    }


def main():
    parser = argparse.ArgumentParser(description='Анализ эффективности токенизатора')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Путь к файлу с обученным токенизатором')
    parser.add_argument('--texts', type=str, nargs='+', required=True,
                       help='Путь(и) к файлам с текстами для анализа')
    parser.add_argument('--domains', type=str, nargs='+', default=None,
                       help='Названия доменов для каждого файла')
    
    args = parser.parse_args()
    
    # Загрузка токенизатора
    print(f"Загрузка токенизатора из {args.tokenizer}...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"Токенизатор загружен. Размер словаря: {len(tokenizer.vocab)}")
    
    # Анализ для каждого домена
    results = []
    domains = args.domains if args.domains else [f"Домен {i+1}" for i in range(len(args.texts))]
    
    for text_file, domain in zip(args.texts, domains):
        print(f"\nЗагрузка текстов из {text_file}...")
        texts = load_texts(text_file)
        result = analyze_tokenizer(tokenizer, texts, domain)
        result['domain'] = domain
        results.append(result)
    
    # Сравнение доменов
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Сравнение эффективности по доменам:")
        print(f"{'='*60}")
        print(f"{'Домен':<20} {'Сжатие (байты)':<18} {'Токены/слово':<15} {'Токены/слово (топ 10%)':<25}")
        print("-" * 80)
        for result in results:
            print(f"{result['domain']:<20} {result['compression_ratio_bytes']:<18.4f} "
                  f"{result['avg_tokens_per_word']:<15.4f} {result['avg_tokens_top_10']:<25.4f}")
        
        # Проверка различий
        compression_ratios = [r['compression_ratio_bytes'] for r in results]
        if max(compression_ratios) - min(compression_ratios) > 0.01:
            print("\n✓ Эффективность токенизации РАЗЛИЧАЕТСЯ между доменами")
        else:
            print("\n✗ Эффективность токенизации НЕ различается между доменами")


if __name__ == '__main__':
    main()

