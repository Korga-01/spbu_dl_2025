"""
Скрипт для построения кривой: размер словаря vs compression ratio
"""

import argparse
import json
from bpe_tokenizer import BPETokenizer, TokenizerConfig


def load_texts(filepath: str) -> list:
    """Загружает тексты из файла"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        paragraphs = content.split('\n\n')
        texts.extend([p.strip() for p in paragraphs if p.strip()])
    return texts


def calculate_compression_ratio(tokenizer: BPETokenizer, texts: list) -> float:
    """Вычисляет коэффициент сжатия"""
    total_tokens = 0
    total_bytes = 0
    
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)
        total_bytes += len(text.encode('utf-8'))
    
    return total_tokens / total_bytes if total_bytes > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Анализ зависимости compression ratio от размера словаря')
    parser.add_argument('--train', type=str, required=True,
                       help='Путь к файлу с обучающим корпусом')
    parser.add_argument('--test', type=str, required=True,
                       help='Путь к файлу с тестовым корпусом')
    parser.add_argument('--output', type=str, default='vocab_size_analysis.json',
                       help='Путь для сохранения результатов')
    parser.add_argument('--vocab-sizes', type=int, nargs='+',
                       default=[1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000],
                       help='Размеры словаря для анализа')
    
    args = parser.parse_args()
    
    # Загрузка данных
    print(f"Загрузка обучающего корпуса из {args.train}...")
    train_texts = load_texts(args.train)
    print(f"Загружено {len(train_texts)} текстовых фрагментов")
    
    print(f"Загрузка тестового корпуса из {args.test}...")
    test_texts = load_texts(args.test)
    print(f"Загружено {len(test_texts)} текстовых фрагментов")
    
    results = []
    
    for vocab_size in args.vocab_sizes:
        print(f"\n{'='*60}")
        print(f"Обучение токенизатора с размером словаря: {vocab_size}")
        print(f"{'='*60}")
        
        # Обучение токенизатора
        config = TokenizerConfig(vocab_size=vocab_size)
        tokenizer = BPETokenizer(config)
        tokenizer.train(train_texts, vocab_size=vocab_size)
        
        # Вычисление compression ratio на тестовом корпусе
        compression_ratio = calculate_compression_ratio(tokenizer, test_texts)
        
        print(f"Compression ratio: {compression_ratio:.4f}")
        
        results.append({
            'vocab_size': vocab_size,
            'compression_ratio': compression_ratio,
            'actual_vocab_size': len(tokenizer.vocab)
        })
    
    # Сохранение результатов
    print(f"\nСохранение результатов в {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Вывод таблицы
    print(f"\n{'='*60}")
    print("Результаты:")
    print(f"{'='*60}")
    print(f"{'Размер словаря':<20} {'Compression Ratio':<20}")
    print("-" * 40)
    for result in results:
        print(f"{result['vocab_size']:<20} {result['compression_ratio']:<20.4f}")
    
    # Построение графика (если доступен matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        vocab_sizes = [r['vocab_size'] for r in results]
        compression_ratios = [r['compression_ratio'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(vocab_sizes, compression_ratios, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Размер словаря', fontsize=12)
        plt.ylabel('Compression Ratio (токены / байты)', fontsize=12)
        plt.title('Зависимость Compression Ratio от размера словаря', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = args.output.replace('.json', '.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nГрафик сохранен в {plot_file}")
    except ImportError:
        print("\nДля построения графика установите matplotlib: pip install matplotlib")


if __name__ == '__main__':
    main()


