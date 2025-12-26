"""
Скрипт для выполнения всех анализов из задания
"""

import argparse
import os
import subprocess
import sys
import locale


def get_console_encoding():
    """Определяет кодировку консоли"""
    try:
        # Пытаемся получить кодировку из locale
        encoding = locale.getpreferredencoding()
        if encoding:
            return encoding
    except:
        pass
    
    # Для Windows используем cp1251 или utf-8
    if sys.platform == 'win32':
        return 'cp1251'
    return 'utf-8'


def run_command(cmd, description):
    """Выполняет команду и выводит результат"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Выполняется: {' '.join(cmd)}")
    print()
    
    # Определяем кодировку консоли
    console_encoding = get_console_encoding()
    
    # Пробуем разные варианты кодировки
    encodings_to_try = [console_encoding, 'utf-8', 'cp1251', 'latin1']
    
    for encoding in encodings_to_try:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding=encoding,
                errors='replace'  # Заменяем нечитаемые символы
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Неизвестная ошибка"
                print(f"ОШИБКА (код возврата {result.returncode}):")
                print(error_msg)
                return False
            
            # Выводим результат
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            return True
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"ОШИБКА при выполнении команды: {e}")
            return False
    
    # Если все кодировки не подошли, пробуем без указания кодировки
    try:
        result = subprocess.run(cmd, errors='replace')
        return result.returncode == 0
    except Exception as e:
        print(f"ОШИБКА при выполнении команды: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Выполнение всех анализов')
    parser.add_argument('--train', type=str, required=True,
                       help='Путь к файлу с обучающим корпусом')
    parser.add_argument('--test', type=str, required=True,
                       help='Путь к файлу с тестовым корпусом (не совпадающим с обучающим)')
    parser.add_argument('--pushkin', type=str, required=True,
                       help='Путь к файлу с корпусом стихотворений Пушкина')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Размер словаря (по умолчанию: 10000)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    # Создание директории для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
    
    # Шаг 1: Обучение токенизатора
    if not run_command(
        [sys.executable, 'train_tokenizer.py',
         '--input', args.train,
         '--output', tokenizer_path,
         '--vocab-size', str(args.vocab_size)],
        "ШАГ 1: Обучение BPE токенизатора"
    ):
        print("Ошибка при обучении токенизатора")
        return
    
    # Шаг 2: Анализ эффективности на тестовом корпусе
    if not run_command(
        [sys.executable, 'analyze_tokenizer.py',
         '--tokenizer', tokenizer_path,
         '--texts', args.test],
        "ШАГ 2: Анализ эффективности токенизации (тестовый корпус)"
    ):
        print("Ошибка при анализе эффективности")
        return
    
    # Шаг 3: Анализ зависимости от размера словаря
    vocab_analysis_path = os.path.join(args.output_dir, 'vocab_size_analysis.json')
    if not run_command(
        [sys.executable, 'vocab_size_analysis.py',
         '--train', args.train,
         '--test', args.test,
         '--output', vocab_analysis_path],
        "ШАГ 3: Анализ зависимости compression ratio от размера словаря"
    ):
        print("Ошибка при анализе зависимости от размера словаря")
        return
    
    # Шаг 4: Анализ неиспользованных токенов на корпусе Пушкина
    unused_tokens_path = os.path.join(args.output_dir, 'unused_tokens.txt')
    if not run_command(
        [sys.executable, 'unused_tokens_analysis.py',
         '--tokenizer', tokenizer_path,
         '--texts', args.pushkin,
         '--output', unused_tokens_path],
        "ШАГ 4: Анализ неиспользованных токенов (корпус Пушкина)"
    ):
        print("Ошибка при анализе неиспользованных токенов")
        return
    
    print(f"\n{'='*60}")
    print("ВСЕ АНАЛИЗЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    print(f"{'='*60}")
    print(f"\nРезультаты сохранены в директории: {args.output_dir}")
    print(f"  - Токенизатор: {tokenizer_path}")
    print(f"  - Анализ размера словаря: {vocab_analysis_path}")
    print(f"  - Неиспользованные токены: {unused_tokens_path}")


if __name__ == '__main__':
    main()


