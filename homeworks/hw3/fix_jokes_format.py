"""
Скрипт для исправления формата файла с анекдотами
Формат должен быть: "id затравки пробел продолжение анекдота"
"""

def fix_jokes_format(prefixes_file="prefixes.txt", jokes_file="generated_jokes.txt", output_file="generated_jokes_fixed.txt"):
    """
    Исправляет формат файла с анекдотами
    Удаляет затравку из начала анекдота, оставляя только продолжение
    
    Args:
        prefixes_file: Файл с затравками
        jokes_file: Файл с сгенерированными анекдотами
        output_file: Выходной файл в правильном формате
    """
    # Загружаем затравки с их ID
    prefixes_with_ids = []
    with open(prefixes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(" ", 1)
                if len(parts) > 1 and parts[0].isdigit():
                    joke_id = parts[0]
                    prefix = parts[1]
                    prefixes_with_ids.append((joke_id, prefix))
                elif line and not line[0].isdigit():
                    # Если нет ID, пропускаем или используем порядковый номер
                    continue
    
    # Загружаем сгенерированные анекдоты
    jokes = []
    with open(jokes_file, "r", encoding="utf-8") as f:
        jokes = [line.strip() for line in f if line.strip()]
    
    print(f"Загружено {len(prefixes_with_ids)} затравок с ID")
    print(f"Загружено {len(jokes)} анекдотов")
    
    # Функция для удаления затравки из начала анекдота
    def remove_prefix(joke, prefix):
        """Удаляет затравку из начала анекдота, оставляя только продолжение"""
        joke_clean = joke.strip()
        prefix_clean = prefix.strip()
        
        # Метод 1: Точное совпадение в начале (самый простой случай)
        if joke_clean.startswith(prefix_clean):
            continuation = joke_clean[len(prefix_clean):].strip()
            # Убираем возможные запятые, точки, двоеточия и пробелы в начале
            continuation = continuation.lstrip(" ,.:;")
            # Если начинается с кавычки, оставляем её
            if continuation.startswith("'"):
                continuation = "'" + continuation[1:].lstrip()
            elif continuation.startswith('"'):
                continuation = '"' + continuation[1:].lstrip()
            return continuation
        
        # Метод 2: По словам - ищем первые слова затравки в начале анекдота
        prefix_words = prefix_clean.split()
        joke_words = joke_clean.split()
        
        if len(joke_words) >= len(prefix_words):
            # Проверяем, совпадают ли первые слова
            if joke_words[:len(prefix_words)] == prefix_words:
                continuation = " ".join(joke_words[len(prefix_words):])
                # Убираем возможные знаки препинания в начале
                continuation = continuation.lstrip(" ,.:;")
                # Если начинается с кавычки, оставляем её
                if continuation.startswith("'"):
                    continuation = "'" + continuation[1:].lstrip()
                elif continuation.startswith('"'):
                    continuation = '"' + continuation[1:].lstrip()
                return continuation
        
        # Метод 3: Ищем затравку как подстроку (даже если не в самом начале)
        # Это нужно, если модель добавила что-то перед затравкой
        idx = joke_clean.find(prefix_clean)
        if idx >= 0:
            # Нашли затравку, удаляем всё до неё включительно
            continuation = joke_clean[idx + len(prefix_clean):].strip()
            # Убираем возможные знаки препинания и пробелы в начале
            continuation = continuation.lstrip(" ,.:;")
            # Если начинается с кавычки, оставляем её
            if continuation.startswith("'"):
                continuation = "'" + continuation[1:].lstrip()
            elif continuation.startswith('"'):
                continuation = '"' + continuation[1:].lstrip()
            return continuation
        
        # Метод 4: Пытаемся найти затравку по ключевым словам
        # Берем последние 2-3 слова затравки и ищем их в начале анекдота
        if len(prefix_words) >= 2:
            last_words = prefix_words[-2:]  # Последние 2 слова
            if len(joke_words) >= 2:
                # Ищем эти слова в начале анекдота
                for i in range(len(joke_words) - 1):
                    if joke_words[i:i+2] == last_words:
                        # Нашли последние слова затравки, удаляем всё до них включительно
                        continuation = " ".join(joke_words[i+2:])
                        continuation = continuation.lstrip(" ,.:")
                        return continuation
        
        # Если ничего не найдено, возвращаем анекдот как есть
        # (лучше оставить полный анекдот, чем потерять часть)
        return joke_clean
    
    # Создаём выходной файл в правильном формате
    with open(output_file, "w", encoding="utf-8") as f:
        # Сопоставляем анекдоты с затравками по порядку
        for i, (joke_id, prefix) in enumerate(prefixes_with_ids):
            if i < len(jokes):
                joke = jokes[i].strip()
                # Удаляем затравку, оставляя только продолжение
                continuation = remove_prefix(joke, prefix)
                # Формат: "id пробел продолжение анекдота"
                f.write(f"{joke_id} {continuation}\n")
            else:
                # Если анекдотов меньше, чем затравок
                print(f"Предупреждение: нет анекдота для затравки {joke_id}")
    
    print(f"\nФайл исправлен и сохранён в {output_file}")
    print(f"Формат: 'id пробел продолжение анекдота' (без затравки)")
    
    # Показываем первые несколько строк
    print(f"\nПервые 5 строк выходного файла:")
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")
            else:
                break

if __name__ == "__main__":
    fix_jokes_format()

