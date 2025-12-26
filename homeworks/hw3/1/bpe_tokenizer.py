"""
BPE Tokenizer с претокенизацией и специальными токенами
Основан на паттернах из GPT-2 и BERT токенизаторов
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Конфигурация токенизатора"""
    vocab_size: int = 10000
    special_tokens: List[str] = None
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                self.unk_token,
                self.pad_token,
                self.bos_token,
                self.eos_token
            ]


class BPETokenizer:
    """
    BPE токенизатор с претокенизацией
    Претокенизация основана на GPT-2 стиле (regex-based)
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Претокенизация реализована в методе pretokenize()
        # (не используем regex паттерн с \p{}, так как стандартный re не поддерживает Unicode категории)
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Преобразует байты в читаемые unicode символы
        Основано на GPT-2 подходе
        """
        bs = (
            list(range(ord("!"), ord("~") + 1)) +
            list(range(ord("¡"), ord("¬") + 1)) +
            list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}
    
    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Получает пары соседних символов в слове"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> str:
        """Применяет BPE к токену"""
        if token in self.cache:
            return self.cache[token]
        
        word = list(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        
        result = ' '.join(word)
        self.cache[token] = result
        return result
    
    def pretokenize(self, text: str) -> List[str]:
        """
        Претокенизация текста
        Правильно обрабатывает русские буквы (кириллицу)
        """
        tokens = []
        current_token = []
        
        for char in text:
            # Проверяем, является ли символ буквой (включая кириллицу) или цифрой
            # isalpha() работает для всех Unicode букв, включая кириллицу
            # isdigit() для цифр
            if char.isalpha() or char.isdigit():
                current_token.append(char)
            elif char.isspace():
                # Пробелы - отдельные токены
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                # Сохраняем пробелы как отдельные токены
                tokens.append(char)
            else:
                # Пунктуация и другие символы
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                tokens.append(char)
        
        # Добавляем последний токен, если есть
        if current_token:
            tokens.append(''.join(current_token))
        
        # Фильтруем пустые токены, но сохраняем пробелы
        return [t for t in tokens if t.strip() or t == ' ']
    
    def _encode_token(self, token: str) -> List[int]:
        """Кодирует один токен в список индексов"""
        # Преобразуем в байты, затем в unicode
        token_bytes = token.encode('utf-8')
        token_unicode = ''.join(self.byte_encoder[b] for b in token_bytes)
        
        # Применяем BPE
        bpe_tokens = self._bpe(token_unicode).split()
        
        # Преобразуем в индексы
        token_ids = []
        for bpe_token in bpe_tokens:
            if bpe_token in self.vocab:
                token_ids.append(self.vocab[bpe_token])
            else:
                token_ids.append(self.vocab.get(self.config.unk_token, 0))
        
        return token_ids
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Кодирует текст в список индексов"""
        self.cache = {}
        
        # Претокенизация
        pretokens = self.pretokenize(text)
        
        # Кодирование
        token_ids = []
        
        if add_special_tokens and self.config.bos_token in self.vocab:
            token_ids.append(self.vocab[self.config.bos_token])
        
        for pretoken in pretokens:
            ids = self._encode_token(pretoken)
            token_ids.extend(ids)
        
        if add_special_tokens and self.config.eos_token in self.vocab:
            token_ids.append(self.vocab[self.config.eos_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Декодирует список индексов в текст"""
        if not self.id_to_token:
            # Если словарь не загружен, создаем обратный словарь
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Пропускаем специальные токены при декодировании
                if token not in self.config.special_tokens:
                    tokens.append(token)
        
        # Объединяем BPE токены
        text = ''.join(tokens)
        
        # Преобразуем из unicode обратно в байты
        try:
            text_bytes = bytearray([self.byte_decoder[c] for c in text])
            return text_bytes.decode('utf-8', errors='replace')
        except:
            return text
    
    def train(self, texts: List[str], vocab_size: int = None):
        """
        Обучает BPE токенизатор на корпусе текстов
        """
        vocab_size = vocab_size or self.config.vocab_size
        
        # Шаг 1: Претокенизация всех текстов
        print("Претокенизация корпуса...")
        pretokenized = []
        for text in texts:
            pretokens = self.pretokenize(text)
            pretokenized.extend(pretokens)
        
        # Шаг 2: Подсчет частот претокенов
        print("Подсчет частот...")
        word_freqs = Counter(pretokenized)
        
        # Шаг 3: Инициализация словаря с байтами и специальными токенами
        vocab = {}
        # Добавляем специальные токены
        for i, token in enumerate(self.config.special_tokens):
            vocab[token] = i
        
        # Добавляем байты
        idx = len(self.config.special_tokens)
        for byte, char in self.byte_encoder.items():
            vocab[char] = idx
            idx += 1
        
        # Шаг 4: Преобразуем слова в последовательности байтов
        print("Преобразование в байты...")
        splits = {}
        for word, freq in word_freqs.items():
            word_bytes = word.encode('utf-8')
            word_chars = ''.join(self.byte_encoder[b] for b in word_bytes)
            splits[word] = list(word_chars)
        
        # Шаг 5: Итеративное слияние пар
        print(f"Обучение BPE (целевой размер словаря: {vocab_size})...")
        merges = []
        num_merges = vocab_size - len(vocab)
        
        for i in range(num_merges):
            pairs = defaultdict(int)
            
            for word, word_list in splits.items():
                for j in range(len(word_list) - 1):
                    pair = (word_list[j], word_list[j + 1])
                    pairs[pair] += word_freqs[word]
            
            if not pairs:
                break
            
            # Выбираем наиболее частую пару
            best_pair = max(pairs, key=pairs.get)
            merges.append(best_pair)
            
            # Создаем новый токен
            new_token = ''.join(best_pair)
            vocab[new_token] = len(vocab)
            
            # Обновляем splits
            new_splits = {}
            for word, word_list in splits.items():
                new_word_list = []
                j = 0
                while j < len(word_list):
                    if (j < len(word_list) - 1 and 
                        word_list[j] == best_pair[0] and 
                        word_list[j + 1] == best_pair[1]):
                        new_word_list.append(new_token)
                        j += 2
                    else:
                        new_word_list.append(word_list[j])
                        j += 1
                new_splits[word] = new_word_list
            
            splits = new_splits
            
            if (i + 1) % 100 == 0:
                print(f"  Выполнено слияний: {i + 1}/{num_merges}")
        
        self.vocab = vocab
        self.merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        print(f"Обучение завершено. Размер словаря: {len(self.vocab)}")
        print(f"Количество слияний: {len(self.merges)}")
    
    def save(self, filepath: str):
        """Сохраняет токенизатор в файл"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'config': {
                'vocab_size': self.config.vocab_size,
                'special_tokens': self.config.special_tokens,
                'unk_token': self.config.unk_token,
                'pad_token': self.config.pad_token,
                'bos_token': self.config.bos_token,
                'eos_token': self.config.eos_token,
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Загружает токенизатор из файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = [tuple(m) for m in data['merges']]
        self.bpe_ranks = {tuple(pair): i for i, pair in enumerate(self.merges)}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        config_data = data['config']
        self.config = TokenizerConfig(
            vocab_size=config_data['vocab_size'],
            special_tokens=config_data['special_tokens'],
            unk_token=config_data['unk_token'],
            pad_token=config_data['pad_token'],
            bos_token=config_data['bos_token'],
            eos_token=config_data['eos_token'],
        )

